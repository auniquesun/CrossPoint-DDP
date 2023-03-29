from __future__ import print_function
import os
import datetime
import torch
import wandb

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from datasets.data import ModelNet40SVM, ScanObjectNNSVM
from models.dgcnn import DGCNN, DGCNN_partseg
from util import IOStream, AverageMeter, AccuracyMeter
from parser import args


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)

    
def cleanup():
    dist.destroy_process_group()


def train(rank):
    if rank == 0:
        os.environ["WANDB_BASE_URL"] = args.wb_url
        wandb.login(key=args.wb_key)
        wandb.init(project="CrossPoint", name=args.exp_name)

    setup(rank)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log', rank=rank)

    if 'ModelNet40' in args.ft_dataset:
        train_set = ModelNet40SVM(partition='train', num_points=args.num_ft_points)
        test_set = ModelNet40SVM(partition='test', num_points=args.num_ft_points)
    elif 'ScanObjectNN' in args.ft_dataset:
        train_set = ScanObjectNNSVM(partition='train', num_points=args.num_ft_points)
        test_set = ScanObjectNNSVM(partition='test', num_points=args.num_ft_points)
    else:
        raise NotImplementedError('Please choose dataset among [ModelNet40, ScanObjectNN]')
    
    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=rank)
    
    samples_per_gpu = args.batch_size // args.world_size
    test_samples_per_gpu = args.test_batch_size // args.world_size

    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=samples_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    test_loader = DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=test_samples_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    # in DGCNN and DGCNN_partseg, args.rank is used to specify the device where get_graph_feature() are executed
    args.rank = rank

    #Try to load models
    if args.model == 'dgcnn':
        point_model = DGCNN(args, cls=args.num_classes).to(rank)
    elif args.model == 'dgcnn_seg':
        point_model = DGCNN_partseg(args).to(rank)
    else:
        raise Exception("Not implemented")
        
    point_model_ddp = DDP(point_model, device_ids=[rank], find_unused_parameters=True)
        
    if args.resume:
        map_location = torch.device('cuda:%d' % rank)
        point_model_ddp.load_state_dict(
            torch.load(args.model_path, map_location=map_location),
            strict=False)   # it is necessary to set `strict=False` when finetuning
        io.cprint("Model Loaded !!")
        
    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(point_model_ddp.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(point_model_ddp.parameters(), lr=args.lr, weight_decay=1e-6)

    lr_scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = CrossEntropyLoss()
    scaler = GradScaler()
    
    ft_test_best_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        point_model_ddp.train()
        # require by DistributedSampler
        train_sampler.set_epoch(epoch)
        
        train_loss = AverageMeter()
        acc_meter = AccuracyMeter()

        train_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        io.cprint(f'[{train_start}] Start training epoch: ({epoch}/{args.epochs})')
        for i, (points,label) in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            
            with autocast():
                batch_size = points.shape[0]
                # points: [batch, 3, num_points]
                points = points.permute(0,2,1).to(rank)
                label = label.to(rank)

                # NOTE: here `loss` has already been averaged by `batch_size`
                pred_classes = point_model_ddp(points)
                # ------ ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                #           The `pred_classes` is expected to contain `raw`, `unnormalized` scores for each class
                #           label.squeeze() is a `batch_size`-Dimension class index tensor
                loss = criterion(pred_classes, label.squeeze())

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            train_loss.update(loss, n=batch_size)
            # x.argmax: low bound begins with 0
            pred_idx = pred_classes.argmax(dim=1)
            pos = acc_meter.pos_count(pred_idx, label.squeeze())
            acc_meter.update(pos, batch_size-pos, n=batch_size)
            
            if i % args.print_freq == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                outstr = '[%s] Epoch: %d/%d, Batch: %d/%d, Acc: %.6f, Loss: %.6f' % \
                    (time, epoch, args.epochs, i, len(train_loader), pos.item()/batch_size, train_loss.avg.item())
                io.cprint(outstr)
        
        ####################
        # Test
        ####################
        with torch.no_grad():
            ft_train_acc = acc_meter.num_pos.item() / acc_meter.total

            test_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            io.cprint('[%s] Start evaluating on the %s test set ...' % (test_start, args.ft_dataset))
            ft_test_loss, ft_test_acc = test(rank, test_loader, point_model_ddp, criterion)
            test_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            io.cprint(f'[{test_end}] Epoch: {epoch}/{args.epochs}, Acc: {ft_test_acc}, Loss: {ft_test_loss}')
        
        if rank == 0:
            if ft_test_acc > ft_test_best_acc:
                ft_test_best_acc = ft_test_acc
                io.cprint('==> Saving Best Model...')
                # For saving DDP model, 
                # refer https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/9
                save_file = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model.pth'.format(epoch=epoch))
                torch.save(point_model_ddp.module.state_dict(), save_file)

            wandb_log = {}
            wandb_log['Train Loss'] = train_loss.avg.item()
            wandb_log['Test Loss'] = ft_test_loss
            wandb_log['Train Accuracy'] = ft_train_acc
            wandb_log['Test Accuracy'] = ft_test_acc
            wandb_log['Best Test Accuracy'] = ft_test_best_acc
            wandb.log(wandb_log)
        
        # In PyTorch 1.1.0 and later, you should call lr_scheduler.step() after optimizer.step()
        lr_scheduler.step()
                
    if rank == 0:
        io.cprint('==> End of DDP Finetuning ...')
        io.cprint(f'==> Final best classification score {ft_test_best_acc}!')
        # We should call wandb.finish() explicitly in multi processes training, 
        # otherwise wandb will hang in this process
        wandb.finish()

    io.close()
    cleanup()


def test(rank, test_loader, point_model_ddp, criterion):
    point_model_ddp.eval()

    test_loss = AverageMeter()
    acc_meter = AccuracyMeter()  
    for (points, label) in test_loader:
        batch_size = points.shape[0]
        # points: [batch, 3, num_points]
        points = points.permute(0,2,1).to(rank)
        label = label.to(rank)

        # pred_classes: [batch, num_classes]
        pred_classes = point_model_ddp(points)
        loss = criterion(pred_classes, label.squeeze())
        test_loss.update(loss, n=batch_size)
        # pred_idx: a batch-Dimension tensor
        pred_idx = pred_classes.argmax(dim=1)
        pos = acc_meter.pos_count(pred_idx, label.squeeze())
        acc_meter.update(pos, batch_size-pos, n=batch_size)
    
    ft_test_loss = test_loss.avg.item()
    ft_test_acc = acc_meter.num_pos.item() / acc_meter.total

    return ft_test_loss, ft_test_acc


if __name__ == "__main__":
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log', rank=0)
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available() and torch.cuda.device_count() > 1
    torch.manual_seed(args.seed)
    
    if args.cuda:
        io.cprint('CUDA is available! Using %d GPUs for DDP training' % args.world_size)
        io.close()

        torch.cuda.manual_seed(args.seed)
        mp.spawn(train, nprocs=args.world_size)

    else:
        io.cprint('CUDA is unavailable! Exit')
        io.close()
