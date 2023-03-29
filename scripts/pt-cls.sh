
PROJ_NAME=CrossPoint

pueue add -g ${PROJ_NAME} \
    python train_crosspoint.py \
    --model dgcnn --exp_name crosspoint_dgcnn_pt \
    --epochs 100 --lr 0.001 --batch_size 12 --test_batch_size 128 \
    --print_freq 500 --k 15 --world_size 6 --backend nccl --num_workers 1
