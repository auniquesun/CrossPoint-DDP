# 2022.08.04
# ------ 不同数据集，记得改物体类别数啊，不然代码报错死活找不到原因

# pueue add -g crosspoint python ft_crosspoint.py --exp_name ft-crosspoint-MN-0 \
#         --ft_dataset ModelNet40 --num_ft_points 1024 \
#         --batch_size 600 --test_batch_size 600 --print_freq 50\
#         --num_workers 1 --num_classes 40 --resume --model dgcnn \
#         --model_path checkpoints/crosspoint_dgcnn_cls3/models/best_model.pth \
#         --lr 0.001 --epochs 300

pueue add -g crosspoint python ft_crosspoint.py --exp_name ft-crosspoint-MN-1 \
        --ft_dataset ModelNet40 --num_ft_points 1024 \
        --batch_size 600 --test_batch_size 600 --print_freq 50\
        --num_workers 1 --num_classes 40 --resume --model dgcnn \
        --model_path checkpoints/crosspoint_dgcnn_cls3/models/best_model.pth \
        --lr 0.001 --epochs 300