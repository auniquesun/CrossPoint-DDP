
PROJ_NAME=CrossPoint

pueue add -g ${PROJ_NAME} \
        python ft_crosspoint.py --exp_name ft-SO-1 \
        --ft_dataset ScanObjectNN --num_ft_points 1024 \
        --batch_size 600 --test_batch_size 600 --print_freq 50\
        --num_workers 1 --num_classes 15 --resume --model dgcnn \
        --model_path checkpoints/crosspoint_dgcnn_pt/models/best_model.pth \
        --lr 0.001 --epochs 300 --num_workers 1