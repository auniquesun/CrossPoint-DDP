
PROJ_NAME=CrossPoint

pueue add -g ${PROJ_NAME} \
    python train_crosspoint.py \
    --model dgcnn_seg --exp_name crosspoint_dgcnn_pt_seg \
    --epochs 100 --lr 0.001  --batch_size 20 --print_freq 200 --k 15 \
    --num_workers 1
