
PROJ_NAME=CrossPoint

pueue add -g ${PROJ_NAME} \
    python train_partseg.py --exp_name ft-partseg-1 \
    --model_path checkpoints/crosspoint_dgcnn_pt_seg/models/best_model.pth \
    --batch_size 60 --k 40 --test_batch_size 48 --epochs 300 \
    --num_workers 1
