python main_visual.py \
    --gpus='0'  \
    --lr=2e-5 \
    --batch_size=500 \
    --num_workers=12 \
    --max_epoch=60 \
    --shaking_prob=0.2 \
    --max_magnitude=0.07 \
    --test=False \
    --n_dimention=500 \
    --temperture=0.07 \
    --save_prefix='checkpoints/' \
    --dataset='/mnt/c/Users/amade/Documents/Development/lipreading_dataset/model_training/lrw/lrw_roi_63_99_191_227_size128_npy_gray_pkl_jpeg' \
    --weights='LipLearner_pretrained_model.pt' \
    
 