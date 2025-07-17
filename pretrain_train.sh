# !!! mask index is used for get force, which is 9 in our rep
# !!! mask rate is the ratio of mask, which is highly depend on the resolution of each sensor
# !!! current tactile is with respect to hand base frame
# !!! TACTILE_RAW_DATA_SCALE is set to 50, which is the scale of the tactile data
# !!! be careful to check the frame of point and force, make sure they are in same frame

python pretrain.py \
    --device=0 \
    --batch_size=512 \
    --num_workers=8 \
    --logging \
    --save_model \
    --max_epoch=100000 \
    --lr=1e-3 \
    --mask_rate=0.01 \
    --replace_rate=0.0 \
    --val_ratio=0.1 \
    --loss_fn=mse \
    --encoder=gat \
    --decoder=gat \
    --num_hidden=128 \
    --num_heads=4 \
    --in_drop=0.0 \
    --attn_drop=0.0 \
    --edge_type="four" \
    --mask_index=9 \
    --train_dataset="tactile_play_data_train" \
    --test_dataset="tactile_play_data_test" \
    --aug_type="rotation"\
    --exp_name="pretrain" \

# --resultant_type="force" \