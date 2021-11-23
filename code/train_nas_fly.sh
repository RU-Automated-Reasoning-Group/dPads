train_fly_specific(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name fly_spec_seed0_test \
    --trial 1 \
    --train_data data/fly_process/train_data.npy \
    --valid_data data/fly_process/val_data.npy \
    --test_data data/fly_process/test_data.npy \
    --train_labels data/fly_process/train_label.npy \
    --valid_labels data/fly_process/val_label.npy \
    --test_labels data/fly_process/test_label.npy \
    --input_type "list" \
    --output_type "atom" \
    --input_size 53 \
    --output_size 7 \
    --num_labels 7 \
    --lossfxn "crossentropy" \
    --max_depth 4 \
    --learning_rate 0.0005 \
    --search_learning_rate 0.0005 \
    --train_valid_split 0.6 \
    --symbolic_epochs 6 \
    --neural_epochs 6 \
    --batch_size 200 \
    --normalize \
    --random_seed 0 \
    --finetune_epoch 25 \
    --finetune_lr 0.00025 \
    --node_share \
    --graph_unfold
}


train_fly_specific 0