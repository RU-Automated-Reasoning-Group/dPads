train_sk152_specific(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name sk152_spec_seed0 \
    --trial 1 \
    --train_data data/sk152_process/train_data.npy \
    --valid_data data/sk152_process/valid_data.npy \
    --test_data data/sk152_process/test_data.npy \
    --train_labels data/sk152_process/train_label.npy \
    --valid_labels data/sk152_process/valid_label.npy \
    --test_labels data/sk152_process/test_label.npy \
    --input_type "list" \
    --output_type "atom" \
    --input_size 75 \
    --output_size 10 \
    --num_labels 10 \
    --lossfxn "crossentropy" \
    --max_depth 4 \
    --learning_rate 0.01 \
    --search_learning_rate 0.01 \
    --train_valid_split 0.6 \
    --symbolic_epochs 6 \
    --neural_epochs 4 \
    --batch_size 200 \
    --random_seed 0 \
    --penalty 0.01 \
    --finetune_epoch 15 \
    --finetune_lr 0.01 \
    --node_share \
    --graph_unfold
}


train_sk152_specific 0