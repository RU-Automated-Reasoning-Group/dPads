train_crim13_first_order(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name crim13_spec_seed4000 \
    --trial 1 \
    --train_data data/crim13_processed/train_crim13_data.npy \
    --valid_data data/crim13_processed/val_crim13_data.npy \
    --test_data data/crim13_processed/test_crim13_data.npy \
    --train_labels data/crim13_processed/train_crim13_labels.npy \
    --valid_labels data/crim13_processed/val_crim13_labels.npy \
    --test_labels data/crim13_processed/test_crim13_labels.npy \
    --input_type "list" \
    --output_type "list" \
    --input_size 19 \
    --output_size 1 \
    --num_labels 1 \
    --lossfxn "softf1" \
    --max_depth 4 \
    --frontier_capacity 8 \
    --learning_rate 0.001 \
    --search_learning_rate 0.001 \
    --train_valid_split 0.6 \
    --symbolic_epochs 6 \
    --neural_epochs 6 \
    --class_weights "1.0,1.5" \
    --cell_depth 2 \
    --batch_size 200 \
    --random_seed 4000 \
    --penalty 0.01 \
    --finetune_epoch 6 \
    --finetune_lr 0.001 \
    --node_share \
    --graph_unfold
}


train_crim13_first_order 1