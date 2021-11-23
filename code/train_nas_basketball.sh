train_basket_specific(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name basket_spec_seed0 \
    --trial 1 \
    --train_data data/basketball_processed/train_basket_data.npy \
    --valid_data data/basketball_processed/valid_basket_data.npy \
    --test_data data/basketball_processed/test_basket_data.npy \
    --train_labels data/basketball_processed/train_basket_labels.npy \
    --valid_labels data/basketball_processed/valid_basket_labels.npy \
    --test_labels data/basketball_processed/test_basket_labels.npy \
    --input_type "list" \
    --output_type "list" \
    --input_size 22 \
    --output_size 6 \
    --num_labels 6 \
    --lossfxn "crossentropy" \
    --max_depth 4 \
    --symbolic_epochs 6 \
    --neural_epochs 4 \
    --train_valid_split 0.6 \
    --batch_size 50 \
    --normalize \
    --random_seed 0 \
    --node_share \
    --graph_unfold
}

train_basket_specific 0