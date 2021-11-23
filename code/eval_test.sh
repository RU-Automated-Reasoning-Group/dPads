CUDA_VISIBLE_DEVICES=0 python3 eval_test.py \
--graph_path "results/crim13_spec_seed0_depth4_nas_001/graph.p" \
--train_data data/crim13_processed/train_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 1 \
--num_labels 1

CUDA_VISIBLE_DEVICES=0 python3 eval_test.py \
--graph_path "results/fly_spec_seed0_depth4_nas_001/graph.p" \
--train_data data/fly_process/train_data.npy \
--test_data data/fly_process/test_data.npy \
--train_labels data/fly_process/train_label.npy \
--test_labels data/fly_process/test_label.npy \
--input_type "list" \
--output_type "atom" \
--input_size 53 \
--output_size 7 \
--num_labels 7 \
--normalize

CUDA_VISIBLE_DEVICES=0 python3 eval_test.py \
--graph_path "results/basket_spec_seed0_depth4_nas_001/graph.p" \
--train_data data/basketball_processed/train_basket_data.npy \
--test_data data/basketball_processed/test_basket_data.npy \
--train_labels data/basketball_processed/train_basket_labels.npy \
--test_labels data/basketball_processed/test_basket_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 22 \
--output_size 6 \
--num_labels 6 \
--normalize

CUDA_VISIBLE_DEVICES=0 python3 eval_test.py \
--graph_path "results/sk152_spec_seed0_depth4_nas_001/graph.p" \
--train_data data/sk152_process/train_data.npy \
--test_data data/sk152_process/test_data.npy \
--train_labels data/sk152_process/train_label.npy \
--test_labels data/sk152_process/test_label.npy \
--input_type "list" \
--output_type "atom" \
--input_size 75 \
--output_size 10 \
--num_labels 10 \
--normalize