export CUDA_VISIBLE_DEVICES=0

# python main.py \
#     --mode train --dataset ethereum --data_path dataset/Crypto \
#     --anormly_ratio 0.5 --num_epochs 30 --batch_size 256 \
#     --input_c 1 --output_c 1

python main.py \
    --mode test --dataset ethereum_log_diff --data_path dataset/Crypto \
    --anormly_ratio 0.5 --num_epochs 30 --batch_size 256 \
    --input_c 1 --output_c 1
