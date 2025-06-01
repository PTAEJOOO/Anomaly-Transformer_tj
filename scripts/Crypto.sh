export CUDA_VISIBLE_DEVICES=0

python main.py \
    --mode train --dataset Crypto --data_path dataset/Crypto \
    --anormly_ratio 0.5 --num_epochs 30 --batch_size 256 \
    --input_c 2 --output_c 2

python main.py \
    --mode test --dataset Crypto --data_path dataset/Crypto \
    --anormly_ratio 0.5 --num_epochs 30 --batch_size 256 \
    --input_c 2 --output_c 2
