export CUDA_VISIBLE_DEVICES=1

seq_len=96
model_name=AttnEmbed

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

patch=(5 5 10 10 10 10 25 25 25 25)
stride=(1 5 1 2 5 10 1 5 10 25)

for n_head in 1 2 4
do
for ((i=0; i<${#patch[@]}; i++))
do
for conv_stride in 24 48 96
do
for alpha in 0.5 0.7 0.9
do
for pred_len in 96 192 336 720
do
for lr in 0.0001 0.0005 0.001 0.002
do
for a_layer in 1 2 3
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 6 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len ${patch[i]}\
      --stride ${stride[i]}\
      --conv_stride $conv_stride\
      --des 'Exp' \
      --train_epochs 20\
      --patience 5 \
      --n_head $n_head \
      --embd_type attention \
      --attn_layer $a_layer \
      --itr 1 --batch_size 128 --learning_rate $lr \
      --alpha $alpha
done
done
done
done
done
done
done