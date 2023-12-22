PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=4

# python -m torch.distributed.launch --nproc_per_node 4 main.py \
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_machines 1 --num_processes 1 main.py \
    --do_train \
    --train_file /home/yuxie/LLM_SFT_Traj/data/Traj_bj/test_no_padding_string.json \
    --validation_file /home/yuxie/LLM_SFT_Traj/data/Traj_bj/val_no_padding_string.json \
    --preprocessing_num_workers 1 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/yuxie/LLM_SFT_Traj/model \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --road_hidden_size 128 \
    --time_hidden_size 128 \
    --has_road_source true \
    --has_time_source false \
    --road_source_path /home/yuxie/public/Dataset/bj_raw_data/bj/road_embedding/road_embedding_HHGCLV3_bj_128.npy \
    --data Traj \
    --add_token_file_path /home/yuxie/LLM_SFT_Traj/model/add_vocab.json \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules query_key_value \
    --total_st_size 41817 \
    --st_hidden_size 128 \

