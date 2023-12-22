PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_machines 1 --num_processes 1 main.py \
    --do_train \
    --train_file /home/yuxie/LLM_SFT_Traj/data/AdvertiseGen/dev.json \
    --validation_file /home/yuxie/LLM_SFT_Traj/data/AdvertiseGen/dev.json \
    --preprocessing_num_workers 1 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/yuxie/public/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \

