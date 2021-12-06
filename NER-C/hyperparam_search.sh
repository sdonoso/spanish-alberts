CUDA_VISIBLE_DEVICES=1
batch_sizes=(64 32 16)
learning_rates=(1e-5 2e-5 3e-5 5e-5)
epochs=(2 3 4)

for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for n_epoch in "${epochs[@]}"; do
            python run_ner.py \
            --model_name_or_path CenIA/distillbert-base-spanish-uncased \
            --max_seq_length 512 \
            --pad_to_max_length False \
            --do_lower_case True \
            --output_dir /data/sdonoso/all_results/ner-c/distillbeto/epochs_"$n_epoch"_bs_"$bs"_lr_"$lr" \
            --use_fast_tokenizer True \
            --language es \
            --train_language es \
            --do_train \
            --do_eval \
            --per_device_eval_batch_size "$bs" \
            --per_device_train_batch_size "$bs" \
            --learning_rate "$lr" \
            --num_train_epochs "$n_epoch" \
            --logging_dir /data/sdonoso/all_results/ner-c/distillbeto/epochs_"$n_epoch"_bs_"$bs"_lr_"$lr" \
            --seed 56 \
            --cache_dir /data/sdonoso/cache \
            --use_auth_token True \
            --evaluation_strategy steps \
            --save_steps 2000 \
            --eval_steps 2000 \
            --load_best_model_at_end True \
            --fp16 \
            ;
        done
    done
done