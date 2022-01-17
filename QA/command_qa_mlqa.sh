python run_qa.py \
  --model_name_or_path CenIA/albert_base_spanish \
  --train_file /home/sdonoso/data/datasets/QA/mlqa/es_squad-translate-train-train-v1.1.json \
  --validation_file /home/sdonoso/data/datasets/QA/mlqa/es_squad-translate-train-dev-v1.1.json \
  --max_seq_length 384 \
  --output_dir /home/sdonoso/data/all_results/qa/mlqa/albert_tiny \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --weight_decay 0.01 \
  --num_train_epochs 4.0 \
  --warmup_ratio 0.1 \
  --doc_stride 128 \
  --logging_dir /home/sdonoso/data/all_results/qa/mlqa/albert_tiny \
  --save_strategy epoch \
  --seed 42 \
  --fp16 \
  --cache_dir /data/sdonoso/cache \
