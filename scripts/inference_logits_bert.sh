python train.py --data_dir  docred_data \
--transformer_type bert \
--model_name_or_path ./model/bert-base-cased \
--load_path checkpoints/bert-annotated-3.pt \
--train_file train_distant.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 1 \
--test_batch_size 1 \
--gradient_accumulation_steps 2 \
--evaluation_steps 500 \
--num_labels 4 \
--classifier_lr 3e-6 \
--learning_rate 1e-6 \
--max_grad_norm 2.0 \
--warmup_ratio 0.06 \
--num_train_epochs 10.0 \
--seed 66 \
--num_class 97
