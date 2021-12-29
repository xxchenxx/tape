CUDA_VISIBLE_DEVICES=0 nohup python examples/five_way_classification.py transformer five_way_classification --batch_size 8 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_0.pkl > 5_filtered.out &


for i in $(seq 1 10); do 
CUDA_VISIBLE_DEVICES=0 nohup python examples/five_way_classification.py transformer five_way_classification --batch_size 8 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_${i}.pkl > 5_fold_${i}.out 
done


for i in $(seq 0 9); do 
CUDA_VISIBLE_DEVICES=0 nohup python examples/three_way_classification.py transformer three_way_classification --batch_size 8 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_${i}.pkl > 3_fold_${i}.out 
done


CUDA_VISIBLE_DEVICES=0 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 0 > 0.out &

CUDA_VISIBLE_DEVICES=1 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 1 > 1.out &


CUDA_VISIBLE_DEVICES=0 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 2 > 2.out &

CUDA_VISIBLE_DEVICES=1 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 3 > 3.out &

CUDA_VISIBLE_DEVICES=2 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 4 > 4.out &

CUDA_VISIBLE_DEVICES=3 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 5 > 5.out &


CUDA_VISIBLE_DEVICES=0 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 6 > 6.out &

CUDA_VISIBLE_DEVICES=1 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 7 > 7.out &

CUDA_VISIBLE_DEVICES=4 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 8 > 8.out &

CUDA_VISIBLE_DEVICES=5 nohup python examples/large_five_way_classification.py transformer large_five_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 9 > 9.out &

# Three way classification

CUDA_VISIBLE_DEVICES=2 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 6 > 3_6.out &

CUDA_VISIBLE_DEVICES=3 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 7 > 3_7.out &

CUDA_VISIBLE_DEVICES=6 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 8 > 3_8.out &

CUDA_VISIBLE_DEVICES=7 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 9 > 3_9.out &