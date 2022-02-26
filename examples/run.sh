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

CUDA_VISIBLE_DEVICES=0 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 6 > 3_6.out &

CUDA_VISIBLE_DEVICES=1 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 7 > 3_7.out &

CUDA_VISIBLE_DEVICES=6 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 8 > 3_8.out &

CUDA_VISIBLE_DEVICES=7 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 9 > 3_9.out &

CUDA_VISIBLE_DEVICES=4 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 4 > 3_4.out &

CUDA_VISIBLE_DEVICES=5 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 5 > 3_5.out &


CUDA_VISIBLE_DEVICES=0 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 3 > 3_3.out &

CUDA_VISIBLE_DEVICES=1 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 2 > 3_2.out &


CUDA_VISIBLE_DEVICES=6 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 1 > 3_1.out &

CUDA_VISIBLE_DEVICES=7 nohup python examples/large_three_way_classification.py transformer large_three_way_classification --batch_size 4 --from_pretrained bert-base  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-fold 0 > 3_0.out &


CUDA_VISIBLE_DEVICES=1 nohup python examples/three_way_classification.py unirep three_way_classification --batch_size 1 --from_pretrained babbler-1900  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_1.pkl --tokenizer unirep  > small_3_1.out &

CUDA_VISIBLE_DEVICES=0 nohup python examples/three_way_classification.py unirep three_way_classification --batch_size 1 --from_pretrained babbler-1900  --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_0.pkl --tokenizer unirep  > small_3_0.out &


CUDA_VISIBLE_DEVICES=1 nohup python examples/three_way_classification.py resnet three_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_4.pkl --tokenizer iupac  > small_3_4_res.out &

CUDA_VISIBLE_DEVICES=2 nohup python examples/three_way_classification.py resnet three_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_5.pkl --tokenizer iupac  > small_3_5_res.out &

CUDA_VISIBLE_DEVICES=4 nohup python examples/three_way_classification.py resnet three_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_6.pkl --tokenizer iupac  > small_3_6_res.out &

CUDA_VISIBLE_DEVICES=5 nohup python examples/three_way_classification.py resnet three_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_7.pkl --tokenizer iupac  > small_3_7_res.out &

CUDA_VISIBLE_DEVICES=5 nohup python examples/three_way_classification.py resnet three_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_8.pkl --tokenizer iupac  > small_3_8_res.out &

CUDA_VISIBLE_DEVICES=2 nohup python examples/three_way_classification.py resnet three_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/3_list.pkl --data-fold thermo/3_fold_9.pkl --tokenizer iupac  > small_3_9_res.out &





CUDA_VISIBLE_DEVICES=1 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_0.pkl --tokenizer iupac  > small_5_0_res.out &

CUDA_VISIBLE_DEVICES=2 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_1.pkl --tokenizer iupac  > small_5_1_res.out &


CUDA_VISIBLE_DEVICES=3 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_2.pkl --tokenizer iupac  > small_5_2_res.out &

CUDA_VISIBLE_DEVICES=4 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_3.pkl --tokenizer iupac  > small_5_3_res.out &


CUDA_VISIBLE_DEVICES=1 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_4.pkl --tokenizer iupac  > small_5_4_res.out &

CUDA_VISIBLE_DEVICES=2 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_5.pkl --tokenizer iupac  > small_5_5_res.out &


CUDA_VISIBLE_DEVICES=3 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_6.pkl --tokenizer iupac  > small_5_6_res.out &

CUDA_VISIBLE_DEVICES=4 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_7.pkl --tokenizer iupac  > small_5_7_res.out &


CUDA_VISIBLE_DEVICES=3 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_8.pkl --tokenizer iupac  > small_5_8_res.out &

CUDA_VISIBLE_DEVICES=4 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_9.pkl --tokenizer iupac  > small_5_9_res.out &



CUDA_VISIBLE_DEVICES=4 nohup python examples/five_way_classification.py resnet five_way_classification --batch_size 2 --gradient_accumulation_steps 1   --num_train_epochs 20 --learning_rate 0.0001   --eval_freq 1  --save_freq 1 --warmup_steps 500 --data-label-set thermo/5_list.pkl --data-fold thermo/5_fold_9.pkl --tokenizer iupac  > small_5_9_res.out &

