task_name="cifar10_cnum100_dist2_skew0.1_seed0"
learning_rate=0.02
sigma=0.25
model="cnn"
fedalg="fedprob_v2"
num_rounds=300
CUDA_VISIBLE_DEVICES=1 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds $num_rounds --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1