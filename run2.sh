task_name="cifar10_cnum100_dist12_skew0.5_seed0"
learning_rate=0.05
sigma=0.25
model="resnet18"
fedalg="fedprob"
num_rounds=600
CUDA_VISIBLE_DEVICES=1 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds $num_rounds --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1