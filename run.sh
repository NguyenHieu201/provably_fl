# task_name="cifar100_cnum50_distcluster_skew0.5_seed0"
# learning_rate=0.05
# sigma=0.25
# model="resnet18"
# fedalg="fedprob"
# num_rounds=300
# session_name="25-5"
# CUDA_VISIBLE_DEVICES=1 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds $num_rounds --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1 --wandb --session_name $session_name


# task_name="cifar100_cnum50_distfeatured_skew0.5_seed0"
# learning_rate=0.05
# sigma=0.25
# model="resnet18"
# fedalg="fedprob"
# num_rounds=300
# session_name="25-5"
# CUDA_VISIBLE_DEVICES=0 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds $num_rounds --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1 --wandb --session_name $session_name


# task_name="cifar100_cnum50_distpareto_skew0.5_seed0"
# learning_rate=0.05
# sigma=0.25
# model="resnet18"
# fedalg="fedprob"
# num_rounds=300
# session_name="25-5"
# CUDA_VISIBLE_DEVICES=0 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds $num_rounds --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1 --wandb --session_name $session_name


task_name="cifar100_cnum50_disttrue_pareto_skew0.5_seed0"
learning_rate=0.05
sigma=0.25
model="resnet18"
fedalg="fedprob"
num_rounds=300
session_name="25-5"
CUDA_VISIBLE_DEVICES=2 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds $num_rounds --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1 --wandb --session_name $session_name