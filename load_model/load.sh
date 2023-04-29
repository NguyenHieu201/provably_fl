task_name="cifar10_cnum100_dist2_skew0.5_seed0"
learning_rate=0.05
sigma=0.5
CUDA_VISIBLE_DEVICES=1 python load_model.py --sigma $sigma --task $task_name --model resnet18 --algorithm fedprob --num_rounds 200 --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1
