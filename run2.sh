task_name="cifar10_cnum100_dist8_skew0.5_seed0"
learning_rate=0.05
sigma=0.25
model="cnn"
fedalg="fedprob_v2"
CUDA_VISIBLE_DEVICES=0 python main.py --sigma $sigma --task $task_name --model $model --algorithm $fedalg --num_rounds 200 --num_epochs 5 --learning_rate $learning_rate --proportion 0.1 --batch_size 10 --eval_interval 1