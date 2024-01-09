# pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
# pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
# pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
# pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
# pip install torch-geometric

# find all configs in configs/
# config=debug
config=$1
# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi g2l task on the same machine
master_addr=127.0.0.4
master_port=29504

# ------------------------ need not change -----------------------------------
config_file=configs/$config\.yaml
output_dir=outputs/$config
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# python3 -m torch.distributed.launch    --master_addr ${split_hosts[0]}  --master_port ${split_hosts[1]}\
CUDA_VISIBLE_DEVICES=$gpus python3.7 -m torch.distributed.launch  --nproc_per_node=$gpun \
--master_addr $master_addr  --master_port $master_port \
train_net.py --config-file $config_file OUTPUT_DIR $output_dir \


