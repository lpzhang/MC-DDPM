#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
# export NCCL_P2P_DISABLE=1

# cuda and cudnn paths
# export PATH=$HOME/.usr/local/cuda-11.0/bin:$PATH
# export LD_LIBRARY_PATH=$HOME/.usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
# export CUDADIR=$HOME/.usr/local/cuda-11.0

# home directory
home_dir=/local_mount/space/cookie1/1/users/lz1009
echo --home_dir=${home_dir}

# conda and python envs
conda_env=${home_dir}/.usr/local/miniconda3/envs/pytorch1.13.1-cuda11.7
export LD_LIBRARY_PATH=${conda_env}/lib:$LD_LIBRARY_PATH
echo --conda-env=${conda_env}

# project directory
proj_dir=${home_dir}/dev/dimo
echo --proj_dir=${proj_dir}

# PYTHONPATH is an environment variable which you can set to 
# add additional directories where python will look for modules and packages.
export PYTHONPATH=${proj_dir}
echo --PYTHONPATH=${PYTHONPATH}

# project infos
data_path=${home_dir}/datasets/CMRxRecon/MICCAIChallenge2023/ChallengeData
func_path=${proj_dir}/cmr_examples/dimo_cmr.py
default_root_dir=./test
echo --data_path=${data_path}
echo --func_path=${func_path}
echo --default_root_dir=${default_root_dir}
echo --script=$0

# data params
# Cine/Mapping
task=Cine
trainlst=${proj_dir}/cmr_examples/datasplitlists_small/train.lst.seed42.s80s20
vallst=${proj_dir}/cmr_examples/datasplitlists_small/val.lst.seed42.s80s20
testlst=none
# masking
scontext=0
tfcontext=1
echo --task=${task}
echo --trainlst=${trainlst}
echo --vallst=${vallst}
echo --testlst=${testlst}
echo --scontext=${scontext}
echo --tfcontext=${tfcontext}

# network params
noise_schedule=cosine
beta_scale=0.5
gd_type=kspace
echo --noise_schedule=${noise_schedule}
echo --beta_scale=${beta_scale}
echo --gd_type=${gd_type}

### loss setttings
### optimizer
optimizer=AdamW # Adam/SGD/AdamW
lr=0.0001 # Adam learning rate
lr_step_size=40 # epoch at which to decrease learning rate
lr_gamma=0.1 # extent to which to decrease learning rate
weight_decay=0.0 # weight regularization strength
momentum=0.99 # SGD momentum factor
### others
save_keys=(scan_metric pred)

echo --optimizer=${optimizer}
echo --lr=${lr}
echo --lr_step_size=${lr_step_size}
echo --lr_gamma=${lr_gamma}
echo --weight_decay=${weight_decay}
echo --momentum=${momentum}
echo --save_keys=${save_keys}

# trainer params
mode=train
accelerator=gpu
devices=1
batch_size=1
num_workers=16
strategy=ddp
max_epochs=50

echo --mode=${mode}
echo --accelerator=${accelerator}
echo --devices=${devices}
echo --batch_size=${batch_size}
echo --num_workers=${num_workers}
echo --strategy=${strategy}
echo --max_epochs=${max_epochs}

# run script
${conda_env}/bin/python -u ${func_path} \
	--data_path ${data_path} --default_root_dir ${default_root_dir} \
	--task ${task} \
	--trainlst ${trainlst} --vallst ${vallst} --testlst ${testlst} \
	--scontext ${scontext} --tfcontext ${tfcontext} \
	--noise_schedule ${noise_schedule} --beta_scale ${beta_scale} \
	--gd_type ${gd_type} \
	--optimizer ${optimizer} \
	--lr ${lr} \
	--lr_step_size ${lr_step_size} \
	--lr_gamma ${lr_gamma} \
	--weight_decay ${weight_decay} \
	--momentum ${momentum} \
	--mode=${mode} --accelerator ${accelerator} --devices ${devices[*]} --batch_size=${batch_size} --num_workers ${num_workers} \
	--strategy ${strategy} --max_epochs ${max_epochs} \
	2>&1 | tee $0.log &
