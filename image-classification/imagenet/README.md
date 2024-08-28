# ImangeNet Trainer


## 单机多卡训练

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 imagenet_elastic_trainer.py imagenet --epochs=30 --resume --checkpoints-path=./checkpoints
```


## 多机Slurm

```bash
fastsrun -N 2 --ntasks-per-node=1 --gpus-per-task=2 \
        python ./imagenet_elastic_trainer.py /mnt/cache/share/images \
                --val-path /mnt/cache/yangyansheng/workspace/data \
                --dist-init-method slurm \
                --checkpoints-path=./checkpoints \
                --batch-size=256  --epochs=30 --resume 
```

## 多机torchrun

```bash
#!/bin/bash

#SBATCH -o 20230609%j.out
#SBATCH -e 20230609%j.err
#SBATCH --job-name=pytorch-trainer
#SBATCH --partition=sdcshare_v100_32g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --comment=wbsM-SC230825.001

export MST_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export OMP_NUM_THREADS=1

srun torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=${SLURM_GPUS_PER_TASK} \
              --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=$MST_ADDR:23445 \
              imagenet_elastic_trainer.py /mnt/cache/share/images/ \
              --val-path=/mnt/cache/yangyansheng/workspace/data/val \
              --batch-size=256  --epochs=30 --resume --checkpoints-path=./checkpoints
```


# PyTorch 分布式训练DDP

## 单结点多卡下手动启动多进程

```python
def train(rank):
  device = torch.device("cuda", rank % torch.cuda.device_count())
  pass

# 进程启动的函数的第一参数为rank，也就是进程的序号
# 该函数的执行是在每一个子进程中
def worker(rank, world_size):
  # 设置4个必须的环境变量
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29544"
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  
  dist.init_process_group(backend="nccl")
  train(rank)
  dist.destroy_process_group()
  

if __name__ == "__main__":
  # 根据卡数来设置进程的个数
  world_size = torch.cuda.device_count()
  # 拉起多个子进程
  torch.multiprocessing.mp.spawn(worker, args=(world_size,), join=True)
```

## 单结点多卡下使用torchrun

使用torchrun可以大大减化我们手动管理子进程的工作。

```python
def train(rank):
  device = torch.device("cuda", rank % torch.cuda.device_count())
  pass

def main():
  # 这里不需要再额外设置任何环境变量
  # 因为torchrun在启动相关进程时，会设置相应的4个环境变量
  dist.init_process_group(backend="nccl")
  train(dist.get_rank())
  dist.destroy_process_group()

if __name__ == "__main__":
  main()
```

torchrun的命令是：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py
```

## 多结点多卡下使用torchrun

多结点多卡下，对于训练的代码部分没有任何区别，我们只需要调整torchrun的启动配置即可。

```bash
torchrun --nnodes=2 \
				 --nproc_per_node=4 \
				 --rdzv_id=qwx12xids \
				 --rdzv_backend=c10d \
				 --rdzv_endpoint=$MST_ADDR:23445 \
				 train.py
```

### 在SLURM下自动获取相关变量的值

```bash
nnodes = ${SLURM_NNODES}
nproc_per_node = ${SLURM_GPUS_PER_TASK}
rdzv_id = ${SLURM_JOB_ID}
MST_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

### 在SenseCore下自动获取相关变量化值

```bash
nnodes = ${WORLD_SIZE}
nproc_per_node = 这个目前还获取不了
rdzv_id = 这个目前还获取不了
rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}
```

## 多结点单进程（多卡）

通过slurm分配的多个任务（task），一个任务中多块卡的情况。

在python中获取相关的环境变量，并设置ddp依赖的4个环境变量

```python
rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
node_list = os.environ["SLURM_NODELIST"]
addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["MASTER_ADDR"] = addr
os.environ["MASTER_PORT"] = args.master_port
```

## SenseCore下设置分布下架构

通过`-d(--distribued)` 来设置，该配置支持设置：`StandAlone|AllReduce|ParameterServer` 

* `-d StandAlone`: 单个结点，不会设置对应的4个环境变量
* `-d AllReduce` : 多结点DDP，会设置4个环境变量


SensCore下启动任务时，会自动设置以下环境变量：

```bash
MASTER_ADDR: master结点的地址
MASTER_PORT: master结点上用于DDP的端口地址
RANK: SensCore分配的结点（Docker Container）的序号，也就是lightning中的NODE_RANK
WORLD_SIZE: 一共分配的结点的个数
```