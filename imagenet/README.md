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