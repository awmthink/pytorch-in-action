# ImangeNet Trainer


## 单机多卡训练

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 imagenet_elastic_trainer.py imagenet --epochs=30 --resume --checkpoints-path=./checkpoints
```