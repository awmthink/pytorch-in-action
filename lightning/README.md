# Run on SLURM

## Fit

```bash
fastsrun -N 4 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task 4 \
    python main.py --trainer.devices=4 --trainer.num_nodes=4 --data.root_dir=/mnt/cache/share/images/
```

## Test

```bash
fastsrun -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task 4 \
    python main.py test --data.root_dir=/mnt/cache/share/images/ --ckpt_path=checkpoints/epoch=0-step=2503.ckpt
```

## S3 Support

```
pip install fsspec[s3]
```

1. `pytorch lightning`对于s3协议的支持，底层使用的是`fsspec`这个库，再下面一层是`s3fs`这个库
2. pytorch lightning的默认logger是tensorboard，它也支持s3协议，但用的不是`fsspec`这个库，所以需要额外的配置。

对于`fsspec`这个库，它建议的的环境变量为：

```bash
export FSSPEC_S3_ENDPOINT_URL=https://...
export FSSPEC_S3_KEY='miniokey...' # 可以替换为 AWS_ACCESS_KEY_ID
export FSSPEC_S3_SECRET='asecretkey...' # 可以替换为AWS_SECRET_ACCESS_KEY
```

而对于`tensorboard`，它使用的环境变量为：

```bash
export AWS_ACCESS_KEY_ID=******
export AWS_SECRET_ACCESS_KEY=*******
export S3_ENDPOINT=******
export S3_VERIFY_SSL=0    
export S3_USE_HTTPS=0 
```

## Todo

- [ ] Go through the baisc, imtermediate, advacned, expert tutorials
- [ ] Auto Resuming After Interruption
