# Implement a Pytorch Trainer Step by Step

## Version 1：单机单卡模式

- [x] 在 ImageNet 模型上进行从零开始的训练与评估，支持命令行下输出训练与评估日志
- [x] 使用 `simple_parsing` 库来进行训练参数的命令行解析，并支持解析`.yaml/.json`格式的配置文件
- [x] 支持使用 `logging` 模块来进行日志输出
- [x] 使用 tqdm 来显示整个训练的进度，同时使用`tqdm.write`取代 logging 来打印 训练过程中的日志打印
- [x] 支持模型 checkpoint (模型参数、优化器状态、调度器状态、global_steps) 的保存，
    - [x] 支持通过参数来控制是否保存 safetensors 格式
    - [x] 仿照 HuggingFace 的实现，在 checkpoint 里保存随机数生成器的状态
- [x] 实验记录保存 TensorBoard
- [x] 支持混合精度训练、梯度裁剪
- [x] 支持梯度累积
- [x] 支持从检查点开始恢复训练


# Version 2：DDP 兼容模式

* 在分布式下需要显式的初始化发布式进程组，在程序结束时，需要 destroy 进程组资源。
* 模型需要调用 `DDP` 将原来的模型包装为 `DDP` 模型。
* 通过 `LOCAL_RANK` 获得当前应该使用的 `gpu_id`
* 数据加载时 `DataLoader` 需要指定 `DistributedSampler`
* 所有的 logger、进度条、Tensboard writer 都只在 main_process 中输出
* 在模型 evalute 的时候，需要将进程组中其他进程里的结果（loss 和 acc） allreduce 在一起。
* save checkpoint 只在全局主进程（rank == 0）中进行执行。对于 DDP 模型，在模型保存前要进行 unwrap； save checkpoint 时最好主动进行次 `dist.barrier`。
* 对于`DistributedSampler`的 DataLoader，由于随机数 seed 控制逻辑与 `RandomSampler` 不一致，所以需要分别处理。



