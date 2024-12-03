# Implement a Pytorch Trainer Step by Step

## Version 1

- [x] 在 ImageNet 模型上进行从零开始的训练与评估，支持命令行下输出训练与评估日志
- [x] 使用 `simple_parsing` 库来进行训练参数的命令行解析，并支持解析`.yaml/.json`格式的配置文件
- [x] 支持使用 `logging` 模块来进行日志输出
- [x] 支持模型 checkpoint 的保存，并支持通过参数来控制是否保存 safetensors 格式
    - [ ] 支持保存多个 checkpoint，并设置 checkpoint 的保存策略（按 eval_metrics 或者 最后 N 个）


## TODO

- [ ] 实验记录保存 TensorBoard
- [ ] 支持混合精度训练
- [ ] 支持梯度累积
- [ ] 支持从检查点开始恢复训练

