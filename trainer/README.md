# Implement a Pytorch Trainer Step by Step

## Version 1

- [x] 在 ImageNet 模型上进行从零开始的训练与评估，支持命令行下输出训练与评估日志
- [x] 使用 `simple_parsing` 库来进行训练参数的命令行解析，并支持解析`.yaml/.json`格式的配置文件


## TODO

- [ ] 支持混合精度训练
- [ ] 支持从检查点开始恢复训练
- [ ] 支持梯度累积
- [ ] 支持使用 logging 模块来进行日志输出
- [ ] 支持保存 val 数据集上的精度最好的 3 次 checkpoint