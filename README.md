# Pytorch tutorials


- [Tensor](./01_Tensor.ipynb)：介绍了pytorch中多维数组`Tensor`的基本使用，包括了它的属性、创建方法以及它支持的常见的运算
- [Automatic Differentiation](./02_AutoDiff.ipynb)：介绍了Pytorch中强的自动微分机制
- [Dataset and Dataloader](./03_DataLoading.ipynb)：介绍了Pytorch中进行数据读取的接口以及自定义扩展的方法
- [Modules](./04_Modules.ipynb)：介绍了定义深度学习中层、块、模型的基础类型Module的基本使用方法
- [Save and Load](./05_SaveAndLoad.ipynb)：介绍了pytorch中数据、模型、优化器等进行序列化保存与加载的机制
- [Finetine](./06_Finetune.ipynb): 介绍了使用一些预训练好的模型在下游任务上进行微调的基本流程


# todo

- [ ] 分布式训练ImageNet
    * https://pytorch.org/tutorials/distributed/home
    * https://github.com/pytorch/examples/tree/main/imagenet
    * https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
- [ ] Tensorboard
- [ ] 混合精度训练
- [ ] pytorch lighting
- [ ] torch script
- [ ] DataSet from Ceph & memcached: https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/TnJXc1Uj/page/KPkEPxAB
- [ ] 使用dali进行预处理加速等: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/use_cases/pytorch/resnet50/pytorch-resnet50.html