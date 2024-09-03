# Pytorch In Action


## Pytorch Basics

- [Tensor](./01_Tensor.ipynb)：介绍了pytorch中多维数组`Tensor`的基本使用，包括了它的属性、创建方法以及它支持的常见的运算
- [Automatic Differentiation](./02_AutoDiff.ipynb)：介绍了Pytorch中强的自动微分机制
- [Dataset and Dataloader](./03_DataLoading.ipynb)：介绍了Pytorch中进行数据读取的接口以及自定义扩展的方法
- [Modules](./04_Modules.ipynb)：介绍了定义深度学习中层、块、模型的基础类型Module的基本使用方法
- [Save and Load](./05_SaveAndLoad.ipynb)：介绍了pytorch中数据、模型、优化器等进行序列化保存与加载的机制
- [Finetine](./06_Finetune.ipynb): 介绍了使用一些预训练好的模型在下游任务上进行微调的基本流程
- [Distributed](./07_Distributed.ipynb): 介绍了Pytorch中分布式训练相关的功能支持，重点介绍了其中的分布式数据并行的原理
- [Tensorboard](./08_Tensorboard.ipynb): 介绍了使用TensorBoard来记录训练过程中的一些Metrics
- [Auto Mixture Precision](./09_AutoMixPrecision.ipynb): 介绍了如何开启自动混合精度来加速模型的训练
- [Pytorch Lightning](./10_PytorchLightning.ipynb): 介绍了使用Pytorch Lightning来模块化我们的训练代码

## Deep learning modules

- [Convolution](./modules/convolution.ipynb)



## Image Classification

- [ImageNet](./imagenet/README.md): 介绍了一个完整的Imagenet上进行图像分类的训练代码，包括了: 快照保存与恢复，多机多卡数据并行等功能、LR Scheduler等。


## transformers

- [transformers库的整体介绍：包括了Pipeline、Tokenizer、Model、Trainer、Dataset、Evaluate等](./transformers/tutorials.ipynb)
- [Tokenizer的各个功能的详细介绍](./transformers/tokenizer.ipynb)
- [Bert模型深入分析各个层和算子的实现细节](./transformers/bert_model.ipynb)
- [DistillBert模型结构的分析与前向过程](./transformers/distilbert_cls.ipynb)
- [Decode模型的代表GPT2的结构与计算过程分析](./transformers/gpt2_model.ipynb)
- [Encoder-Deocder结构的代表T5模型的详细分析](./transformers/t5_model.ipynb)
- [基于小型BERT模型rbt3的文本分类的完整Finetune流程](./transformers/text_cls_finetune.ipynb)
