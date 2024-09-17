# Pytorch In Action

A hands-on repository dedicated to building mainstream deep learning models from scratch using PyTorch


## Pytorch Basics

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| Tensor | [Tensor](./pytorch-basics/01_Tensor.ipynb) | 介绍了pytorch中多维数组`Tensor`的基本使用，包括了它的属性、创建方法以及它支持的常见的运算 |
| Automatic Differentiation | [AutoDiff](./pytorch-basics/02_AutoDiff.ipynb)     | 介绍了Pytorch中强的自动微分机制 |
| Dataset and Dataloader | [DataLoading](./pytorch-basics/03_DataLoading.ipynb)| 介绍了Pytorch中进行数据读取的接口以及自定义扩展的方法 |
| Modules | [Modules](./pytorch-basics/04_Modules.ipynb) | 介绍了定义深度学习中层、块、模型的基础类型Module的基本使用方法 |
| Save and Load | [SaveAndLoad](./pytorch-basics/05_SaveAndLoad.ipynb)| 介绍了pytorch中数据、模型、优化器等进行序列化保存与加载的机制  |


## Pytorch Advanced

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| Finetune | [Finetune](./pytorch-advanced/06_Finetune.ipynb) | 介绍了使用一些预训练好的模型在下游任务上进行微调的基本流程 |
| Distributed | [Distributed](./pytorch-advanced/07_Distributed.ipynb)| 介绍了Pytorch中分布式训练相关的功能支持，重点介绍了其中的分布式数据并行的原理 |
| Tensorboard | [Tensorboard](./pytorch-advanced/08_Tensorboard.ipynb)| 介绍了使用TensorBoard来记录训练过程中的一些Metrics |
| Auto Mixture Precision | [AutoMixPrecision](./pytorch-advanced/09_AutoMixPrecision.ipynb)| 介绍了如何开启自动混合精度来加速模型的训练 |
| Pytorch Lightning | [PytorchLightning](./pytorch-advanced/10_PytorchLightning.ipynb)| 介绍了使用Pytorch Lightning来模块化我们的训练代码 |


## Models

| Title | Notebooks | 说明 |
| :---: | :---: | --- |
| Convolution实现 | [convolution.ipynb](./models/convolution.ipynb) | 从零开始分别实现了单通道卷积、多输入输出通道卷积、以及各种其他高效实现方案，包括 im2col 等，同时通过代码演示了转置卷积的实现原理以及卷积的反向传播实现原理。 |
| RNN 实现 | [simple_rnn.ipynb](./models/simple_rnn.ipynb), [lstm.ipynb](./models/lstm.ipynb),[gru.ipynb](./models/gru.ipynb) | 分别实现了 SimpleRNN、LSTM、GRU的单 Cell 以及多层双向网络的实现。 |
| Transformer 架构实现 | [Transformer.ipynb](./models/transformer.ipynb)| 从零开始实现了 Token Embedding、位置编码、多头注意力模块等，并实现和验证了 EncodeLayer 以及 DecodeLayer 中的计算细节，比如 Padding mask 和 casual mask 的计算，在推理时自回归式的进行生成等 |
| 使用 Seq2Seq 模型来进行机器翻译 | [seq2seq.ipynb](./models/seq2seq.ipynb) | 演示了机器翻译数据集的预处理过程，通过 torch 中对于 Transformer 架构的支持，构建了一个 6 层的 Enocer-Decoder 架构的模型，实现了其正向的计算过程和整个模型的训练过程。｜


## Image Classification

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| ImageNet 训练 | [ImageNet](./imagenet/README.md) | 介绍了一个完整的Imagenet上进行图像分类的训练代码，包括了: 快照保存与恢复，多机多卡数据并行等功能、LR Scheduler等 |

## NLP

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| transformers库的整体介绍 | [transformers](./transformers/tutorials.ipynb) | 包括了Pipeline、Tokenizer、Model、Trainer、Dataset、Evaluate等 |
| Tokenizer  | [Tokenizer](./transformers/tokenizer.ipynb)    | 介绍了Tokenizer的详细功能 |
| Bert | [Bert模型](./transformers/bert_model.ipynb)   | 深入分析Bert模型的各个层和算子的实现细节 |
| DistillBert | [DistillBert](./transformers/distilbert_cls.ipynb)| 分析DistillBert模型的结构与前向过程 |
| GPT2 | [GPT2模型](./transformers/gpt2_model.ipynb)    | 解析了GPT2模型的结构与计算过程 |
| T5  | [T5模型](./transformers/t5_model.ipynb)       | 分析了Encoder-Deocder结构的T5模型的详细原理与计算流程 |
| 文本分类Finetune流程 | [Finetune](./transformers/text_cls_finetune.ipynb)| 基于小型BERT模型rbt3的文本分类的完整Finetune流程 |

