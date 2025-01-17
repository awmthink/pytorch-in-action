# Pytorch In Action

A hands-on repository dedicated to building modern deep learning layers, models and tasks from scratch using PyTorch.

## Pytorch Basics

| Notebooks | 说明 |
|:---:| ---|
| [Tensor](./pytorch-basics/Tensor.ipynb)| 介绍了pytorch中多维数组`Tensor`的基本使用，包括了它的属性、创建方法以及它支持的常见的运算 |
| [Automatic Differentiation](./pytorch-basics/AutoGrad.ipynb) | 介绍了Pytorch中强大的自动微分机制，并尝试剖析其中背后的机制 |
| [Modules](./pytorch-basics/Modules.ipynb) | 介绍了定义深度学习中层、块、模型的基础 类型Module的基本使用方法，并从源码角度分析了 Module 模块背后对于状态成员以及子 Module 的遍历机制 |
| [Dataset and Dataloader](./pytorch-basics/DataLoading.ipynb)| 介绍了Pytorch中进行数据读取的接口以及自定义扩展的方法，从源码的角色分析了 Dataloader 的运作机制 |
| [Save and Load](./pytorch-basics/SaveAndLoad.ipynb)| 介绍了pytorch中数据、模型、优化器等进行序列化保存与加载的机制  |


## Pytorch Advanced

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| Distributed | [Distributed](./pytorch-advanced/Distributed.ipynb)| 介绍了Pytorch中分布式训练相关的功能支持，重点介绍了其中的分布式数据并行的原理 |
| Tensorboard | [Tensorboard](./pytorch-advanced/Tensorboard.ipynb)| 介绍了使用TensorBoard来记录训练过程中的一些Metrics |
| Auto Mixture Precision | [AutoMixPrecision](./pytorch-advanced/AutoMixPrecision.ipynb)| 介绍了如何开启自动混合精度来加速模型的训练 |
| Pytorch Lightning | [PytorchLightning](./pytorch-advanced/10_PytorchLightning.ipynb)| 介绍了使用Pytorch Lightning来模块化我们的训练代码 |
| Pytorch Image Models | [Timm](./timm/tutuorials.ipynb)| 介绍了如何使用 timm 库来获取主流的视觉模型以及预训练权重，我们也可以基于这些模型进行扩展 |
| transformers库的整体介绍 | [transformers](./transformers/tutorials.ipynb) | 包括了Pipeline、Tokenizer、Model、Trainer、Dataset、Evaluate等 |


## OpenAI Triton Tutorials

参考 Triton 官方的 [Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)，加入一些注释与图解，方便更容易理解。

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| VectorAdd | [VectorAdd](./triton/vector-add.ipynb)| 入门性的介绍，如何使用 triton 来编码 CUDA 世界中的 Hello,world |
| Fused Softmax | [Fused Softmax](./triton/fused-softmax.ipynb)| 使用 Triton 来实现算子融合，并通过优化并发的线程块的数量，使得 Triton 版本的性能超过 Pytorch 原生算子，比原生手写实现快 4 倍 |
| Low Memory Dropout | [Low Memory Dropout](./triton/low-memory-dropout.ipynb)| 重点演示了 Triton 中的随机数生成，应用在 Dropout 上可以使得在 Backward 阶段进行 Mask 矩阵重计算，从而节省 Mask 所占用的显存。 |
| Matrix Multiplication | [Matrix Multiplication ](./triton/matrix-multiplication.ipynb)| 使用 Triton 来实现经典的矩阵乘法，通过优化输出矩阵的计算顺序，来优化 L2 缓存。 |
| Layer Norm | [LayerNorm](./triton/layer-norm.ipynb)|  |

## Models

### Fundamental Modules

| Notebooks | 说明 |
| :---: | --- |
| [Convolution 实现](./nnlayers/convolution.ipynb) | 从零开始分别实现了单通道卷积、多输入输出通道卷积、以及各种其他高效实现方案，包括 im2col 等，同时通过代码演示了转置卷积的实现原理以及卷积的反向传播实现原理。 |
| [Simple RNN 实现](./nnlayers/simple_rnn.ipynb)| 从零实现了单个的 RNN Cell、多层 RNN 网络、双向 RNN 网络 |
| [LSTM 实现](./nnlayers/lstm.ipynb)| 从零实现了单个的 LSTM Cell、多层 LSTM 网络、双向 LSTM网络 |
| [GRU 实现](./nnlayers/gru.ipynb) | 从零实现了单个的  GRU Cell、多层 GRU 网络、 GRU 网络 |
| [Transformer 架构实现](./nnlayers/transformer.ipynb)| 从零开始实现了 Token Embedding、位置编码、多头注意力模块等，并实现和验证了 EncodeLayer 以及 DecodeLayer 中的计算细节，比如 Padding mask 和 casual mask 的计算，在 infernece 模式下的自回归式的进行生成结果等。 |

### ConvNets & MLP

| Notebooks | 说明 |
| :---: | --- |
| [VGG](./models/convnets/VGG.ipynb)  | 介绍了经典的 CNN 架构 VGG 模型，包括 VGG 的网络结构的设计特点和设计动机，并通过代码从零构建了 VGG 的网络。 |
| [GoogLeNet](./models/convnets/GoogLeNet.ipynb)| 介绍了经典的 Inception 构架的模型，从零实现了 Inception 模块和完整的 GoogLeNet 网络。 |
| [MobileNet](./models/convnets/MobileNet.ipynb)| 介绍了面向移动设备的轻量级卷积网络架构 MobileNet，介绍了其核心的深度可分离卷积的实现思路，并从零实现了整个 MobileNetV1 的架构 |
| [ResNet](./models/convnets/ResNet.ipynb)| 介绍了 CNN 的最具有代表性的网络结构 ResNet，并从零开始逐步构建 ResNetBootleNeckLayer，ResNetStage，最后手动实现了一个完整的 ResNet50的结构。 |
| [DenseNet](./models/convnets/DenseNet.ipynb)| TODO |
| [EfficientNet](./models/convnets/efficientnet.ipynb)| TODO |
| [RegNet](./models/convnets/RegNet.ipynb)| TODO |
| [MLPMixer](./models/convnets/MLPMixer.ipynb)| 通过代码实现 MixerBlock，展示了如何通过只使用 MLP 来替换 SelfAttention 和 Conv 实现图像分类上的高效的模型结构。 |
| [ConvMixer](./models/convnets/ConvMixer.ipynb) | 实现了 ConvMixer 的模型结构，展示了在一个 patch 化的输入上进行 depth-wise 的卷积以及 1x1 卷积的一种模块设计。 |
| [ConvNext](./models/convnets/ConvNeXt.ipynb) | 实现了 ConvNext 中核心的 ConvNeXtBlock，展示了如何通过 7x7 的 Depthwise Conv 和 Pointwise Conv 来模拟 Transformer Block 结构。 |


### Vision Trasnformer Model

[近两年有哪些ViT(Vision Transformer)的改进算法？ - 盘子正的回答 - 知乎](https://www.zhihu.com/question/538049269/answer/2532582294) 

| Notebooks | 说明 |
| :---: | --- |
| [Vision Transformer](./models/vision-transformers/ViT.ipynb) | 重点实现了 ViT 架构中 PatchEmbedding 的部分，并介绍了当输入分辨率与预训练模型不一致时，如何对位置编码进行插值 |
| [Masked AutoEncoder](./models/vision-transformers/MAE.ipynb) | 从零实现了 MAE 整个模型中的各个关键部分，尤其是对于图像的预处理部分，如何进行 Random Mask |
| [Swin Transformer](./models/vision-transformers/Swin.ipynb) | 实现了 SwinTransformer 的模型结构，从零开始实现了 PatchEmbedding、窗口化的自注意力机制、Shiftd Windows 机制、PathMerging 等 |
| [CvT: Convolutional vision Transformer](./models/vision-transformers/CvT.ipynb) | TODO |
| [DiNAT: Dilated NAT](./models/vision-transformers/DiNAT.ipynb) | TODO |
| [DEtection TRansfomrer](./models/vision-transformers/DETR.ipynb) | TODO |
| [MobileViT](./models/vision-transformers/MobileViT.ipynb) | TODO |
| [DeiT: Data-Efficient ViT](./models/vision-transformers/DeiT.ipynb) | TODO |
| [BEiT](./models/vision-transformers/BEiT.ipynb) | TODO |
| [DINO](./models/vision-transformers/DINO.ipynb) | TODO |


### Text Transformer Model

| Title | Notebooks | 说明 |
| :---: | :---: | --- |
| 使用 Seq2Seq 模型来进行机器翻译 | [seq2seq.ipynb](./models/seq2seq.ipynb) | 演示了机器翻译数据集的预处理过程，通过 torch 中对于 Transformer 架构的支持，构建了一个 6 层的 Enocer-Decoder 架构的模型，实现了其正向的计算过程和整个模型的训练过程。|
| BERT | [bert.ipynb](./models/bert.ipynb) | 深入分析Bert模型的各个层和算子的实现细节 |
| GPT2 | [gpt2.ipynb](./models/gpt2.ipynb) | 解析了GPT2模型的结构与计算过程 |
| T5 | [t5.ipynb](./models/t5.ipynb) | 分析了Encoder-Deocder结构的T5模型的详细原理与计算流程 |
| Llama | [llama.ipynb](./models/llama.ipynb) | TODO |

### Multi-Modality Model

| Title | Notebooks | 说明 |
| :---: | :---: | --- |

## CV Tasks

| Notebooks | 说明 |
|:---:|---|
| [COCO 数据集介绍](./cv-tasks/object_detection/coco_dataset.ipynb)  | 介绍了 COCO 数据集的背景、标注格式等，并用代码演示了如何通过自定义数据集来加载 COCO 检测数据集 |
| [YOLO 目标检测](./cv-tasks/object_detection/yolo.ipynb)  | 介绍了经典的目标检测算法框架 YOLO 系列，从 YOLOV1 开始，介绍了 YOLO 系列发展。 |

## NLP Tasks 

| Title | Notebooks | 说明 |
|:---:|:---:|---|
| Tokenizer  | [Tokenizer](./nlp-tasks/tokenizer.ipynb)    | 介绍了Tokenizer的详细功能 |
| DistillBert | [DistillBert](./nlp-tasks/distilbert_cls.ipynb)| 分析DistillBert模型的结构与前向过程，并使用 DistillBert来微调一个文本分类模型 |
| 文本分类Finetune流程 | [Finetune](./nlp-tasks/text_cls_finetune.ipynb)| 基于小型BERT模型rbt3的文本分类的完整Finetune流程 |