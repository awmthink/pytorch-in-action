====================
Pytorch In Action
====================

A hands-on repository dedicated to building modern deep learning layers, models and tasks from scratch using PyTorch.

.. toctree::
   :caption: Pytorch Basics
   :maxdepth: 1

   Tensors <pytorch-basics/Tensors>
   Automatic Differentiation <pytorch-basics/AutoGrad>
   Modules <pytorch-basics/Modules>
   Dataset and Dataloader <pytorch-basics/DataLoading>
   Save and Load <pytorch-basics/SaveAndLoad>

.. toctree::
   :caption: Pytorch Advanced
   :maxdepth: 1

   Distributed Training <pytorch-advanced/Distributed>

.. toctree::
   :caption: ConvNets
   :maxdepth: 1

   VGG <models/convnets/VGG>
   GoogLeNet <models/convnets/GoogLeNet>
   ResNet <models/convnets/ResNet>
   MobileNet <models/convnets/MobileNet>
   ConvNeXt <models/convnets/ConvNeXt>
   MLPMixer <models/convnets/MLPMixer>
   ConvMixer <models/convnets/ConvMixer>


.. toctree::
   :caption: Vision Trasnformer
   :maxdepth: 1

   ViT：VisionTrasnformer <models/vision-transformers/ViT.ipynb>
   MAE: Masked AutoEncoder <models/vision-transformers/MAE.ipynb>
   SwinTrasnformer <models/vision-transformers/Swin.ipynb>
   CvT: Convolutional vision Transformer <models/vision-transformers/CvT.ipynb>
   DiNAT: Dilated Neigborhood Attention Transformer <models/vision-transformers/DiNAT.ipynb>
   MobileViT <models/vision-transformers/MobileViT.ipynb>
   DETR: DEtection TRansformer <models/vision-transformers/DETR.ipynb>
   MaskFormer: Mask Classification-based Segmentation Transformer <models/vision-transformers/MaskFormer.ipynb>
   OneFormer: Universal Architecture For Segmentation Tasks <models/vision-transformers/OneFormer.ipynb>


.. toctree::
   :caption: Compute Vision Tasks
   :maxdepth: 1

   COCO Dataset <cv-tasks/object_detection/coco_dataset>
   YOLO  <cv-tasks/object_detection/yolo>
   Transfer Learning ViT Image Classifiction <cv-tasks/classification/transfer-learning-image-classification>
   LoRAFinetning Vision Transformer <cv-tasks/classification/LoRA-Image-Classification>


.. toctree::
   :caption: OpenAI Triton Kernels
   :maxdepth: 1

   Vector Add <triton/vector-add>
   Fused Softmax <triton/fused-softmax>
   Matrix Multiplication <triton/matrix-multiplication>
   Low Memory Dropout <triton/low-memory-dropout>
   LayerNorm <triton/layer-norm>
