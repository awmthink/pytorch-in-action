# model
model = "resnet18"
weights = "ResNet18_Weights.IMAGENET1K_V1"
pretrain = False

# optimizer
learning_rate = 0.01

# data loader
dataset_root_dir = "/data/datasets/ImageNet-1K/raw/ImageNet-1K"
num_workers = 4
batch_size = 32

# trainer
num_epochs = 1
accelerator = "gpu"
strategy = "ddp"
devices = 4
precision = "16-mixed"
log_dir = "./"
profiler = "simple"
checkpoint_path = "./checkpoints/"
