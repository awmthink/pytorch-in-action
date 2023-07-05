from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torchvision.models as tv_models
import pytorch_lightning as pl
import torchmetrics


class ImageNetClassifier(pl.LightningModule):
    def __init__(self, model_name, weights, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        # model
        self.model = tv_models.get_model(model_name, weights=weights)
        num_classes = self.model.fc.out_features
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.acc_top1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.acc_top5 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, acc_top1, acc_top5 = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc@1": acc_top1,
                "train_acc@5": acc_top5,
            },
            on_step=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc_top1, acc_top5 = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc@1": acc_top1,
                "val_acc@5": acc_top5,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc_top1, acc_top5 = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc@1": acc_top1,
                "test_acc@5": acc_top5,
            }
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        acc_top1 = self.acc_top1(logits, y)
        acc_top5 = self.acc_top5(logits, y)

        return loss, acc_top1, acc_top5

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape((x.shape[0], -1))
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
