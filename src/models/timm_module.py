from typing import Any, List, Optional

import hydra
import torch
from coral_pytorch.layers import CoralLayer
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression import MeanAbsoluteError


class OrdinalRegressor(LightningModule):
    def __init__(
        self,
        num_classes: int,
        backbone: Any = None,
        head: str = "linear",
        loss: Any = None,
        optim: Any = None,
        sche: Any = None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = hydra.utils.instantiate(self.hparams.backbone, features_only=True)
        ch = self.net.feature_info.channels()[-1]
        if head == "linear":
            classifer = nn.Linear(ch, num_classes)
        elif head == "coral":
            classifer = CoralLayer(size_in=ch, num_classes=num_classes)
        elif head == "corn":
            classifer = nn.Linear(ch, num_classes - 1)
        else:
            raise ValueError(f"Unknown head type: {head}")

        self.output_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), classifer)

        # loss function
        self.criterion = hydra.utils.instantiate(loss)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mae = MeanAbsoluteError()
        self.train_acc = Accuracy(num_classes=num_classes)
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for logging best accuracy achieved so far
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.output_layer(self.net(x)[-1])

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        criterion_dict = self.criterion(logits, y)
        loss = criterion_dict.pop("loss")
        preds = criterion_dict.pop("preds")
        return loss, preds, y, criterion_dict  # remaining are metrics

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, metrics = self.step(batch)

        # log train metrics
        self.train_mae(preds.float(), targets.float())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)
        for remaining in metrics:
            self.log(
                f"train/{remaining}",
                metrics[remaining],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, metrics = self.step(batch)

        # log val metrics
        self.val_mae(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        for remaining in metrics:
            self.log(
                f"val/{remaining}", metrics[remaining], on_step=False, on_epoch=True, prog_bar=True
            )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        mae = self.val_mae.compute()  # get val accuracy from current epoch
        # compute and log best so far val mae
        self.val_mae_best.update(mae)
        self.log("val/mae_best", self.val_mae_best.compute(), on_epoch=True, prog_bar=True)

    # TODO: add test step
    def test_step(self, batch: Any, batch_idx: int):
        pass
        # loss, preds, targets, metrics = self.step(batch)

        # # log test metrics
        # self.test_mae(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/mae", self.test_mae, on_step=False, on_epoch=True)
        # for remaining in metrics:
        #     self.log(f"test/{remaining}", metrics[remaining], on_step=False, on_epoch=True)

        # return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        opt = hydra.utils.instantiate(self.hparams.optim, params=self.parameters())
        sch = hydra.utils.instantiate(self.hparams.sche, optimizer=opt)
        return [opt], [sch]
