# Sebastian Raschka 2020-2021
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import math

import torch
import torch.nn.functional as F
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss, corn_loss


def normal_sampling(mean, label_k, std=2):
    return math.exp(-((label_k - mean) ** 2) / (2 * std**2)) / (math.sqrt(2 * math.pi) * std)


def get_dist_batch(targets, num_classes, std=2):
    dist_batch = []
    for age in targets:
        dist = [normal_sampling(int(age.item()), i, std) for i in range(num_classes)]
        dist = [i if i > 1e-15 else 1e-15 for i in dist]
        dist_batch.append(dist)
    return torch.tensor(dist_batch).float().to(targets.device)


class DexLoss(torch.nn.Module):
    def __init__(self, num_classes: int, softmax: bool = True):
        super(DexLoss, self).__init__()
        self.num_classes = num_classes
        self.softmax = softmax

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kargs):
        loss = F.cross_entropy(logits, target.long())
        a = torch.arange(0, self.num_classes, dtype=torch.float32).to(logits.device)
        if self.softmax:
            logits = logits.softmax(1)
        else:
            logits = logits.sigmoid()
        pred = torch.squeeze((logits * a).sum(1, keepdim=True), dim=1)
        pred = torch.round(pred)
        mae = (pred - target).abs().mean()
        return {"loss": loss + mae, "preds": pred}


class CoralLoss(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        levels = levels_from_labelbatch(target, num_classes=self.num_classes).type_as(logits)

        loss = coral_loss(logits, levels.type_as(logits))
        probas = torch.sigmoid(logits)
        pred = proba_to_label(probas)
        return {"loss": loss, "preds": pred}


class CornLoss(CoralLoss):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):

        loss = corn_loss(logits, target, num_classes=self.num_classes)
        pred = corn_label_from_logits(logits)
        return {"loss": loss, "preds": pred}
