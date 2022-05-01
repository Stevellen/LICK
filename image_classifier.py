from typing import List, Dict, Union, Any

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as opt
import torchmetrics as tm
import pytorch_lightning as pl


class LightningImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        num_labels,
        lr,
        optimizer,
        loss_fn,
        class_weights=None,
        **model_kwargs
    ):
        super().__init__()

        self.model = model
        self.num_labels = num_labels
        self.lr = lr
        self.optimizer = optimizer

    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    
        


def get_model(
    model: str,
    num_labels: int=-1,
    pretrained:bool=True,
    freeze_backbone: bool=True,
    class_weights: List[float]=None,
    optimizer: str='AdamW',
    lr: float=1e-3,
    loss_fn: Any = None
) -> LightningImageClassifier:
    model = getattr(models, model)(pretrained=pretrained)
    
    if num_labels > 0: # if we have a different number of labels for this task we create a new classifier head
        if hasattr(model, 'fc'): # model has a sinlge identifiable nn.Linear layer
            model.fc = nn.Linear(model.fc.in_features, num_labels)
        elif hasattr(model, 'classifier'): # model has an output contained in a nn.Sequential container
            output = list(model.classifier.children())[-1]
            model.classifier = nn.Sequential(
                *list(model.classifier.children())[:-1],
                nn.Linear(output.in_features, num_labels)
            )
            for layer in model.classifier.children():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        elif hasattr(model, 'heads'): # handles ViT model outputs
            output = model.heads.head
            model.heads.head = nn.Linear(output.in_features, num_labels)

    if freeze_backbone: # prevent gradient calculation for all but the final child
        for child in list(model.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False

    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss() if num_labels in [1,2] else nn.CrossEntropyLoss()
    return LightningImageClassifier(model, num_labels, lr, getattr(opt, optimizer), loss_fn, class_weights=class_weights)
