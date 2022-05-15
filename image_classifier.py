from typing import List, Dict, Union, Any

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as opt
import torchmetrics as tm
import pytorch_lightning as pl
import torchmetrics as tm

from metrics import get_cmat, get_metrics


class LightningImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        num_labels,
        lr,
        optimizer,
        loss_fn,
        threshold
    ):
        super().__init__()

        self.model = model
        self.num_labels = num_labels
        self.lr = lr
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.activation_fn = nn.Sigmoid() if num_labels == 1 else nn.Softmax(dim=1)
        self.train_metrics, self.validation_metrics, self.test_metrics = get_metrics(threshold)
        self.validation_cm = tm.ConfusionMatrix(
            num_classes = 2 if num_labels in [1,2] else num_labels,
            normalize='true'
        )
        self.test_cm = tm.ConfusionMatrix(
            num_classes = 2 if num_labels in [1,2] else num_labels,
            normalize='true'
        )

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    ###############################################################################################
    ##                            Step/Epoch Function Definitions                                ##
    ###############################################################################################


    def _step(self, batch):
        inputs, labels = batch
        preds = self(inputs).squeeze(1)
        loss = self.loss_fn(preds, labels)
        return self.activation_fn(preds), labels, loss
    
    def training_step(self, batch, batch_idx):
        preds, labels, loss = self._step(batch)
        self.log_training_metrics(preds, labels.int(), loss, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, labels, loss = self._step(batch)
        self.log_validation_metrics(preds, labels.int(), loss, batch_idx)
        return self.activation_fn(preds), labels

    def validation_epoch_end(self, outputs):
        preds, labels = zip(*outputs)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        self.logger.experiment.add_figure(
            'test_confusion_matrix',
            get_cmat(self.validation_cm(preds, labels.int()).cpu().numpy(), stage='validation'),
            global_step=self.current_epoch,
            close=True
        )

    def test_step(self, batch, batch_idx):
        preds, labels, loss = self._step(batch)
        self.log_test_metrics(preds, labels.int(), loss, batch_idx)
        return self.activation_fn(preds), labels

    def test_epoch_end(self, outputs):
        preds, labels = zip(*outputs)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        self.logger.experiment.add_figure(
            'test_confusion_matrix',
            get_cmat(self.test_cm(preds, labels.int()).cpu().numpy(), stage='test'),
            global_step=self.current_epoch,
            close=True
        )

    ###############################################################################################
    ##                                  Metric Logging Definitions                               ##
    ###############################################################################################
    @torch.no_grad()
    def log_training_metrics(self, preds, labels, loss, batch_idx):
        outputs = self.train_metrics(preds, labels)
        self.log_dict(outputs)
        self.log('training_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.logger.experiment.add_histogram(
            'training_prediction_distribution',
            preds,
            global_step=self.global_step
        )
        self.logger.experiment.add_histogram(
            'training_loss_distribution',
            loss,
            global_step=self.global_step
        )

    @torch.no_grad()
    def log_validation_metrics(self, preds, labels, loss, batch_idx):
        outputs = self.validation_metrics(preds, labels)
        self.log_dict(outputs)
        self.log('validation_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.logger.experiment.add_histogram(
            'validation_prediction_distribution',
            preds,
            global_step=self.global_step
        )
        self.logger.experiment.add_histogram(
            'validation_loss_distribution',
            loss,
            global_step=self.global_step
        )

    @torch.no_grad()
    def log_test_metrics(self, preds, labels, loss, batch_idx):
        outputs = self.test_metrics(preds, labels)
        self.log_dict(outputs)
        self.log('test_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.logger.experiment.add_histogram(
            'test_prediction_distribution',
            preds,
            global_step=self.global_step
        )
        self.logger.experiment.add_histogram(
            'test_loss_distribution',
            loss,
            global_step=self.global_step
        )



def get_model(
    model: str,
    num_labels: int = -1,
    pretrained:bool = True,
    freeze_backbone: bool = True,
    class_weights: List[float] = None,
    optimizer: str = 'AdamW',
    lr: float = 1e-3,
    threshold: float = 0.5
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

    loss_fn = nn.BCEWithLogitsLoss(class_weights) if num_labels in [1,2] else nn.CrossEntropyLoss(class_weights)

    return LightningImageClassifier(
        model,
        num_labels,
        lr,
        getattr(opt, optimizer),
        loss_fn,
        threshold
    )
