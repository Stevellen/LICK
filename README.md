# Simple Lightning Image Classification toolKit (SLICK)

### This repository is a work in progress and is not functional yet.

SLICK is a simple toolkit to enable training of image classification models on the Pytorch-Lightning framework.

This toolkit is designed to be used with any of the classification models available in the [Torchvision](https://pytorch.org/vision/stable/models.html#classification) library, but the `LightningImageClassifier` object should be able to facilitate the training of any image classification model. 

A variety of metrics are recorded by SLICK during the training, validation and test phases of an image classifiaction experiment. These artifacts are stored as [Tensorboard](https://www.tensorflow.org/tensorboard) logs for later analysis.