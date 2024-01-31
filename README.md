# byol

This repo is a Pytorch implementation of BYOL for image classification tasks which uses Pytorch Lightning as its training wrapper. The current supported backbones are:

- ResNet18
- ...


# BYOL (Bootstrap Your Own Latent) - Self-Supervised Image Representation Learning

BYOL is a self-supervised approach to image representation learning. It relies on two neural networks, referred to as the online and target networks, that interact and learn from each other. The goal is to train the online network to predict the target network representation of an augmented view of an image. Simultaneously, the target network is updated with a slow-moving average of the online network.

## Resources

- [BYOL Paper](https://arxiv.org/abs/2006.07733): The original research paper describing the BYOL approach.
- [lightly](https://github.com/lightly-ai/lightly): A computer vision framework for self-supervised learning, which can be used to implement BYOL and other self-supervised learning methods.


## Examples

Some examples and explanations of the BYOL's functionality for tuning tasks can be found on the `examples` directory.

