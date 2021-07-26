# -*- coding: utf-8 -*-
"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

To run the training in a distributed fashion in AWS.
First follow https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/80fe1ab73c6b2b3cefcd5ba0e4ed7609/aws_distributed_training_tutorial.ipynb#scrollTo=6sDCKgjyXBRT
to set up two p2 instances that can mutually talk.
Find out private ip address of each.
Copy data file and transfer_learning_tutorial.py to each node.
scp -i ~/Documents/aws_secrets/xwjiang-test.pem pytorch/transfer_learning_tutorial.py ubuntu@ec2-34-216-171-232.us-west-2.compute.amazonaws.com:/home/ubuntu/pytorch/
Run this with `python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="172.31.59.30" --master_port=1234 transfer_learning_tutorial.py`.
"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import uuid
import sys

# plt.ion()  # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

def main():
    # parse the local_rank argument from command line for the current process
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    # setup the distributed backend for managing the distributed training
    torch.distributed.init_process_group('nccl')
    global_rank = torch.distributed.get_rank()

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # Setup the distributed sampler to split the dataset to each GPU.
    distributed_sampler_for_training = DistributedSampler(image_datasets['train'])
    dataloader_for_training = DataLoader(image_datasets['train'], batch_size=16, num_workers=4, sampler=distributed_sampler_for_training)
    dataloader_for_test = DataLoader(image_datasets['val'], batch_size=4,
                                 shuffle=True,
                                 num_workers=4,
                                 )
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device('cuda', args.local_rank)

    ######################################################################
    # Visualize a few images
    # ^^^^^^^^^^^^^^^^^^^^^^
    # Let's visualize a few training images so as to understand the data
    # augmentations.

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))
    #
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    #
    # # imshow(out, title=[class_names[x] for x in classes])


    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.
    #
    # returns model - only meaningful when it's running in the process with global_rank == 0


    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('LocalRank {} | Epoch {}/{}'.format(args.local_rank, epoch, num_epochs - 1))
            print('-' * 10)

            # training phase
            model.train()
            per_epoch_since = time.time()
            for inputs, labels in dataloader_for_training:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize
                    loss.backward()
                    optimizer.step()

            scheduler.step()
            if global_rank == 0:
                per_epoch_time_elapsed = time.time() - per_epoch_since
                print('Per epoch training time: {:.6f}s'.format(
                    per_epoch_time_elapsed))
            else:
                continue

            print('evaluating...')
            # test phase
            model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader_for_test:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    outputs = model.module(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes['val']
            epoch_acc = running_corrects.double() / dataset_sizes['val']

            print('eval Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if global_rank == 0:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    ######################################################################
    # Visualizing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # Generic function to display predictions for a few images
    #

    def visualize_model(model, title, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
        fig.suptitle(title)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader_for_test):
                inputs = inputs.to(device)
                # labels = labels.to(device)

                outputs = model.module(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        plt.savefig('results/' + str(uuid.uuid4()) + '.jpg')
                        return
            model.train(mode=was_training)
            plt.savefig('results/' + str(uuid.uuid4()) + '.jpg')


    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #

    model_ft = models.resnet18(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    model_ft = torch.nn.parallel.DistributedDataParallel(model_ft, device_ids=[args.local_rank],
                                                         output_device=args.local_rank)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.module.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    ######################################################################
    #

    if global_rank == 0:
        visualize_model(model_ft, 'Funetuning the convnet')


    ######################################################################
    # ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad == False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)
    model_conv = torch.nn.parallel.DistributedDataParallel(model_conv, device_ids=[args.local_rank],
                                                           output_device=args.local_rank)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.module.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # On CPU this will take about half the time compared to previous scenario.
    # This is expected as gradients don't need to be computed for most of the
    # network. However, forward does need to be computed.
    #

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)

    ######################################################################
    #
    if global_rank == 0:
        visualize_model(model_conv, 'ConvNet as fixed feature extractor')

    # plt.ioff()
    # plt.show()

    ######################################################################
    # Further Learning
    # -----------------
    #
    # If you would like to learn more about the applications of transfer learning,
    # checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
    #
    #


if __name__ == "__main__":
    main()
