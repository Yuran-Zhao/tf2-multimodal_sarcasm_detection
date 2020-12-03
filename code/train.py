from features import *
from model import *
from utils import *


def train(model, train_loader, val_loader, loss_fn, optimizer, number_of_epoch):
    """Complete the training

    Args:
        model (tf.): [description]
        train_loader ([type]): [description]
        val_loader ([type]): [description]
        loss_fn ([type]): [description]
        optimizer ([type]): [description]
        number_of_epoch ([type]): [description]
    """
