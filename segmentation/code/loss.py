import torch
import torch.nn as nn


def dice_loss(y_true, y_pred):
    y_true_f = torch.flatten(y_true.permute(1, 0, 2), start_dim=1)
    y_pred_f = torch.flatten(y_pred.permute(1, 0, 2), start_dim=1)

    intersection = (y_pred_f * y_true_f).sum(dim=1)
    smooth = 0.0001
    return 1 - ((2 * intersection + smooth) / (y_true_f.sum(dim=1) + y_pred_f.sum(dim=1) + smooth)).mean()


def focal_loss(y_true, y_pred):
    loss = nn.CrossEntropyLoss(reduction='none')(y_true, y_pred)
    pt = torch.exp(-loss)
    return torch.mean(1 * (1 - pt) ** 2 * loss)
