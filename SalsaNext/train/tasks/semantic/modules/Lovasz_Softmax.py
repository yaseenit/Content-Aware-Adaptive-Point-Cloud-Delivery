"""

MIT License

Copyright (c) 2018 Maxim Berman
Copyright (c) 2020 Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""
import torch
import torch.nn as nn
from torch.autograd import Variable


try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, label_type='class'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes, label_type=label_type)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes, label_type=label_type)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present', label_type='class'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.

    if label_type == 'criticality':
        labels = labels_to_crit(labels)
        pred   = torch.argmax(probas, dim=1)
        pred   = labels_to_crit(pred)
    
    average_probas = False # Collaps classes into criticalities or not
    regularization = None # One of ['reciprocal_criticality', 'linex', 'sigmoidal_linex']
    if label_type == 'criticality' and average_probas:
        probas = proba_to_crit(probas)

    C = probas.size(1) # num classes or num criticalities
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        if label_type == 'criticality' and not average_probas:
            c  = label_to_crit(c)

        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue

        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]

        errors = (Variable(fg) - class_pred).abs()
        if label_type == 'criticality':
            if regularization == 'reciprocal_criticality':
                errors /= (1+c) # Lower criticality levels should have more impact
            if regularization == 'linex':
                regularization_factor = linex_flat(labels, pred, c=0.3)
                regularization_factor += 1
                errors *= regularization_factor       
            if regularization == 'sigmoidal_linex':
                regularization_factor = linex_flat(labels, pred, c=1.0)
                regularization_factor = sigmoidal(regularization_factor)
                regularization_factor += 1
                errors *= regularization_factor

        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

# TODO: Better read those from the config file
c2l=[
    [], # "Moving" labels are ignored in Salsanext
    [1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [0]
]

def linex_flat(labels: torch.tensor, predictions: torch.tensor, c: float=0.5) -> torch.tensor:
    delta = ((predictions.float()+1.0) / (labels.float()+1.0)) - 1.0
    dc = delta*c
    loss = torch.exp(dc) - dc - 1.0
    return loss

def sigmoidal(x: torch.tensor) -> torch.tensor:
    y = x / (x+1)
    return y

def label_to_crit(label: int) -> int:
    """
    Convert a class label to its corresponding criticality label

    :param label: A class label
    
    :return: A criticality label
    """
    for level in range(5):
        if label in c2l[level]:
            return level

def labels_to_crit(labels: torch.tensor) -> torch.tensor:
    """
    Convert an array of class labels to an array of their corresponding criticality labels.

    :param labels: A torch tensor holding class labels
    
    :return: A torch tensor holding criticality labels
    """
    levels = torch.clone(labels)

    for level in range(5):
        for label in c2l[level]:
            levels[labels == label] = level
    return levels

def proba_to_crit(probas: torch.tensor) -> torch.tensor:
    """
    Collaps a tensor of class probabilities into a tensor of criticality probabilties
    """
    levels =  []
    for level in range(5):
        mask = c2l[level]
        if mask == []:
            levels.append(torch.zeros((probas.shape[0])))
        else:
            levels.append(probas[:, mask].mean(dim=1))
    
    return torch.vstack(levels).movedim(0, -1)

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None, label_type='class'):
        """
            target: One of ['class', 'criticality']
        """

        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        self.label_type = label_type

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore, label_type=self.label_type)
