from cProfile import label
from numpy import absolute
import torch
import torch.nn as nn

class ClassDifferenceLoss(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None, absolute=True, normalize=True, label_type='class', activation=None):
        super(ClassDifferenceLoss, self).__init__()
        # TODO: Add support for classes and ignore argument

        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        self.absolute = absolute
        self.normalize = normalize
        self.label_type = label_type
        self.activation = activation

    def forward(self, probas, labels):
        error = self.mean_error(probas, labels, label_type=self.label_type, per_image=self.per_image, ignore=self.ignore, normalize=self.normalize, absolute=self.absolute, activation=self.activation)
        return error

    def label2crit(self, labels: torch.tensor) -> torch.tensor:
        """
        Convert a tensor of class labels to a tensor of criticality labels
        labels: [B, H, W] tensor
        """
        levels = torch.clone(labels)
        # TODO: Better read those from the config file
        c2l=[
            [], # "Moving" labels are ignored in Salsanext
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [0]
        ]
        for level in range(5):
            for label in c2l[level]:
                levels[labels == label] = level
        return levels

    def mean_error(self, probas, labels, label_type='class', classes='present', per_image=False, ignore=None, normalize=False, absolute=True, activation=None):
        """
        Mean Error
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        label_type: One of ['class', 'criticality']
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        pred = torch.argmax(probas, dim=1) # Collaps [B,C,H,W] to [B,H,W]
 
        if label_type == 'class':
            pred   = self.label2crit(pred)
            labels = self.label2crit(labels)

        diffs = pred - labels
        diffs = diffs.float()

        if activation == 'relu':
            diffs       = nn.functional.relu(diffs)        

        if absolute:
            diffs = torch.abs(diffs)

        # TODO: Infer num_classes or make it configurable
        if normalize:
            diffs /= 2 # num_classes -1

        dim=0
        if per_image:
            dim=1

        diffs = torch.flatten(diffs, start_dim=dim)
        loss  = torch.mean(diffs, dim=dim)
        return loss