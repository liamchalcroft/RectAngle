import torch
from torch import nn

# Loss function

class WeightedBCE(nn.Module):
  def __init__(self, weights=None):
      super().__init__()
      self.weights = weights
      
  def forward(self, inputs, targets):
    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    if self.weights is not None:
        assert len(self.weights) == 2

        loss = weights[1] * (targets * torch.log(inputs)) + \
        weights[0] * ((1 - targets) * torch.log(1 - inputs))
    else:
        loss = targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)

    return loss


class DiceLoss(nn.Module):
  """ Loss function based on Dice-Sorensen Coefficient (L = 1 - Dice)
  Input arguments:
    soft : boolean, default = True
           Select whether to use soft labelling or not. If true, dice calculated
           directly on sigmoid output without converting to binary. If false,
           sigmoid output converted to binary based on threshold value
    smooth : float, default = 1e-7
             Smoothing value to add to numerator and denominator of Dice. Low
             value will prevent inf or nan occurrence.
    threshold : float, default = 0.5
                Threshold of sigmoid activation values to convert to binary.
                Only applied if soft=False.
  """
  # Standard Dice loss, with variable smoothing constant
  def __init__(self, soft=True, threshold=0.5, eps=1e-7):
      super().__init__()
      self.eps = eps
      self.soft = soft
      self.threshold = threshold

  def forward(self, inputs, targets):
    # Assume already in int form - binarise function available
    # Seems to perform very well without binary - soft dice?

    if not self.soft:
      inputs = self.BinaryDice(inputs, self.threshold)

    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    intersection = torch.sum(inputs * targets)
    dice = ((2. * intersection) + self.eps) / \
            (torch.sum(inputs) + torch.sum(targets) + self.eps)

    return (1 - dice)

  @staticmethod
  def BinaryDice(image, threshold=0.5):
    return (image > threshold).int()


class Precision(object):
  """ Precision metric (TP/(TP+FP))
  """
  def __init__(self, eps=1e-7):
    super().__init__()
    self.eps = eps

  def __call__(self, inputs, targets):
    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    TP = torch.sum(inputs * targets)
    FP = torch.sum((inputs == 1) & (targets == 0))

    return (TP + self.eps)/(TP + FP + self.eps)


class Recall(object):
  """ Recall metric (TP/(TP+FN))
  """
  def __init__(self, eps=1e-7):
    super().__init__()
    self.eps = eps

  def __call__(self, inputs, targets):
    inputs = inputs.view(-1).float()
    targets = targets.view(-1).float()

    TP = torch.sum(inputs * targets)
    FP = torch.sum((inputs == 0) & (targets == 1))

    return (TP + self.eps)/(TP + FP + self.eps)
    

class Accuracy(object):
  """ Simple binary classifier accuracy
  """
  def __init__(self):
    super().__init__()

  def __call__(self, inputs, targets):
    # inputs = inputs.view(-1).float()
    # targets = targets.view(-1).float()

    correct = (torch.round(inputs) == targets).sum().item()

    return correct / inputs.size(0)
