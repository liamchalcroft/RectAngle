import torch
from torch import nn

# Loss function

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
  def __init__(self, soft=True, smooth=1e-7, threshold=0.5):
      super().__init__()
      self.smooth = smooth
      self.soft = soft
      self.threshold = threshold

  def forward(self, inputs, targets):
    # Assume already in int form - binarise function available
    # Seems to perform very well without binary - soft dice?

    if not self.soft:
      inputs = BinaryDice(inputs, self.threshold)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = torch.sum(inputs * targets)
    dice = ((2. * intersection) + self.smooth) / \
            (torch.sum(inputs) + torch.sum(targets) + self.smooth)

    return (1 - dice)

  @staticmethod
  def BinaryDice(image, threshold=0.5):
    return (image > threshold).int()
