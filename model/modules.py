import torch
from torch import nn


class UNetBlock(nn.Module):
  """ Basic convolution block for use in encoder and decoder of U-Net.
  Input arguments:
    ch_in : int
            Number of input channels to block.
    n_feat : int
             Number of feature channels in block.
    device : string, default = 'cpu'
             Device to store weights on.
  """
  def __init__(self, ch_in, n_feat, device='cpu'):

    super().__init__()

    self.conv1 = torch.nn.Sequential(
        nn.Conv2d(in_channels=ch_in, out_channels=n_feat, 
                  kernel_size=3, padding=1).to(device),
        nn.ReLU(inplace=True).to(device),
        nn.BatchNorm2d(num_features=n_feat).to(device))
    
    self.conv2 = torch.nn.Sequential(
        nn.Conv2d(in_channels=n_feat, out_channels=n_feat, 
                  kernel_size=3, padding=1).to(device),
        nn.ReLU(inplace=True).to(device),
        nn.BatchNorm2d(num_features=n_feat).to(device)
        )
    
    self.conv1.to(device)
    self.conv2.to(device)
    
  def forward(self, x):
    x = self.conv1(x)
    x = x + self.conv2(x)
    return x


class AttentionGate(nn.Module):
  """ Attention gate module based on work described in 
  https://arxiv.org/pdf/1804.03999.pdf
  Input arguments:
    encoder_channels : int
                       Number of input channels from encoder
    decoder_channels : int
                       Number of input channels from decoder (typically will
                       equal 2 * encoder_channels)
    intermediate_channels : int
                            Number of channels to use in intermediate
                            activations of attention gate. Typically use same
                            value as encoder_channels.
  """
  def __init__(self, encoder_channels, 
               decoder_channels, intermediate_channels):
    
    super().__init__()

    self.gate_weight = nn.Sequential(
        nn.Conv2d(decoder_channels, intermediate_channels, kernel_size=1),
        nn.BatchNorm2d(intermediate_channels)
    )

    self.activation_weight = nn.Sequential(
        nn.Conv2d(encoder_channels_channels, intermediate_channels, kernel_size=1),
        nn.BatchNorm2d(intermediate_channels)
    )

    self.relu = nn.ReLU(inplace=True)

    self.psi = nn.Sequential(
        nn.Conv2d(intermediate_channels, 1, kernel_size=1),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
    )

  def forward(self, x_enc, x_dec):
    g = self.gate_weight(x_dec)
    x = self.activation_weight(x_enc)
    x = self.relu(x + g)
    x = self.psi(x)
    return x * x_enc


# class LambdaGate(nn.Module):
#   """ (WORK IN PROGRESS) Gating module based on Lambda layers described in
#   https://arxiv.org/pdf/2102.08602.pdf
#   Input arguments:

#   """
#   def __init__(self, ):
