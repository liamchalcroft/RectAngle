import torch
from torch import nn
from .modules import UNetBlock, AttentionGate
from torchvision.models import densenet161
from torchvision.transforms import Normalize


class UNet(nn.Module):
  """ U-Net model for 2D segmentation.
  Input arguments:
    device : string, default = 'cpu'
             Device to which all modules are loaded.
    ch_in : int, default = 1
            Number of channels in network input. N.B. many of the available
            augmentations only currently support a single-channel input.
    ch_out : int, default = 1
             Number of channels(classes) in output segmentation. N.B. available
             loss function and post-processing only supports single-channel.
    first_layer : int, default = 16
                  Number of feature channels in first UNetBlock module.
    n_layers : int, default = 4
               Number of layers in network. Subsequent layers have double the
               channels of the previous, e.g. for (first_layer=16, n_layers=4)
               the network will have layers (16, 32, 64, 128).
    gate : string, default = None
           Choice of gating module for skip connections.
           Options:
                  * 'attention' : AttentionGate module
                  * 'lambda' : LambdaGate module (not yet available)
                  * None : No gate used
  """
  def __init__(self, device='cpu', ch_in=1, ch_out=1,
               first_layer=16, n_layers=4, 
               gate=None):
        
    super().__init__()

    self.device = device

    self.attn_list = None

    if gate == 'attention':
      attn_list = []
      self.attn_list = attn_list
    
    enc_list = []
    down_list = []

    enc_list.append(UNetBlock(ch_in, first_layer, device).to(device))
    if self.attn_list:
      attn_list.append(AttentionGate(first_layer, first_layer*2, 
                                    first_layer*2).to(device))
    down_list.append(nn.MaxPool2d(kernel_size=2, stride=2).to(device))

    for i in range(n_layers-1):
      enc_list.append(UNetBlock(first_layer*(2**i), 
                                          first_layer*(2**(i+1)),
                                     device).to(device))
      if self.attn_list:
        attn_list.append(AttentionGate(first_layer*(2**i), 
                                       first_layer*(2**(i+1)), 
                                       first_layer*(2**(i+1))).to(device))
      down_list.append(nn.MaxPool2d(kernel_size=2, stride=2).to(device))

    self.enc_list = nn.ModuleList(enc_list)
    if self.attn_list:
      self.attn_list = nn.ModuleList(attn_list)
    self.down_list = nn.ModuleList(down_list)
      
    self.bottleneck = UNetBlock(first_layer*(2**(n_layers-1)),
                                     first_layer*(2**(n_layers)),
                                     device).to(device)

    dec_list = []
    up_list = []

    for i in range(n_layers):
      up_list.append(nn.ConvTranspose2d(first_layer*(2**(n_layers-(i))), 
                                        first_layer*(2**(n_layers-(i+1))), 
                                        kernel_size=2, stride=2).to(device))
      dec_list.append(UNetBlock(first_layer*(2**(n_layers-(i))), 
                                     first_layer*(2**(n_layers-(i+1))),
                                     device).to(device))
      
    self.dec_list = nn.ModuleList(dec_list)
    self.up_list = nn.ModuleList(up_list)
      
    self.head = nn.Conv2d(in_channels=first_layer, out_channels=ch_out, 
                          kernel_size=1).to(device)
    
    self.enc_list.to(device)
    if self.attn_list:
      self.attn_list.to(device)
    self.down_list.to(device)
    self.bottleneck.to(device)
    self.dec_list.to(device)
    self.up_list.to(device)
    self.head.to(device)
    
    
  def forward(self, x):
    # # pad images to (64,64)
    # shape = list(x.shape)
    # shape[2] = shape[3] = 64
    # pad_x = torch.zeros(shape)
    # pad_x[:,:,:58,:52] = x
    # x = pad_x.to(self.device)

    enc_features = []

    for i in range(len(self.enc_list)):
      x = self.enc_list[i](x)
      enc_features.append(x)
      x = self.down_list[i](x)
    
    x = self.bottleneck(x)

    enc_features = enc_features[::-1]

    for i in range(len(self.dec_list)):
      if self.attn_list:
        atnn_enc = self.attn_list[i](enc_features[i], x)
      x = self.up_list[i](x)
      if self.attn_list:
        x = torch.cat((x, attn_enc), dim=1)
      else:
        x = torch.cat((x, enc_features[i]), dim=1)
      x = self.dec_list[i](x)

    x = self.head(x)

    # # crop to original size of (58,52)
    # x = x[:,:,:58,:52]
    
    return torch.sigmoid(x)


class DenseNet(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.normalise = Normalize(0.449, 0.226)
    self.resample = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.features = nn.Sequential(*list(model.features)[1:])
    self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features=2208, out_features=1, bias=True), nn.Sigmoid())

  def forward(self, x):
    x = self.normalise(x)
    x = self.resample(x)
    x = self.features(x)
    # x = torch.squeeze(x)
    x = self.classifier(x)
    return x


def MakeDenseNet(freeze_weights=True, pretrain=True):
  cnn = densenet161(pretrained=pretrain)

  if freeze_weights:
    for param in cnn.parameters():
      param.requires_grad=False

  return DenseNet(cnn)
