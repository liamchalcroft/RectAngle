import os
import argparse

from rectangle.utils.transforms import Affine

parser = argparse.ArgumentParser(prog='train',
                                description="Train RectAngle model. See list of available arguments for more info.")

parser.add_argument('--train',
                    '--tr',
                    metavar='train',
                    type=str,
                    action='store',
                    default='./miccai_us_data/train.h5',
                    help='Path to training data. Note that for ensemble this should include train + val pre-split.')

parser.add_argument('--val',
                    '--v',
                    metavar='val',
                    type=str,
                    action='store',
                    default=None,
                    help='Path to validation data.')

parser.add_argument('--label',
                    '--l',
                    metavar='label',
                    type=str,
                    action='store',
                    default='random',
                    help="Label sampling strategy. Should be string of {'random', 'vote', 'mean'}.")

parser.add_argument('--ensemble',
                    '--en',
                    metavar='ensemble',
                    type=str,
                    action='store',
                    default=None,
                    help='Number of ensembled models.')

parser.add_argument('--gate',
                    '--g',
                    metavar='gate',
                    type=str,
                    action='store',
                    default=None,
                    help='(Optional) Attention gating.')

parser.add_argument('--lr_schedule',
                    '--lrs',
                    metavar='lr_schedule',
                    type=str,
                    action='store',
                    default=None,
                    help="Method for scheduling of learning rate. {None, 'lambda', 'exponential', 'reduce_on_plateau'}")

parser.add_argument('--odir',
                    '--o',
                    metavar='odir',
                    type=str,
                    action='store',
                    default='./',
                    help='Path to output folder.')

parser.add_argument('--depth',
                    '--d',
                    metavar='depth',
                    type=str,
                    action='store',
                    default='5',
                    help='Depth of U-Net architecture used.')

parser.add_argument('--epochs',
                    '--ep',
                    metavar='epochs',
                    type=str,
                    action='store',
                    default='200',
                    help='Max number of training epochs per model.')

parser.add_argument('--batch',
                    '--b',
                    metavar='batch',
                    type=str,
                    action='store',
                    default='32',
                    help='Batch size. Note images are large (~400x~300).')

parser.add_argument('--seed',
                    '--s',
                    metavar='seed',
                    type=str,
                    action='store',
                    default=None,
                    help='Random seed for training.')


args = parser.parse_args()

## convert arguments to useable form
if args.ensemble:
    ensemble = int(args.ensemble)
else:
    ensemble = None

## run training
import rectangle as rect
import h5py
import torch
import random
import numpy as np

print("Code running")
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# set seeds for repeatable results
if args.seed:
    seed = int(args.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

f_train = h5py.File(args.train, 'r')
train_data = rect.utils.io.H5DataLoader(f_train, label=args.label)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print("Cuda available!")
else:
    device = torch.device('cpu')
    print("Using CPU!")

if args.val:
    f_val = h5py.File(args.val, 'r')
    val_data = rect.utils.io.H5DataLoader(f_val, label='vote')

model = rect.model.networks.UNet(n_layers=int(args.depth), device=device,
                                    gate=args.gate)

trainer = rect.utils.train.Trainer(model, ensemble=ensemble, outdir=args.odir, device=device,
                                    nb_epochs=int(args.epochs), lr_schedule=args.lr_schedule)

#Manually setting Affine Transforms
AffineTransform = rect.utils.transforms.Affine(prob = 0.3, scale = (1,1), degrees = 5, shear = 0, translate = 0)

if args.val:
    trainer.train(train_data, val_data, train_pre=[rect.utils.transforms.z_score(), rect.utils.transforms.Flip(), AffineTransform],
                    val_pre=[rect.utils.transforms.z_score()], train_batch=int(args.batch))
else:
    trainer.train(train_data, train_pre=[rect.utils.transforms.z_score(), rect.utils.transforms.Flip(), AffineTransform], 
                    val_pre=[rect.utils.transforms.z_score()], train_batch=int(args.batch))

