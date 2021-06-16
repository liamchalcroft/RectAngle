## CLI for running training

import os
import sys
import argparse

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

parser.add_argument('--test',
                    '--te',
                    metavar='test',
                    type=str,
                    action='store',
                    default=None,
                    help='Path to test data.')

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

parser.add_argument('--classifier',
                    '--c',
                    metavar='classifier',
                    type=bool,
                    action='store',
                    default=True,
                    help='Use of classifier for pre-screening. If selected will train without and then perform test without + with.')

args = parser.parse_args()

## convert arguments to useable form
if args.ensemble:
    ensemble = int(args.ensemble)
else:
    ensemble = None

## run training

import rectangle as rect
import h5py
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress
import numpy as np
from datetime import date
import os

f_train = h5py.File(args.train, 'r')
train_data = rect.utils.io.H5DataLoader(f_train, label=args.label)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

if args.val:
    f_val = h5py.File(args.val, 'r')
    val_data = rect.utils.io.H5DataLoader(f_val, label='vote')

if args.test:
    f_test = h5py.File(args.test, 'r')
    test_data = rect.utils.io.H5DataLoader(f_test, label='vote')

model = rect.model.networks.UNet(n_layers=int(args.depth), device=device,
                                    gate=args.gate)

trainer = rect.utils.train.Trainer(model, ensemble=ensemble, outdir=args.odir,
                                    nb_epochs=int(args.epochs))

if args.val:
    trainer.train(train_data, val_data, train_pre=[rect.utils.transforms.z_score(), rect.utils.transforms.Flip(), rect.utils.transforms.Affine(), rect.utils.transforms.SpeckleNoise()], 
                    val_pre=[rect.utils.transforms.z_score()], train_batch=int(args.batch))
else:
    trainer.train(train_data, train_pre=[rect.utils.transforms.z_score(), rect.utils.transforms.Flip(), rect.utils.transforms.Affine(), rect.utils.transforms.SpeckleNoise()], 
                    val_pre=[rect.utils.transforms.z_score()], train_batch=int(args.batch))

if args.test:
    trainer.test(test_data, test_pre=[rect.utils.transforms.z_score()], 
              test_post=[rect.utils.transforms.Binary(), rect.utils.transforms.KeepLargestComponent()])

if args.classifier:
    class_train_data = rect.utils.io.ClassifyDataLoader(f_train)
    if args.val:
        class_val_data = rect.utils.io.ClassifyDataLoader(f_val)
    else:
        class_val_data = None

    class_model = rect.model.networks.MakeDenseNet(freeze_weights=False).to(device)
    class_trainer = rect.utils.train.ClassTrainer(class_model, outdir=os.path.join(args.odir, 'classlogs'),
                                         ensemble=None, early_stop=1000)

    class_trainer.train(class_train_data, class_val_data, train_batch=int(args.batch))

    threshRange = np.linspace(0, 0.6, 20)

    if args.test:
        for i, thresh in enumerate(threshRange):
            test_screen_data = rect.utils.io.PreScreenLoader(class_model.eval(), f_test, label='vote', threshold = thresh)
            trainer.test(test_screen_data, test_pre=[rect.utils.transforms.z_score()], 
                        test_post=[rect.utils.transforms.Binary(), rect.utils.transforms.KeepLargestComponent()], oname='class_thresh_{}'.format(i))

