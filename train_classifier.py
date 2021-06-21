import os
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

parser.add_argument('--ensemble',
                    '--en',
                    metavar='ensemble',
                    type=str,
                    action='store',
                    default=None,
                    help='Number of ensembled models.')

parser.add_argument('--freeze',
                    '--f',
                    metavar='freeze',
                    type=bool,
                    action='store',
                    default=False,
                    help='Freeze CNN weights (pre-trained on ImageNet).')

parser.add_argument('--odir',
                    '--o',
                    metavar='odir',
                    type=str,
                    action='store',
                    default='./',
                    help='Path to output folder.')

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

# set seeds for repeatable results
if args.seed:
    seed = int(args.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

f_train = h5py.File(args.train, 'r')

if args.val:
    f_val = h5py.File(args.val, 'r')

if args.test:
    f_test = h5py.File(args.test, 'r')

class_train_data = rect.utils.io.ClassifyDataLoader(f_train)
if args.val:
    class_val_data = rect.utils.io.ClassifyDataLoader(f_val)
else:
    class_val_data = None
if args.test:
    class_test_data = rect.utils.io.ClassifyDataLoader(f_test)
else:
    class_test_data = None

class_model = rect.model.networks.MakeDenseNet(freeze_weights=args.freeze).to(device)
class_trainer = rect.utils.train.ClassTrainer(class_model, outdir=os.path.join(args.odir),
                                        ensemble=ensemble, nb_epochs=int(args.epochs))

class_trainer.train(class_train_data, class_val_data, train_batch=int(args.batch))
