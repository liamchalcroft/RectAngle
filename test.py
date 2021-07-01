import os
import argparse

parser = argparse.ArgumentParser(prog='test',
                                description="Test RectAngle model. See list of available arguments for more info.")

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

parser.add_argument('--weights',
                    '--w',
                    metavar='weights',
                    type=str,
                    nargs='*',
                    action='store',
                    default=None,
                    help='Path to saved model weights.')

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

parser.add_argument('--classifier',
                    '--c',
                    metavar='classifier',
                    type=bool,
                    action='store',
                    default=False,
                    help='Use of classifier for pre-screening. If selected will train without and then perform test without + with.')

parser.add_argument('--classweights',
                    '--cw',
                    metavar='classweights',
                    type=str,
                    action='store',
                    default=None,
                    help='Path to trained weights for classifier.')

parser.add_argument('--threshold',
                    '--th',
                    metavar='threshold',
                    type=str,
                    action='store',
                    default='0.5',
                    help='Activation threshold for classifier.')

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
    if args.weights:
        ensemble=int(len(args.weights))
    else:
        ensemble = 1

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

model = [rect.model.networks.UNet(n_layers=int(args.depth), device=device,
                                    gate=args.gate) for e in range(ensemble)]

for n, m in enumerate(model):
    m.load_state_dict(torch.load(args.weights[n], map_location=device))

if args.classifier==True:
    class_model = rect.model.networks.MakeDenseNet(freeze_weights=False).to(device)
if args.classweights:
    class_model.load_state_dict(torch.load(args.classweights))

f_test = h5py.File(args.test, 'r')
if args.classifier:
    test_data = rect.utils.io.PreScreenLoader(class_model.eval(), f_test, label=args.label, threshold=float(args.thresh))
else:
    test_data = rect.utils.io.H5DataLoader(f_test, label='vote')

trainer = rect.utils.train.Trainer(model, ensemble=ensemble, outdir=args.odir, device=device)

trainer.test(test_data, test_pre=[rect.utils.transforms.z_score()], oname='run', 
            test_post=[rect.utils.transforms.Binary()], overlap='contour')
