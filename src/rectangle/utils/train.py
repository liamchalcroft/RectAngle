import torch
from torch import nn
from rectangle.utils.metrics import DiceLoss
from torch.optim import Adam
from copy import deepcopy
from os import path, makedirs
from datetime import date
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np


class Trainer(nn.Module):
    def __init__(self, model, nb_epochs=200, outdir='./logs',
     loss=DiceLoss(), metric=DiceLoss(), opt='adam',
     print_interval=50, val_interval=50, device='cuda',
     early_stop=10, ensemble=None):

        super().__init__()

        self.model = model
        self.nb_epochs = nb_epochs
        self.loss = loss
        self.metric = metric
        self.print_interval = print_interval
        self.val_interval = val_interval
        self.early_stop = early_stop
        self.ensemble = ensemble
        self.outdir = outdir
        self.device = device

        if self.ensemble == 0:
            self.ensemble = None

        if self.ensemble:
            if isinstance(self.model, list):
                if len(self.model) == self.ensemble:
                    self.model_ensemble = self.model
                else:
                    raise ValueError('List of models given but shape does not match ensemble.')
            else:
                self.model_ensemble = []
                for i in range(self.ensemble):
                    self.model_ensemble.append(deepcopy(model))

        if opt == 'adam':
            # if self.ensemble: 
            #     opt = [Adam(model.parameters()) for model in self.model_ensemble]
            # else:
            #     opt = Adam(model.parameters())
            opt = Adam(model.parameters())

        self.opt = opt


    # need to fix ensemble data - k-fold?
    def train(self, train_data, val_data=None, oname=None, 
        train_pre=None, train_post=None, train_batch=64, train_shuffle=True,
        val_pre=None, val_post=None, val_batch=8, val_shuffle=True):

        if self.ensemble:
            if val_data:
                print('Ensemble uses K-fold cross-val and generates\
                    val data - val_data input will be ignored.')
            train_list = []
            val_list = []
            length = int(len(train_data)//self.ensemble)
            length_list = [length] * self.ensemble
            if sum(length_list) != len(train_data):
                length_list[0] += len(train_data) - sum(length_list)
            data_list = random_split(train_data, length_list)
            for i in range(self.ensemble):
                val_list.append(DataLoader(data_list[i], val_batch, val_shuffle))
                train_list.append(DataLoader(ConcatDataset(data_list[:i]+data_list[i+1:]), 
                train_batch, train_shuffle))
        else:
            train = DataLoader(train_data, train_batch, train_shuffle)
            val = DataLoader(val_data, val_batch, val_shuffle)

        if not oname:
            oname = date.today()
            oname = oname.strftime("%b-%d-%Y")
        if self.ensemble:
            loss_log_ensemble = np.empty((len(self.model_ensemble),\
                self.nb_epochs))
            loss_log_ensemble[:] = np.nan
            dice_log_ensemble = np.empty((len(self.model_ensemble),\
                int(self.nb_epochs//self.val_interval)))
            dice_log_ensemble[:] = np.nan
            for i, model in enumerate(self.model_ensemble):
                early_ = 0
                train = train_list[i]
                val = val_list[i]
                print('Beginning training of model #{}'.format(i))
                for epoch in range(self.nb_epochs):
                    if self.early_stop:
                        if early_ == self.early_stop:
                            break
                    loss_epoch = []
                    model.train()
                    for step, (input, label) in enumerate(train):
                        input, label = input.to(self.device), label.to(self.device)
                        self.opt.zero_grad()
                        if train_pre:
                            for aug in train_pre:
                                input = aug(input)
                        pred = model(input)
                        if train_post:
                            for aug in train_post:
                                pred = aug(pred)
                        loss_ = self.loss(pred, label)
                        loss_.backward()
                        self.opt.step()
                        loss_epoch.append(loss_.item())
                    loss_log_ensemble[i,epoch] = np.mean(loss_epoch)
                    if epoch % self.print_interval == 0:
                        print('Epoch #{}: Mean Dice Loss: {}'.format(epoch, loss_log_ensemble[i,epoch]))
                    if epoch % self.val_interval == 0:
                        dice_epoch = []
                        model.eval()
                        with torch.no_grad():
                            for input, label in val:
                                input, label = input.to(self.device), label.to(self.device)
                                if val_pre:
                                    for aug in val_pre:
                                        input = aug(input)
                                pred = model(input)
                                if val_post:
                                    for aug in val_post:
                                        pred = aug(pred)
                                dice_metric = self.metric(pred, label)
                                dice_epoch.append(1 - dice_metric.item())
                            dice_log_ensemble[i,int(epoch//self.val_interval)] = np.mean(dice_epoch)
                        if epoch >= self.val_interval:
                            if dice_log_ensemble[i,int(epoch//self.val_interval)] > dice_log_ensemble[i,int(epoch//self.val_interval)-1]:
                                early_ = 0
                                path_ = path.join(self.outdir,\
                                    'model',oname,'ensemble/model_{}'.format(i))
                                if not path.exists(path_):
                                    makedirs(path_)
                                torch.save(model.state_dict(), path.join(path_,'{}.pth'.format(epoch)))
                            else:
                                early_ += 1
                        if epoch % self.print_interval == 0:
                            print('Mean Validation Dice: {}'.format(dice_log_ensemble[i,int(epoch//self.val_interval)]))
                print('Finished training of model #{}'.format(i))
        else:
            loss_log = np.empty(self.nb_epochs)
            loss_log[:] = np.nan
            dice_log = np.empty(int(self.nb_epochs//self.val_interval))
            dice_log[:] = np.nan
            early_ = 0
            model = self.model
            for epoch in range(self.nb_epochs):
                if self.early_stop:
                    if early_ == self.early_stop:
                        break
                loss_epoch = []
                model.train()
                for step, (input, label) in enumerate(train):
                    input, label = input.to(self.device), label.to(self.device)
                    self.opt.zero_grad()
                    if train_pre:
                        for aug in train_pre:
                            input = aug(input)
                    pred = model(input)
                    if train_post:
                        for aug in train_post:
                            pred = aug(pred)
                    loss_ = self.loss(pred, label)
                    loss_.backward()
                    self.opt.step()
                    loss_epoch.append(loss_.item())
                loss_log[epoch] = np.mean(loss_epoch)
                if epoch % self.print_interval == 0:
                    print('Epoch #{}: Mean Dice Loss: {}'.format(epoch, loss_log[epoch]))
                if epoch % self.val_interval == 0:
                    dice_epoch = []
                    model.eval()
                    with torch.no_grad():
                        for input, label in val:
                            input, label = input.to(self.device), label.to(self.device)
                            if val_pre:
                                for aug in val_pre:
                                    input = aug(input)
                            pred = model(input)
                            if val_post:
                                for aug in val_post:
                                    pred = aug(pred)
                            dice_metric = self.metric(pred, label)
                            dice_epoch.append(1 - dice_metric.item())
                        dice_log[int(epoch//self.val_interval)] = np.mean(dice_epoch)
                    if epoch % self.print_interval == 0:
                        print('Mean Validation Dice: {}'.format(dice_log[int(epoch//self.val_interval)]))
                    if epoch >= self.val_interval:
                        if dice_log[int(epoch//self.val_interval)] > dice_log[int(epoch//self.val_interval)-1]:
                            early_ = 0
                            path_ = path.join(self.outdir,\
                                'model',oname)
                            if not path.exists(path_):
                                makedirs(path_)
                            torch.save(model.state_dict(), path.join(path_,'{}.pth'.format(epoch)))
                        else:
                            early_ += 1
        print('\nTraining Complete')

        if self.ensemble:
            plt.figure(figsize=(8,6))
            plt.plot(np.linspace(0,self.nb_epochs,self.nb_epochs, dtype=int), \
                np.mean(loss_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,self.nb_epochs, dtype=int), \
                np.mean(loss_log_ensemble, axis=0)+np.std(loss_log_ensemble, axis=0),\
                np.mean(loss_log_ensemble, axis=0)-np.std(loss_log_ensemble, axis=0),\
                alpha=0.3)
            plt.plot(np.linspace(0,self.nb_epochs,\
                int(self.nb_epochs//self.val_interval), dtype=int), np.mean(dice_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,int(self.nb_epochs//self.val_interval), dtype=int), \
                np.mean(dice_log_ensemble, axis=0)+np.std(dice_log_ensemble, axis=0),\
                np.mean(dice_log_ensemble, axis=0)-np.std(dice_log_ensemble, axis=0),\
                alpha=0.3)
            plt.xlabel('Epoch #')
            plt.legend(['Train Loss', 'Validation Dice'])
            path_ = path.join(self.outdir,\
                                'training/plots')
            if not path.exists(path_):
                makedirs(path_)
            plt.savefig(path.join(path_, '{}.png'.format(oname)))
            path_ = path.join(self.outdir,\
                                'training/table')
            if not path.exists(path_):
                makedirs(path_)
            np.savetxt(path.join(path_, 'loss_{}.csv'.format(oname)),\
                loss_log_ensemble, delimiter=',')
            np.savetxt(path.join(path_, 'dice_{}.csv'.format(oname)),\
                dice_log_ensemble, delimiter=',')
        else:
            plt.figure(figsize=(8,6))
            plt.plot(np.linspace(0,self.nb_epochs,self.nb_epochs, dtype=int), loss_log)
            plt.plot(np.linspace(0,self.nb_epochs,\
                int(self.nb_epochs//self.val_interval), dtype=int), dice_log)
            plt.xlabel('Epoch #')
            plt.legend(['Train Loss', 'Validation Dice'])
            path_ = path.join(self.outdir,\
                                'training/plots')
            if not path.exists(path_):
                makedirs(path_)
            plt.savefig(path.join(path_, '{}.png'.format(oname)))
            path_ = path.join(self.outdir,\
                                'training/table')
            if not path.exists(path_):
                makedirs(path_)
            np.savetxt(path.join(path_, 'loss_{}.csv'.format(oname)),\
                loss_log, delimiter=',')
            np.savetxt(path.join(path_, 'dice_{}.csv'.format(oname)),\
                dice_log, delimiter=',')


    def test(self, test_data, oname=None, 
    test_pre=None, test_post=None):
        if not oname:
            oname = date.today()
            oname = oname.strftime("%b-%d-%Y")

        test = DataLoader(test_data, 1)
        dice_log = []
        self.model.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(test):
                input, label = input.to(self.device), label.to(self.device)
                if test_pre:
                    for aug in val_pre:
                        input = aug(input)
                if self.ensemble:
                    pred = [model(input) for model in self.model_ensemble]
                    pred = torch.cat(pred, dim=0)
                    pred = torch.mean(pred, dim=0)
                else:
                    pred = self.model(input)
                if test_post:
                    for aug in test_post:
                        pred = aug(pred)
                dice_metric = self.metric(pred, label)
                dice_log.append(1-dice_metric.item())

                input_img = input.detach().cpu().numpy()
                pred_img = pred.detach().cpu().numpy()
                label_img = label.detach().cpu().numpy()

                input_img = input_img[0,0,...]
                pred_img = pred_img[0,0,...]
                label_img = label_img[0,...]

                plt.figure()
                plt.subplot(131)
                plt.imshow(input_img, cmap='gray')
                plt.axis('off')
                plt.title('Image')
                plt.subplot(132)
                plt.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Prediction (DSC={:.2f})'.format(dice_log[i]))
                plt.subplot(133)
                plt.imshow(label_img, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.title('Ground Truth')

                path_ = path.join(self.outdir,\
                                'testing/plots')
                if not path.exists(path_):
                    makedirs(path_)
                plt.savefig(path.join(path_, 'pred{}_{}.png'.format(i, oname)))

        dice_log = np.array(dice_log)
        path_ = path.join(self.outdir,\
                                'testing/table')
        if not path.exists(path_):
            makedirs(path_)
        np.savetxt(path.join(path_, 'dice_{}.csv'.format(oname)),\
                dice_log, delimiter=',')

        print('Testing complete')
