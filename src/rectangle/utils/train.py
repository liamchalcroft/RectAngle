import torch
from torch import nn
from rectangle.utils.metrics import DiceLoss
from torch.optim import Adam
from copy import deepcopy
from os import path
from datetime import date
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset


class Trainer(nn.Module):
    def __init__(self, model, nb_epochs=200, outdir='./logs',
     loss=DiceLoss(), metric=DiceLoss(), opt='adam',
     print_interval=50, val_interval=50, device='cuda',
     early_stop=10, ensemble=None):

        self.model = model
        self.train = train
        self.val = val
        self.nb_epochs
        self.loss = loss
        self.metric = metric
        self.print_interval = print_interval
        self.val_interval = val_interval
        self.early_stop = early_stop
        self.ensemble = ensemble
        self.outdir = outdir

        if self.ensemble == 0:
            self.ensemble = None

        if self.ensemble:
            if len(model) != self.ensemble:
                self.model_ensemble = []
                for i in range(self.ensemble):
                    self.model_ensemble.append(deepcopy(model))
            else:
                self.model_ensemble = model

        if opt == 'adam':
            if self.ensemble: 
                opt = [Adam(model.parameters()) for model in self.model_ensemble]
            else:
                opt = Adam(model.parameters())


    # need to fix ensemble data - k-fold?
    def train(self, train_data, val_data=None, oname=None, 
        train_pre=None, train_post=None, train_batch=64, train_shuffle=True,
        val_pre=None, val_post=None, val_batch=64, val_shuffle=True):

        if self.ensemble:
            if val_data:
                print('Ensemble uses K-fold cross-val and generates\
                    val data - val_data input will be ignored.')
            train_list = []
            val_list = []
            length = int(len(train_data)//self.ensemble)
            length_list = [length] * self.ensemble
            data_list = random_split(train_data, length_list)
            for i in range(self.ensemble):
                list_k = data_list
                val_list.append(list_k[i])
                list_k.remove(list_k[i])
                train_list.append(ConcatDataset(list_k))
        else:
            train = DataLoader(train_data, train_batch, shuffle)
            val = DataLoader(val_data, train_batch, shuffle)

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
                        pred = self.model(input)
                        if train_post:
                            for aug in train_post:
                                pred = aug(pred)
                        loss_ = self.loss(pred, label)
                        loss_.backward()
                        optim.step()
                        loss_epoch.append(loss_.item())
                    loss_log_ensemble[i,epoch] = np.mean(loss_epoch)
                    if epoch % print_interval == 0:
                        print('Epoch #{}: Mean Dice Loss: {}'.format(epoch, mean_loss))
                    if epoch % self.val_interval == 0:
                        dice_epoch = []
                        model.eval()
                        with torch.no_grad():
                            for input, label in val:
                                input, label = input.to(self.device), label.to(self.device)
                                if val_pre:
                                    for aug in val_pre:
                                        input = aug(input)
                                pred = self.model(input)
                                if val_post:
                                    for aug in val_post:
                                        pred = aug(pred)
                                dice_metric = self.metric(pred, label)
                                dice_epoch.append(1 - dice_metric.item())
                            dice_log_ensemble[i,epoch] = np.mean(dice_epoch)
                            if dice_log[epoch] > dice_log[epoch-1]:
                                early_ = 0
                                torch.save(model.state_dict(), path.join(self.outdir,\
                                    'model',oname,'ensemble/{}'.format(i)))
                            else:
                                early_ += 1
                        if epoch % print_interval == 0:
                            print('Mean Validation Dice: {}'.format(dice_metric))
                print('Finished training of model #{}'.format(i))
                loss_log_ensemble.append(loss_log)
                dice_log_ensemble.append(dice_log)
        else:
            loss_log = np.empty((1,\
            self.nb_epochs))
            loss_log[:] = np.nan
            dice_log = np.empty((1,\
                int(self.nb_epochs//self.val_interval)))
            dice_log[:] = np.nan
            early_ = 0
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
                    pred = self.model(input)
                    if train_post:
                        for aug in train_post:
                            pred = aug(pred)
                    loss_ = self.loss(pred, label)
                    loss_.backward()
                    optim.step()
                    loss_epoch.append(loss_.item())
                loss_log[epoch] = np.mean(loss_epoch)
                if epoch % print_interval == 0:
                    print('Epoch #{}: Mean Dice Loss: {}'.format(epoch, mean_loss))
                if epoch % self.val_interval == 0:
                    dice_epoch = []
                    model.eval()
                    with torch.no_grad():
                        for input, label in val:
                            input, label = input.to(self.device), label.to(self.device)
                            if val_pre:
                                for aug in val_pre:
                                    input = aug(input)
                            pred = self.model(input)
                            if val_post:
                                for aug in val_post:
                                    pred = aug(pred)
                            dice_metric = self.metric(pred, label)
                            dice_epoch.append(1 - dice_metric.item())
                        dice_log[epoch] = np.mean(dice_epoch)
                    if epoch % print_interval == 0:
                        print('Mean Validation Dice: {}'.format(dice_metric))
                    if dice_log[epoch] > dice_log[epoch-1]:
                        early_ = 0
                        torch.save(model.state_dict(), path.join(self.outdir,'model',oname))
                    else:
                        early_ += 1
        print('\nTraining Complete')

        if self.ensemble:
            plt.figure(figsize=(8,6))
            plt.plot(np.linspace(0,self.nb_epochs,self.nb_epochs), \
                np.mean(loss_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,self.nb_epochs), \
                np.mean(loss_log_ensemble, axis=0)+np.std(loss_log_ensemble, axis=0),\
                np.mean(loss_log_ensemble, axis=0)-np.std(loss_log_ensemble, axis=0),\
                alpha=0.3)
            plt.plot(np.linspace(0,self.nb_epochs,\
                self.nb_epochs//self.val_interval), np.mean(dice_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,self.nb_epochs), \
                np.mean(dice_log_ensemble, axis=0)+np.std(dice_log_ensemble, axis=0),\
                np.mean(dice_log_ensemble, axis=0)-np.std(dice_log_ensemble, axis=0),\
                alpha=0.3)
            plt.xlabel('Epoch #')
            plt.legend(['Train Loss', 'Validation Dice'])
            plt.savefig(path.join(outdir,'training/plots/{}.png'.format(oname)))

            np.savetxt(path.join(outdir,'training/table/loss_{}.csv'.format(oname)),\
                loss_log_ensemble, delimiter=',')
            np.savetxt(path.join(outdir,'training/table/dice_{}.csv'.format(oname)),\
                dice_log_ensemble, delimiter=',')
        else:
            plt.figure(figsize=(8,6))
            plt.plot(np.linspace(0,self.nb_epochs,self.nb_epochs), loss_log)
            plt.plot(np.linspace(0,self.nb_epochs,\
                self.nb_epochs//self.val_interval), dice_log)
            plt.xlabel('Epoch #')
            plt.legend(['Train Loss', 'Validation Dice'])
            plt.savefig(path.join(outdir,'training/plots/{}.png'.format(oname)))

            np.savetxt(path.join(outdir,'training/table/loss_{}.csv'.format(oname)),\
                loss_log, delimiter=',')
            np.savetxt(path.join(outdir,'training/table/dice_{}.csv'.format(oname)),\
                dice_log, delimiter=',')


    def test(self, test_data, oname=None, 
    test_pre=None, test_post=None):
        if not oname:
            oname = date.today()
            oname = oname.strftime("%b-%d-%Y")

        test = DataLoader(test_data, 1)
        dice_log = []
        model.eval()
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
                label_img = label_img[0,0,...]

                plt.figure()
                plt.subplot(131)
                plt.imshow(input_img, cmap='gray')
                plt.title('Image')
                plt.subplot(132)
                plt.imshow(pred_img, cmap='gray')
                plt.title('Prediction')
                plt.subplot(131)
                plt.imshow(label_img, cmap='gray')
                plt.title('Ground Truth')
                plt.savefig(path.join(outdir,'testing/plot/{}_{}.csv'.format(i, oname)))

        dice_log = np.array(dice_log)
        np.savetxt(path.join(outdir,'testing/table/dice_{}.csv'.format(oname)),\
                dice_log, delimiter=',')

        print('Testing complete')
