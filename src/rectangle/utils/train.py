import torch
from torch import nn
from rectangle.utils.metrics import DiceLoss, Precision, Recall, Accuracy
from torch.optim import Adam
from copy import deepcopy
from os import path, makedirs
from datetime import date
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from scipy.ndimage import laplace
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Trainer(nn.Module):
    def __init__(self, model, nb_epochs=200, outdir='./logs',
     loss=DiceLoss(), metric=DiceLoss(), opt='adam',
     print_interval=1, val_interval=1, device='cuda',
     early_stop=10, lr_schedule=None, ensemble=None): 

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

            self.writer = [SummaryWriter(log_dir=path.join(outdir,'runs/model_{}'.format(i))) for i in range(self.ensemble)]
        else:
            self.writer = SummaryWriter(log_dir=path.join(outdir,'runs'))

        if opt == 'adam':
            if self.ensemble: 
                opt = [Adam(model.parameters()) for model in self.model_ensemble]
            else:
                opt = Adam(model.parameters())
            # opt = Adam(model.parameters())

        self.opt = opt

        if lr_schedule:
            if lr_schedule not in ['lambda', 'exponential', 'reduce_on_plateau']:
                raise ValueError('Available learning rate schedules are LambdaLR, ExponentialLR or ReduceLROnPlateau.')
            elif self.ensemble:
                lr_schedule = [lr_schedule for model in self.model_ensemble]

        self.lr_schedule = lr_schedule

    def train(self, train_data, val_data=None, oname=None, 
        train_pre=None, train_post=None, train_batch=128, train_shuffle=True,
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
                dice_max = 0
                train = train_list[i]
                val = val_list[i]
                opt_ = self.opt[i]
                writer_ = self.writer[i]
                if self.lr_schedule:
                    lr_schedule_ = self.lr_schedule[i]
                    if lr_schedule_ == 'lambda':
                        lr_schedule_ = torch.optim.lr_scheduler.LambdaLR(opt_, lambda epoch: 0.95 ** epoch)
                    elif lr_schedule_ == 'exponential':
                        lr_schedule_ = torch.optim.lr_scheduler.ExponentialLR(opt_, 0.95)
                    else:
                        lr_schedule_ = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_)
                else:
                    lr_schedule_ = None

                print('Beginning training of model #{}'.format(i))
                for epoch in range(self.nb_epochs):
                    if self.early_stop:
                        if early_ == self.early_stop:
                            break
                    loss_epoch = []
                    model.train()
                    for step, (input, label) in enumerate(train):
                        input, label = input.to(self.device), label.to(self.device)
                        opt_.zero_grad()
                        if train_pre:
                            for aug in train_pre:
                                input = aug(input)
                        pred = model(input)
                        if train_post:
                            for aug in train_post:
                                pred = aug(pred)
                        loss_ = self.loss(pred, label)
                        loss_.backward()
                        opt_.step()
                        loss_epoch.append(loss_.item())
                    if lr_schedule_ and lr_schedule_ != 'reduce_on_plateau':
                        lr_schedule_.step()
                    loss_log_ensemble[i,epoch] = np.nanmean(loss_epoch)
                    if epoch % self.print_interval == 0:
                        writer_.add_scalar('train/dice_loss_ensemble', loss_, epoch)
                        writer_.add_scalar('train/dice_coefficient_ensemble', 1-loss_, epoch)
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
                            dice_log_ensemble[i,int(epoch//self.val_interval)] = np.nanmean(dice_epoch)
                            if lr_schedule_ == 'reduce_on_plateau':
                                lr_schedule_.step(1-np.nanmean(dice_epoch))

                            writer_.add_scalar('val/dice_loss_ensemble', dice_metric, epoch)
                            writer_.add_scalar('val/dice_coefficient_ensemble', 1-dice_metric, epoch)

                            ## show some (e.g.,10) example images in tensorboard
                            ex_num = 10
                            ex_label = label[:ex_num,0]
                            ex_pred = pred[:ex_num,0]
                            ex_image = torch.cat([ex_label,ex_pred], dim=2)

                            ex_images = ex_image.reshape(-1,ex_image.shape[2])
                            image_grid = (make_grid(ex_images, nrow=ex_num)[0]+0.5)/ex_num                     
                            writer_.add_images(
                                "val/example_images_ensemble",
                                image_grid,
                                epoch,
                                dataformats="HW",
                            )

                        if epoch >= self.val_interval:
                            if dice_log_ensemble[i,int(epoch//self.val_interval)] > dice_max:
                                early_ = 0
                                dice_max = dice_log_ensemble[i,int(epoch//self.val_interval)]
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
            dice_max = 0
            model = self.model
            if self.lr_schedule:
                if self.lr_schedule == 'lambda':
                    self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda epoch: 0.95 ** epoch)
                elif self.lr_schedule == 'exponential':
                    self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.95)
                else:
                    self.lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)

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
                    if self.lr_schedule and self.lr_schedule != 'reduce_on_plateau':
                        self.lr_schedule.step()
                    loss_epoch.append(loss_.item())
                loss_log[epoch] = np.nanmean(loss_epoch)
                if epoch % self.print_interval == 0:
                    self.writer.add_scalar('train/dice_loss', loss_, epoch)
                    self.writer.add_scalar('train/dice_coefficient', 1-loss_, epoch)
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
                            if self.lr_schedule == 'reduce_on_plateau':
                                self.lr_schedule.step(dice_metric) # monitors validation loss 
                            dice_epoch.append(1 - dice_metric.item())
                        dice_log[int(epoch//self.val_interval)] = np.nanmean(dice_epoch)

                        self.writer.add_scalar('val/dice_loss', dice_metric, epoch)
                        self.writer.add_scalar('val/dice_coefficient', 1-dice_metric, epoch)

                        ## show some (e.g.,10) example images in tensorboard
                        ex_num = 10
                        ex_label = label[:ex_num,0]
                        ex_pred = pred[:ex_num,0]
                        ex_image = torch.cat([ex_label,ex_pred], dim=2)

                        ex_images = ex_image.reshape(-1,ex_image.shape[2])
                        image_grid = (make_grid(ex_images, nrow=ex_num)[0]+0.5)/ex_num                       
                        self.writer.add_images(
                            "val/example_images",
                            image_grid,
                            epoch,
                            dataformats="HW",
                        )

                    if epoch % self.print_interval == 0:
                        print('Mean Validation Dice: {}'.format(dice_log[int(epoch//self.val_interval)]))
                    if epoch >= self.val_interval:
                        if dice_log[int(epoch//self.val_interval)] > dice_max:
                            early_ = 0
                            dice_max = dice_log[int(epoch//self.val_interval)]
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
                np.nanmean(loss_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,self.nb_epochs, dtype=int), \
                np.nanmean(loss_log_ensemble, axis=0)+np.nanstd(loss_log_ensemble, axis=0),\
                np.nanmean(loss_log_ensemble, axis=0)-np.nanstd(loss_log_ensemble, axis=0),\
                alpha=0.3)
            plt.plot(np.linspace(0,self.nb_epochs,\
                int(self.nb_epochs//self.val_interval), dtype=int), np.nanmean(dice_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,int(self.nb_epochs//self.val_interval), dtype=int), \
                np.nanmean(dice_log_ensemble, axis=0)+np.nanstd(dice_log_ensemble, axis=0),\
                np.nanmean(dice_log_ensemble, axis=0)-np.nanstd(dice_log_ensemble, axis=0),\
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
    test_pre=None, test_post=None, overlap='contour'):
        if not oname:
            oname = date.today()
            oname = oname.strftime("%b-%d-%Y")

        test = DataLoader(test_data, 1, shuffle=False)
        dice_log = []
        prec_log = []
        rec_log = []
        precision = Precision()
        recall = Recall()
        self.model.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(test):
                input, label = input.to(self.device), label.to(self.device)
                if test_pre:
                    for aug in test_pre:
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
                dice_log.append(1-dice_metric.item().detach().cpu().numpy())
                prec_log.append(precision(pred, label).detach().cpu().numpy())
                rec_log.append(recall(pred, label).detach().cpu().numpy())

                input_img = input.detach().cpu().numpy()
                pred_img = pred.detach().cpu().numpy()
                label_img = label.detach().cpu().numpy()

                input_img = np.squeeze(input_img)
                pred_img = np.squeeze(pred_img)
                label_img = np.squeeze(label_img)

                if overlap=='contour':
                    input_img -= input_img.min()
                    input_img *= 1.0/input_img.max()
                    label_img = laplace(label_img)
                    pred_img = laplace(pred_img)
                    label_img = (label_img != 0)
                    pred_img = (pred_img != 0)
                    label_img = np.ma.masked_where(label_img == 0, label_img)
                    pred_img = np.ma.masked_where(pred_img == 0, pred_img)
                    plt.figure()
                    plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.imshow(label_img, cmap='Greens', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.imshow(pred_img, cmap='Reds', vmin=0, vmax=1)
                    plt.axis('off')
                elif overlap=='mask':
                    input_img -= input_img.min()
                    input_img *= 1.0/input_img.max()
                    label_img = np.ma.masked_where(label_img == 0, label_img)
                    pred_img = np.ma.masked_where(pred_img == 0, pred_img)
                    plt.figure()
                    plt.imshow(input_img, cmap='gray', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.imshow(label_img, cmap='Greens', vmin=0, vmax=1)
                    plt.axis('off')
                    plt.imshow(pred_img, cmap='Reds', vmin=0, vmax=1)
                    plt.axis('off')
                else:
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

        dice_log = np.array(dice_log, dtype=float)
        prec_log = np.array(prec_log, dtype=float)
        rec_log = np.array(rec_log, dtype=float)

        plt.figure()
        plt.scatter(rec_log, prec_log)
        plt.plot([0,0.5,1], [0.5,0.5,0.5], '--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('AUC = {:.2f}'.format(np.sum(prec_log * rec_log)/np.size(prec_log)))
        plt.savefig(path.join(path_, 'prec_rec_{}'.format(oname)))


        print('Mean Dice score: {:.2f}±{:.3f}, Mean Precision: {:.2f}±{:.3f}, Mean Recall: {:.2f}±{:.3f}'.format(np.mean(dice_log), np.std(dice_log), np.mean(prec_log), np.std(prec_log), np.mean(rec_log), np.std(rec_log)))
        path_ = path.join(self.outdir,\
                                'testing/table')
        if not path.exists(path_):
            makedirs(path_)
        np.savetxt(path.join(path_, 'dice_{}.csv'.format(oname)),\
                dice_log, delimiter=',')
        np.savetxt(path.join(path_, 'precision_{}.csv'.format(oname)),\
                prec_log, delimiter=',')
        np.savetxt(path.join(path_, 'recall_{}.csv'.format(oname)),\
                rec_log, delimiter=',')

        print('Testing complete')


class ClassTrainer(nn.Module):
    def __init__(self, model, nb_epochs=200, outdir='./logs',
     loss=nn.BCELoss(), metric=Accuracy(), opt='adam',
     print_interval=1, val_interval=5, device='cuda',
     early_stop=5, ensemble=None):

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
            if self.ensemble: 
                opt = [Adam(model.parameters()) for model in self.model_ensemble]
            else:
                opt = Adam(model.parameters())
            # opt = Adam(model.parameters())

        self.opt = opt

        self.writer = SummaryWriter(log_dir=path.join(outdir,'runs'))


    def train(self, train_data, val_data=None, oname=None, 
        train_pre=None, train_post=None, train_batch=128, train_shuffle=True,
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
            acc_log_ensemble = np.empty((len(self.model_ensemble),\
                int(self.nb_epochs//self.val_interval)))
            acc_log_ensemble[:] = np.nan
            for i, model in enumerate(self.model_ensemble):
                early_ = 0
                acc_max = 0
                train = train_list[i]
                val = val_list[i]
                opt_ = self.opt[i]
                print('Beginning training of model #{}'.format(i))
                for epoch in range(self.nb_epochs):
                    if self.early_stop:
                        if early_ == self.early_stop:
                            break
                    loss_epoch = []
                    model.train()
                    for step, (input, label) in enumerate(train):
                        input, label = input.to(self.device), label.to(self.device)
                        opt_.zero_grad()
                        if train_pre:
                            for aug in train_pre:
                                input = aug(input)
                        pred = model(input)
                        if train_post:
                            for aug in train_post:
                                pred = aug(pred)
                        loss_ = self.loss(pred, label)
                        loss_.backward()
                        opt_.step()
                        loss_epoch.append(loss_.item())
                    loss_log_ensemble[i,epoch] = np.nanmean(loss_epoch)
                    if epoch % self.print_interval == 0:
                        self.writer.add_scalar('class_train/dice_loss_ensemble', loss_, epoch) 
                        self.writer.add_scalar('class_train/dice_coefficient_ensemble', 1-loss_, epoch)
                        print('Epoch #{}: Mean acc Loss: {}'.format(epoch, loss_log_ensemble[i,epoch]))
                    if epoch % self.val_interval == 0:
                        acc_epoch = []
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
                                acc_metric = self.metric(pred, label)
                                acc_epoch.append(acc_metric)
                            acc_log_ensemble[i,int(epoch//self.val_interval)] = np.nanmean(acc_epoch)
                            self.writer.add_scalar('class_val/dice_loss_ensemble', acc_metric, epoch) 
                            self.writer.add_scalar('class_val/dice_coefficient_ensemble', 1-acc_metric, epoch)
                        if epoch >= self.val_interval:
                            if acc_log_ensemble[i,int(epoch//self.val_interval)] > acc_max:
                                early_ = 0
                                acc_max = acc_log_ensemble[i,int(epoch//self.val_interval)]
                                path_ = path.join(self.outdir,\
                                    'model',oname,'ensemble/model_{}'.format(i))
                                if not path.exists(path_):
                                    makedirs(path_)
                                torch.save(model.state_dict(), path.join(path_,'{}.pth'.format(epoch)))
                            else:
                                early_ += 1
                        if epoch % self.print_interval == 0:
                            print('Mean Validation acc: {}'.format(acc_log_ensemble[i,int(epoch//self.val_interval)]))
                print('Finished training of model #{}'.format(i))
        else:
            loss_log = np.empty(self.nb_epochs)
            loss_log[:] = np.nan
            acc_log = np.empty(int(self.nb_epochs//self.val_interval))
            acc_log[:] = np.nan
            early_ = 0
            acc_max = 0
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
                loss_log[epoch] = np.nanmean(loss_epoch)
                if epoch % self.print_interval == 0:
                    self.writer.add_scalar('class_train/dice_loss', loss_, epoch) 
                    self.writer.add_scalar('class_train/dice_coefficient', 1-loss_, epoch)
                    print('Epoch #{}: Mean acc Loss: {}'.format(epoch, loss_log[epoch]))
                if epoch % self.val_interval == 0:
                    acc_epoch = []
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
                            acc_metric = self.metric(pred, label)
                            acc_epoch.append(acc_metric)
                        acc_log[int(epoch//self.val_interval)] = np.nanmean(acc_epoch)
                        self.writer.add_scalar('class_val/dice_loss', acc_metric, epoch) 
                        self.writer.add_scalar('class_val/dice_coefficient', 1-acc_metric, epoch)
                    if epoch % self.print_interval == 0:
                        print('Mean Validation acc: {}'.format(acc_log[int(epoch//self.val_interval)]))
                    if epoch >= self.val_interval:
                        if acc_log[int(epoch//self.val_interval)] > acc_max:
                            early_ = 0
                            acc_max = acc_log[int(epoch//self.val_interval)]
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
                np.nanmean(loss_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,self.nb_epochs, dtype=int), \
                np.nanmean(loss_log_ensemble, axis=0)+np.nanstd(loss_log_ensemble, axis=0),\
                np.nanmean(loss_log_ensemble, axis=0)-np.nanstd(loss_log_ensemble, axis=0),\
                alpha=0.3)
            plt.plot(np.linspace(0,self.nb_epochs,\
                int(self.nb_epochs//self.val_interval), dtype=int), np.nanmean(acc_log_ensemble, axis=0))
            plt.fill_between(np.linspace(0,self.nb_epochs,int(self.nb_epochs//self.val_interval), dtype=int), \
                np.nanmean(acc_log_ensemble, axis=0)+np.nanstd(acc_log_ensemble, axis=0),\
                np.nanmean(acc_log_ensemble, axis=0)-np.nanstd(acc_log_ensemble, axis=0),\
                alpha=0.3)
            plt.xlabel('Epoch #')
            plt.legend(['Train Loss', 'Validation acc'])
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
            np.savetxt(path.join(path_, 'acc_{}.csv'.format(oname)),\
                acc_log_ensemble, delimiter=',')
        else:
            plt.figure(figsize=(8,6))
            plt.plot(np.linspace(0,self.nb_epochs,self.nb_epochs, dtype=int), loss_log)
            plt.plot(np.linspace(0,self.nb_epochs,\
                int(self.nb_epochs//self.val_interval), dtype=int), acc_log)
            plt.xlabel('Epoch #')
            plt.legend(['Train Loss', 'Validation acc'])
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
            np.savetxt(path.join(path_, 'acc_{}.csv'.format(oname)),\
                acc_log, delimiter=',')
