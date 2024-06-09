import os 
import gc
import sys
import copy
import time
import math
import random
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from trainer.build import get_model
from utils import RANK, LOGGER, colorstr, init_seeds
from utils.data_utils import DLoader
from utils.filesys_utils import *




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if self.is_ddp else 1
        self.dataloaders = {}
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path
        self.denoising = self.config.denoising

        # color channel init
        self.convert2grayscale = True if self.config.color_channel==3 and self.config.convert2grayscale else False
        self.color_channel = 1 if self.convert2grayscale else self.config.color_channel
        self.config.color_channel = self.color_channel
        

        # sanity check
        assert self.config.color_channel in [1, 3], colorstr('red', 'image channel must be 1 or 3, check your config..')
        assert self.config.model_type in ['AE', 'CAE'], colorstr('red', 'model must be AE or CAE, check your config..')

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args

        # init model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['validation']
        self.model = self._init_model(self.config, self.mode)
        
        
        # train params
        # self.batch_size = self.config.batch_size
        # self.epochs = self.config.epochs
        # self.lr = self.config.lr

        # split trainset to trainset and valset and make dataloaders
        if self.config.MNIST_train:
            # set to MNIST size
            self.config.width, self.config.height = 28, 28

            # init train, validation, test sets
            self.mnist_path = self.config.MNIST.path
            self.mnist_valset_proportion = self.config.MNIST.MNIST_valset_proportion
            self.trainset = dsets.MNIST(root=self.mnist_path, transform=transforms.ToTensor(), train=True, download=True)
            valset_l = int(len(self.trainset)*self.mnist_valset_proportion)
            trainset_l = len(self.trainset) - valset_l
            self.trainset, self.valset = random_split(self.trainset, [trainset_l, valset_l])
            self.testset = dsets.MNIST(root=self.mnist_path, transform=transforms.ToTensor(), train=False, download=True)
        else:
            pass
            self.trainset, self.valset, self.testset = DLoader(self.trainset), DLoader(self.valset), DLoader(self.testset)
        
        self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.dataloaders['val'] = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)
        if self.mode == 'test':
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)


        self.criterion = nn.MSELoss()
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            checkpoints = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return model

        # init model and tokenizer
        resume_success = False
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        model = get_model(config, self.device)

        # resume model or resume model after applying peft
        if do_resume and not resume_success:
            model = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
        return model

        
    def train(self):
        best_val_loss = float('inf') if not self.continuous else self.loss_data['best_val_loss']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)

            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss = 0
                for i, (x, _) in enumerate(self.dataloaders[phase]):
                    if self.denoising:
                        noise = torch.zeros_like(x)
                        noise = nn.init.normal_(noise, mean=self.config.noise_mean, std=self.config.noise_std)
                        x = x.to(self.device)
                        noise = noise.to(self.device)
                        noise_x = x + noise
                    else:
                        x = x.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output, latent_variable = self.model(noise_x) if self.denoising else self.model(x)
                        loss = self.criterion(output, x)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    total_loss += loss.item()*x.size(0)
                    if i % 100 == 0:
                        print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                print('{} loss: {:4f}\n'.format(phase, epoch_loss))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                if phase == 'val':
                    val_loss_history.append(epoch_loss)
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer)
            
            print("time: {} s\n".format(time.time() - start))

        print('best val loss: {:4f}, best epoch: {:d}\n'.format(best_val_loss, best_epoch))
        self.model.load_state_dict(best_model_wts)
        self.loss_data = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
        return self.model, self.loss_data
    

    def test(self, result_num, visualization):
        if result_num > len(self.dataloaders['test'].dataset):
            print('The number of results that you want to see are larger than total test set')
            sys.exit()
        
        
        # concatenate all testset for t-sne and results
        with torch.no_grad():
            total_x, total_output, total_latent_variable, total_y, total_noise_x = [], [], [], [], []
            test_loss = 0
            self.model.eval()
            for x, y in self.dataloaders['test']:
                if self.denoising:
                    noise = torch.zeros_like(x)
                    noise = nn.init.normal_(noise, mean=self.config.noise_mean, std=self.config.noise_std)
                    x = x.to(self.device)
                    noise = noise.to(self.device)
                    noise_x = x + noise
                else:
                    x = x.to(self.device)

                output, latent_variable = self.model(noise_x) if self.denoising else self.model(x)
                test_loss += self.criterion(output, x).item() * x.size(0)

                if self.denoising:
                    total_noise_x.append(noise_x.detach().cpu())
                total_x.append(x.detach().cpu())
                total_output.append(output.detach().cpu())
                total_latent_variable.append(latent_variable.detach().cpu())
                total_y.append(y.detach().cpu())
            
            if self.denoising:
                total_noise_x = torch.cat(tuple(total_noise_x), dim=0)
            total_x = torch.cat(tuple(total_x), dim=0)
            total_output = torch.cat(tuple(total_output), dim=0)
            total_latent_variable = torch.cat(tuple(total_latent_variable), dim=0)
            total_y = torch.cat(tuple(total_y), dim=0)
        print('testset loss: {}'.format(test_loss/len(self.dataloaders['test'].dataset)))


        # select random index of the data
        ids = set()
        while len(ids) != result_num:
            ids.add(random.randrange(len(total_output)))
        ids = list(ids)


        # save the result img 
        print('start result drawing')
        k = 0
        plt.figure(figsize=(7, 3*result_num))
        for id in ids:
            if self.denoising:
                orig = total_noise_x[id].squeeze(0) if self.color_channel == 1 else total_noise_x[id].permute(1, 2, 0)
            else:
                orig = total_x[id].squeeze(0) if self.color_channel == 1 else total_x[id].permute(1, 2, 0)
            out = total_output[id].squeeze(0) if self.color_channel == 1 else total_output[id].permute(1, 2, 0)
            plt.subplot(result_num, 2, 1+k)
            plt.imshow(orig, cmap='gray')
            plt.subplot(result_num, 2, 2+k)
            plt.imshow(out, cmap='gray')
            k += 2
        plt.savefig(self.config.base_path+'result/'+self.config.result_img_name)


        # visualization
        if visualization and not self.config.MNIST_train:
            print('Now visualization is possible only for MNIST dataset. You can revise the code for your own dataset and its label..')
            sys.exit()

        if visualization:        
            # latent variable visualization
            print('start visualizing the latent variables')
            np.random.seed(42)
            tsne = TSNE()
            total_latent_variable = total_latent_variable.view(total_latent_variable.size(0), -1)
            x_test_2D = tsne.fit_transform(total_latent_variable)
            x_test_2D = (x_test_2D - x_test_2D.min())/(x_test_2D.max() - x_test_2D.min())

            plt.figure(figsize=(10, 10))
            plt.scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=total_y.numpy())
            cmap = plt.cm.tab10
            image_positions = np.array([[1., 1.]])
            for index, position in enumerate(x_test_2D):
                dist = np.sum((position - image_positions) ** 2, axis=1)
                if np.min(dist) > 0.02: # if far enough from other images
                    image_positions = np.r_[image_positions, [position]]
                    imagebox = mpl.offsetbox.AnnotationBbox(
                        mpl.offsetbox.OffsetImage(torch.squeeze(total_x).cpu().numpy()[index], cmap='binary'),
                        position, bboxprops={'edgecolor': cmap(total_y.numpy()[index]), 'lw': 2})
                    plt.gca().add_artist(imagebox)
            plt.axis('off')
            plt.savefig(self.config.base_path+'result/'+self.config.visualization_img_name)

