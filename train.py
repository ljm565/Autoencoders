import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import copy
import time
import pickle
import math
from sklearn.manifold import TSNE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os 

from config import Config
from utils_func import save_checkpoint, make_img_data
from utils_data import DLoader
from model_autoencoder import AE, CAE



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.color_channel = self.config.color_channel
        assert self.color_channel in [1, 3]
        self.convert2grayscale = True if self.color_channel==3 and self.config.convert2grayscale else False
        self.color_channel = 1 if self.convert2grayscale else self.color_channel
        self.model_type = self.config.model_type
        assert self.model_type in ['AE', 'CAE']
        self.denoising = self.config.denoising

        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr


        # split trainset to trainset and valset and make dataloaders
        if self.config.MNIST_train:
            # for reproducibility
            torch.manual_seed(999)

            # set to MNIST size
            self.config.width, self.config.height = 28, 28

            self.MNIST_valset_proportion = self.config.MNIST_valset_proportion
            self.trainset = dsets.MNIST(root=self.base_path, transform=transforms.ToTensor(), train=True, download=True)
            self.trainset, self.valset = random_split(self.trainset, [len(self.trainset)-int(len(self.trainset)*self.MNIST_valset_proportion), int(len(self.trainset)*self.MNIST_valset_proportion)])
            self.testset = dsets.MNIST(root=self.base_path, transform=transforms.ToTensor(), train=False, download=True)
        else:
            os.makedirs(self.base_path+'data', exist_ok=True)

            if os.path.isdir(self.base_path+'data/'+self.config.data_name):
                with open(self.base_path+'data/'+self.config.data_name+'/train.pkl', 'rb') as f:
                    self.trainset = pickle.load(f)
                with open(self.base_path+'data/'+self.config.data_name+'/val.pkl', 'rb') as f:
                    self.valset = pickle.load(f)
                with open(self.base_path+'data/'+self.config.data_name+'/test.pkl', 'rb') as f:
                    self.testset = pickle.load(f)
            else:
                os.makedirs(self.base_path+'data/'+self.config.data_name, exist_ok=True)
                self.trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                transforms.Resize((self.config.height, self.config.width)),
                                                transforms.ToTensor()]) if self.convert2grayscale else \
                            transforms.Compose([transforms.Resize((self.config.height, self.config.width)),
                                                transforms.ToTensor()]) 
                self.custom_data_proportion = self.config.custom_data_proportion
                assert math.isclose(sum(self.custom_data_proportion), 1)
                assert len(self.custom_data_proportion) <= 3
                
                if len(self.custom_data_proportion) == 3:
                    data = make_img_data(self.config.train_data_path, self.trans)
                    self.train_len, self.val_len = int(len(data)*self.custom_data_proportion[0]), int(len(data)*self.custom_data_proportion[1])
                    self.test_len = len(data) - self.train_len - self.val_len
                    self.trainset, self.valset, self.testset = random_split(data, [self.train_len, self.val_len, self.test_len], generator=torch.Generator().manual_seed(999))

                elif len(self.custom_data_proportion) == 2:
                    data1 = make_img_data(self.config.train_data_path, self.trans)
                    data2 = make_img_data(self.config.test_data_path, self.trans)
                    if self.config.two_folders == ['train', 'val']:
                        self.train_len = int(len(data1)*self.custom_data_proportion[0]) 
                        self.val_len = len(data1) - self.train_len
                        self.trainset, self.valset = random_split(data1, [self.train_len, self.val_len], generator=torch.Generator().manual_seed(999))
                        self.testset = data2
                    elif self.config.two_folders == ['val', 'test']:
                        self.trainset = data1
                        self.val_len = int(len(data2)*self.custom_data_proportion[0]) 
                        self.test_len = len(data2) - self.val_len
                        self.valset, self.testset = random_split(data2, [self.val_len, self.test_len], generator=torch.Generator().manual_seed(999))
                    else:
                        print("two folders must be ['train', 'val] or ['val', 'test']")
                        raise AssertionError

                elif len(self.custom_data_proportion) == 1:
                    self.trainset = make_img_data(self.config.train_data_path, self.trans)
                    self.valset = make_img_data(self.config.val_data_path, self.trans)
                    self.testset = make_img_data(self.config.test_data_path, self.trans)
                
                with open(self.base_path+'data/'+self.config.data_name+'/train.pkl', 'wb') as f:
                    pickle.dump(self.trainset, f)
                with open(self.base_path+'data/'+self.config.data_name+'/val.pkl', 'wb') as f:
                    pickle.dump(self.valset, f)
                with open(self.base_path+'data/'+self.config.data_name+'/test.pkl', 'wb') as f:
                    pickle.dump(self.testset, f)

            self.trainset, self.valset, self.testset = DLoader(self.trainset), DLoader(self.valset), DLoader(self.testset)
        
        self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.dataloaders['val'] = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)
        if self.mode == 'test':
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)


        if self.model_type == 'AE':
            self.model = AE(self.config, self.color_channel).to(self.device)
        elif self.model_type == 'CAE':
            self.model = CAE(self.config, self.color_channel).to(self.device)
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
                        print('model pt file is being saved\n')
                        save_checkpoint(self.model_path, self.model, self.optimizer)
            
            print("time: {} s\n".format(time.time() - start))

        print('best val loss: {:4f}, best epoch: {:d}\n'.format(best_val_loss, best_epoch))
        self.model.load_state_dict(best_model_wts)
        self.loss_data = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
        return self.model, self.loss_data
    

    def test(self, result_num, visualization):
        if result_num > len(self.dataloaders['test'].dataset):
            print('The number of results that you want to see are larger than total test set')
            raise AssertionError
        
        
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
        while 1:
            if len(ids) == result_num:
                break
            id = np.random.randint(1, high=len(total_output))
            ids.add(id)
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
            raise AssertionError

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

