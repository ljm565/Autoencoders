import gc
import sys
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist

from tools import TrainingLogger
from trainer.build import get_model, get_data_loader
from utils import RANK, LOGGER, colorstr, init_seeds
from utils.training_utils import init_progress_bar
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

        # init model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['train', 'validation', 'test']
        self.model = self._init_model(self.config, self.mode)
        self.dataloaders = get_data_loader(self.config, self.modes, self.is_ddp)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
        # init criterion, optimizer, etc.
        self.epochs = self.config.epochs
        self.criterion = nn.MSELoss()
        if self.is_training_mode:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return model

        # init model and tokenizer
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        model = get_model(config, self.device)

        # resume model or resume model after applying peft
        if do_resume:
            model = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        
        return model


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()

            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.model.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            pbar = init_progress_bar(train_loader, self.is_rank_zero, ['MSE Loss'], nb)

        for i, (x, _) in pbar:
            self.train_cur_step += 1
            batch_size = x.size(0)
            x = x.to(self.device)
            if self.denoising:
                noise = torch.zeros_like(x)
                noise = nn.init.normal_(noise, mean=self.config.noise_mean, std=self.config.noise_std)
                noise = noise.to(self.device)
                noise_x = x + noise
                
            self.optimizer.zero_grad()
            output, latent_variable = self.model(noise_x) if self.denoising else self.model(x)
            loss = self.criterion(output, x)
            loss.backward()
            self.optimizer.step()

            if self.is_rank_zero:
                self.training_logger.update(phase, epoch+1, self.train_cur_step, batch_size, **{'train_loss': loss.item()})
                loss_log = [loss.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
            
        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        with torch.no_grad():
            if self.is_rank_zero:
                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                pbar = init_progress_bar(val_loader, self.is_rank_zero, ['MSE Loss'], nb)

                self.model.eval()

                for i, (x, _) in pbar:
                    batch_size = x.size(0)
                    x = x.to(self.device)
                    if self.denoising:
                        noise = torch.zeros_like(x)
                        noise = nn.init.normal_(noise, mean=self.config.noise_mean, std=self.config.noise_std)
                        noise = noise.to(self.device)
                        noise_x = x + noise
                    
                    output, latent_variable = self.model(noise_x) if self.denoising else self.model(x)
                    loss = self.criterion(output, x)

                    self.training_logger.update(
                        phase, 
                        epoch, 
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'validation_loss': loss.item()}, 
                    )

                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                    pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
                
                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(self.wdir, self.model)
                    self.training_logger.save_logs(self.save_dir)
        

    def tsne_test(self, phase, result_num):
        if result_num > len(self.dataloaders[phase].dataset):
            LOGGER.info(colorstr('red', 'The number of results that you want to see are larger than total test set'))
            sys.exit()

        # make directory
        vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs') 
        os.makedirs(vis_save_dir, exist_ok=True)
        
        # concatenate all testset for t-sne and results
        with torch.no_grad():
            total_x, total_output, total_latent_variable, total_y, total_noise_x = [], [], [], [], []
            test_loss = 0
            self.model.eval()
            for x, y in self.dataloaders[phase]:
                x = x.to(self.device)
                if self.denoising:
                    noise = torch.zeros_like(x)
                    noise = nn.init.normal_(noise, mean=self.config.noise_mean, std=self.config.noise_std)
                    noise = noise.to(self.device)
                    noise_x = x + noise

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

        LOGGER.info(colorstr('green', f'testset loss: {test_loss/len(self.dataloaders[phase].dataset)}'))

        # select random index of the data
        ids = random.sample(range(len(total_output)), result_num)

        # save the result img 
        LOGGER.info('start result drawing')
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

        plt.savefig(os.path.join(vis_save_dir, 'autoencoder_result.png'))

        # t-sne visualization
        if not self.config.MNIST_train:
            LOGGER.info(colorstr('red', 'Now visualization is possible only for MNIST dataset. You can revise the code for your own dataset and its label..'))
            sys.exit()

        # latent variable visualization
        LOGGER.info('start visualizing the latent space')
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
        plt.savefig(os.path.join(vis_save_dir, 'tsne_result.png'))

