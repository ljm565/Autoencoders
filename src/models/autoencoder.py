import torch.nn as nn


class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.height = config.height
        self.width = config.width
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout
        self.color_channel = config.color_channel
        
        self.encoder = nn.Sequential(
            nn.Linear(self.height*self.width*self.color_channel, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.height*self.width*self.color_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent_variable = self.encoder(x)
        output = self.decoder(latent_variable)
        output = output.view(batch_size, -1, self.height, self.width)

        return output, latent_variable



class CAE(nn.Module):
    def __init__(self, config):
        super(CAE, self).__init__()
        self.height = config.height
        self.width = config.width
        self.color_channel = config.color_channel

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.color_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=self.color_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        latent_variable = self.encoder(x)
        output = self.decoder(latent_variable)

        return output, latent_variable