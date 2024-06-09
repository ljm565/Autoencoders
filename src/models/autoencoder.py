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