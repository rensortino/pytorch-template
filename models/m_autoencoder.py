from torch import nn
from modules import Encoder, Decoder


class Autoencoder(nn.Module):
    def __init__(self, channels, width, height, hidden_size=256):

        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.channels = channels
        self.width = width
        self.height = height
        self.hidden_size = hidden_size

        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.reshape(x_hat.shape[0], self.channels, self.width, self.height)
        return x_hat