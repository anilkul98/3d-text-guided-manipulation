import torch.nn as nn
import torch
from torch.nn.utils import weight_norm

class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.dec1_4 = nn.Sequential(weight_norm(nn.Linear(259,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    weight_norm(nn.Linear(512,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    weight_norm(nn.Linear(512,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    weight_norm(nn.Linear(512,253)), nn.ReLU(), nn.Dropout(dropout_prob))
                                    
        self.dec4_8 = nn.Sequential(weight_norm(nn.Linear(512,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    weight_norm(nn.Linear(512,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    weight_norm(nn.Linear(512,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    weight_norm(nn.Linear(512,512)), nn.ReLU(), nn.Dropout(dropout_prob),
                                    nn.Linear(512,1))

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.dec1_4(x_in)
        x = self.dec4_8(torch.cat([x, x_in], dim=1))
        
        return x
