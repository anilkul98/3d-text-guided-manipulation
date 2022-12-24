import torch

class _G(torch.nn.Module):
  def __init__(self, dim, latent_len):
    super(_G, self).__init__()
    self.dim = dim
    self.latent_len = latent_len
    
    padd = (0, 0, 0)
    if self.dim == 32:
        padd = (1,1,1)

    self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.latent_len, self.dim*8, kernel_size=4, stride=2, padding=padd),
            torch.nn.BatchNorm3d(self.dim*8),
            torch.nn.ReLU()
        )
    self.layer2 = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(self.dim*8, self.dim*4, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim*4),
        torch.nn.ReLU()
    )
    self.layer3 = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(self.dim*4, self.dim*2, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim*2),
        torch.nn.ReLU()
    )
    self.layer4 = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(self.dim*2, self.dim, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim),
        torch.nn.ReLU()
    )
    self.layer5 = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(self.dim, 1, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.Sigmoid()
    )
    
  
  def forward(self, x):
    out = x.view(-1, self.latent_len, 1, 1, 1)
    # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
    out = self.layer1(out)
    # print(out.size())  # torch.Size([100, 512, 4, 4, 4])
    out = self.layer2(out)
    # print(out.size())  # torch.Size([100, 256, 8, 8, 8])
    out = self.layer3(out)
    # print(out.size())  # torch.Size([100, 128, 16, 16, 16])
    out = self.layer4(out)
    # print(out.size())  # torch.Size([100, 64, 32, 32, 32])
    out = self.layer5(out)
    # print(out.size())  # torch.Size([100, 1, 64, 64, 64])

    return out
  
class _D(torch.nn.Module):
  def __init__(self, dim, latent_len):
    super(_D, self).__init__()
    self.dim = dim
    self.latent_len = latent_len
    

    padd = (0,0,0)
    if self.dim == 32:
      padd = (1,1,1)

    self.layer1 = torch.nn.Sequential(
        torch.nn.Conv3d(1, self.dim, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim),
        torch.nn.LeakyReLU(negative_slope=0.2)
    )
    self.layer2 = torch.nn.Sequential(
        torch.nn.Conv3d(self.dim, self.dim*2, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim*2),
        torch.nn.LeakyReLU(negative_slope=0.2)
    )
    self.layer3 = torch.nn.Sequential(
        torch.nn.Conv3d(self.dim*2, self.dim*4, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim*4),
        torch.nn.LeakyReLU(negative_slope=0.2)
    )
    self.layer4 = torch.nn.Sequential(
        torch.nn.Conv3d(self.dim*4, self.dim*8, kernel_size=4, stride=2, padding=(1, 1, 1)),
        torch.nn.BatchNorm3d(self.dim*8),
        torch.nn.LeakyReLU(negative_slope=0.2)
    )
    self.layer5 = torch.nn.Sequential(
        torch.nn.Conv3d(self.dim*8, 1, kernel_size=4, stride=2, padding=padd),
        torch.nn.Sigmoid()
    )

  def forward(self, x):
    out = x.view(-1, 1, self.dim, self.dim, self.dim)
    # print(out.size()) # torch.Size([100, 1, 64, 64, 64])
    out = self.layer1(out)
    # print(out.size())  # torch.Size([100, 64, 32, 32, 32])
    out = self.layer2(out)
    # print(out.size())  # torch.Size([100, 128, 16, 16, 16])
    out = self.layer3(out)
    # print(out.size())  # torch.Size([100, 256, 8, 8, 8])
    out = self.layer4(out)
    # print(out.size())  # torch.Size([100, 512, 4, 4, 4])
    out = self.layer5(out)
    # print(out.size())  # torch.Size([100, 200, 1, 1, 1])

    return out