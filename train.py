import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader

from data.shapenet import ShapeNetVox
from model import _G, _D
from util.lr_sh import  MultiStepLR


if torch.cuda.is_available():
    print("using cuda")
    
if __name__ == "__main__":
    with open('config.yaml') as fp:
        config = yaml.load(fp, Loader=SafeLoader)
        
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["shape_dir"], exist_ok=True)
    os.makedirs(config["out_model_dir"], exist_ok=True)
    os.makedirs(config["out_d_loss_dir"], exist_ok=True)
    os.makedirs(config["out_g_loss_dir"], exist_ok=True)
    os.makedirs(config["out_d_fake_loss_dir"], exist_ok=True)
    os.makedirs(config["out_d_real_loss_dir"], exist_ok=True)
    os.makedirs(config["out_d_acc_dir"], exist_ok=True)
    os.makedirs(config["out_d_fake_acc_dir"], exist_ok=True)
    os.makedirs(config["out_d_real_acc_dir"], exist_ok=True)
    
    # To clear log file starting to training
    os.system("rm {}/logs.txt".format(config["output_dir"]))
    
    trainset = ShapeNetVox('all')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(trainset,  batch_size=config["batch_size"], shuffle=True)
    # val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=True)

    G = _G(config["dim"], config["latent_len"])
    D = _D(config["dim"], config["latent_len"])

    G.to(device)
    D.to(device)

    D_optim = optim.Adam(D.parameters(), lr=config["d_lr"], betas=(config["beta"], config["beta"]))
    G_optim = optim.Adam(G.parameters(), lr=config["g_lr"], betas=(config["beta"], config["beta"]))
    
    if config["lrsh"]:
        D_scheduler = MultiStepLR(D_optim, milestones=[500, 1000])

    loss_fn = nn.BCELoss()

    for epoch in range(config["epoch"]):
        print(f"Epoch: {epoch}")
        d_losses = []
        d_fake_losses = []
        d_real_losses = []
        g_losses = []
        d_acc = []
        d_fake_accs = []
        d_real_accs = []
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_gpu = batch["voxel"].to(device)

            Z = torch.Tensor(batch_gpu.shape[0], config["latent_len"]).normal_(0, 0.33).to(device)
            real_labels = torch.ones((batch_gpu.shape[0], )).to(device)
            fake_labels = torch.zeros((batch_gpu.shape[0], )).to(device)
            
            if config["soft_label"]:
                real_labels = torch.Tensor(batch_gpu.shape[0]).uniform_(0.7, 1.2).to(device)
                fake_labels = torch.Tensor(batch_gpu.shape[0]).uniform_(0, 0.3).to(device)

            # ============= Train the discriminator =============#
            d_real = D(batch_gpu)
    #         print(d_real.shape, real_labels.shape)
            d_real = torch.squeeze(d_real)
            d_real_loss = loss_fn(d_real, real_labels)


            fake = G(Z)
            d_fake = D(fake)
            d_fake = torch.squeeze(d_fake)
            d_fake_loss = loss_fn(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_losses.append(d_loss.item())

            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))
            d_acc.append(d_total_acu.item())

            if d_total_acu <= config["d_acc_thres"]:
                D.zero_grad()
                d_loss.backward()
                D_optim.step()

            # =============== Train the generator ===============#

            Z = torch.Tensor(batch_gpu.shape[0], config["latent_len"]).normal_(0, 0.33).to(device)

            fake = G(Z)
            d_fake = D(fake)
            d_fake = torch.squeeze(d_fake)
            g_loss = loss_fn(d_fake, real_labels)
            g_losses.append(g_loss.item())

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_optim.step()
            
            d_fake_losses.append(d_fake_loss.item())
            d_real_losses.append(d_real_loss.item())
            d_fake_accs.append(float(torch.mean(d_real_acu).detach().cpu()))
            d_real_accs.append(float(torch.mean(d_real_acu).detach().cpu()))
            
            if i%4 == 0:
                with open(os.path.join(config["output_dir"], "logs.txt"), "a") as fp:
                    fp.write(f"Epoch: {epoch}, iteration: {i}, d_real_acu: {torch.mean(d_real_acu)}, d_fake_acu: {torch.mean(d_fake_acu)}, d_total_acu: {d_total_acu}\n")
                    fp.write(f"d_real_loss: {d_real_loss}, d_fake_loss: {d_fake_loss}, g_loss: {g_loss}\n\n")
                    with open(os.path.join(config["shape_dir"], f'e_{epoch}_i_{i}.npy'), 'wb') as fp:
                        np.save(fp, (fake[0].detach().cpu().numpy().squeeze() > 0.5).astype(int))
                # visualize_occupancy((fake[0].detach().cpu().numpy().squeeze() > 0.5).astype(int), flip_axes=True)
            
        
        torch.save(G.state_dict(), os.path.join(config["out_model_dir"], "G_{}.pth".format(epoch)))
        torch.save(D.state_dict(), os.path.join(config["out_model_dir"], "D_{}.pth".format(epoch)))
        
        d_losses = np.array(d_losses)
        g_losses = np.array(g_losses)
        d_acc = np.array(d_acc)
        d_fake_losses = np.array(d_fake_losses)
        d_real_losses = np.array(d_real_losses)
        d_real_accs = np.array(d_real_accs)
        d_fake_accs = np.array(d_fake_accs)
        
        with open(os.path.join(config["out_d_loss_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, d_losses)
        
        with open(os.path.join(config["out_g_loss_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, g_losses)
            
        with open(os.path.join(config["out_d_fake_loss_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, d_fake_losses)
        
        with open(os.path.join(config["out_d_real_loss_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, d_real_losses)
        
        with open(os.path.join(config["out_d_acc_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, d_acc)
        
        with open(os.path.join(config["out_d_fake_acc_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, d_fake_accs)
        
        with open(os.path.join(config["out_d_real_acc_dir"], f'{epoch}.npy'), 'wb') as fp:
            np.save(fp, d_real_accs)
        
        epoch_dloss = np.mean(d_losses)
        epoch_gloss = np.mean(g_losses)
        epoch_dacc = np.mean(d_acc)
        with open(os.path.join(config["output_dir"], "logs.txt"), "a") as fp:
            fp.write(f"Epoch: {epoch}, d_loss: {epoch_dloss}, g_loss: {epoch_gloss}, dacc: {epoch_dacc}\n")
            fp.write("==================================================================================\n")

        if config['lrsh']:
            try:
                D_scheduler.step()
            except Exception as e:
                print("fail lr scheduling", e)

