import torch
import config as c

import numpy as np
from crop_dataset import create_data_loader
from preprocessing import calculate_coordinates
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from crop_dataset import create_data_loader
from gan.discriminator import Discriminator
from gan.generator import Generator


# custom weights initialization called on netG and netD
from net import evaluate_model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define params for learning
    epochs=4
    lr=0.0002
    beta1=0.5
    over=4
    wtl2 = 0.999
    resume_epoch=0

    # creating networks
    netG = Generator()
    netG.apply(weights_init)

    netD = Discriminator()
    netD.apply(weights_init)

    netD = netD.to(device)
    netG = netG.to(device)

    # creating loss functions
    criterion = nn.BCELoss().to(device)

    REAL_LABEL = 1.
    FAKE_LABEL = 0.

    # load data
    test_loader, val_loader, train_loader, test_batch_size, val_batch_size, train_batch_size = create_data_loader()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    G_losses = []
    D_losses = []

    run_validation_batch_num = 200
    best_val_loss = np.inf

    batch_ind = 0
    for epoch in range(resume_epoch,epochs):
        train_loader_size = len(train_loader)
        for X, y, meta, target in train_loader:
            batch_size = X.shape[0]
            X = X.to(device)
            y = y.to(device)

            # start the discriminator by training with real data---
            netD.zero_grad()
            label = torch.full((batch_size,), REAL_LABEL, device=device)
            output = netD(y).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train the discriminator with fake data---
            fake = netG(X)
            label.fill_(FAKE_LABEL)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            #train the generator now---
            netG.zero_grad()
            label.fill_(REAL_LABEL)
            output = netD(fake).view(-1)
            errG_D = criterion(output, label)

            errG_l2 = torch.zeros((batch_size,), dtype=torch.float32)
            for i in range(batch_size):
                crop_size = meta[i]['crop_size']
                crop_center = meta[i]['crop_center']
                st_x, en_x, st_y, en_y = calculate_coordinates(crop_size, crop_center)
                sample_fake = fake[i, 0, st_y:en_y, st_x:en_x]
                sample_target = target[i].to(device)
                errG_l2[i] = ((sample_fake - sample_target)**2).mean()

            errG_l2 = errG_l2.mean().to(device)
            errG = errG_D * 0.001 + errG_l2
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('Epochs: [%d / %d] Baches:[%d / %d] Loss_crop_G: %.4f'
                  % (epoch, epochs, batch_ind, train_loader_size, errG_l2.data))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (batch_ind + 1) % run_validation_batch_num == 0:
                val_loss = evaluate_model(netG, val_loader, device)
                val_loss_nom = val_loss.item()
                print(f'Validation set, Loss: {val_loss_nom:.4f}')
                if val_loss_nom <= best_val_loss:
                    best_val_loss = val_loss_nom
                    torch.save(netG, c.BEST_MODEL_FILE)
            batch_ind += 1

            # TODO validation set
        # TODO at the end calc cost of all

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()