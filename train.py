import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from tensorboardX import SummaryWriter
from os import path, makedirs
from options.options import TrainOptions, printInfo
from glob import glob
from network_styleGAN import StyleDiscriminator, StyleGenerator
from tqdm import tqdm
from dataset import Comic_Dataset, HumanFace_Dataset
from torchvision.utils import save_image
from math import sqrt, ceil
from loss import R1Penalty, R2Penalty
import numpy as np

if __name__ == "__main__":
    opts = TrainOptions().parse()
    if opts.dataset == 0:
        dataset = Comic_Dataset()
    else:
        dataset = HumanFace_Dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)

    start_epoch = 0
    D = StyleDiscriminator()
    G = StyleGenerator()

    if opts.resume:
        resume_file = path.join(opts.model_dir, opts.resume)
        if path.exists(resume_file):
            state = torch.load(opts.resume)
            G.load_state_dict(state['G'])
            D.load_state_dict(state['D'])
            start_epoch = state['epoches']
            printInfo("Succcessfully load the pre-trained models!")
        else:
            printInfo("Cannot load the pre-trained model! Traing from the beginning!")
    else:
        printInfo("Train from the beginning!")
    
    writer = SummaryWriter(log_dir=opts.log_dir)
    writer.add_graph(G, torch.rand(1, 128))
    writer.add_graph(D, torch.randn(1, 3, 256, 256))
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        printInfo("Multiple GPU: %d GPUs are available!" % torch.cuda.device_count())
        G = nn.DataParallel(G, device_ids=[0,1,2,3])
        D = nn.DataParallel(D, device_ids=[0,1,2,3])
    G.to(opts.device)
    D.to(opts.device)

    G_optimizer = optim.Adam(G.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    # G_scheduler = optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.9)
    # D_scheduler = optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.9)

    #Training
    TEST_SIZE = 16
    fix_z = torch.randn([TEST_SIZE, 128]).to(opts.device)
    softplus = nn.Softplus()
    Loss_D_list = [0.0]
    Loss_G_list = [0.0]

    for epoch in range(start_epoch, opts.epoches):
        printInfo("Epoch: %d  @lr: %f" % (epoch, G_optimizer.param_groups[0]['lr']))
        bar = tqdm(data_loader)
        loss_D_list = []
        loss_G_list = []
        for i, real_img in enumerate(bar):
            D.zero_grad()
            real_img = real_img.to(opts.device)
            real_score = D(real_img)
            fake_img = G(torch.randn(real_img.size(0), 128).to(opts.device))
            fake_score = D(fake_img.detach())
            d_loss = softplus(fake_score).mean() + softplus(-real_score).mean()

            if opts.r1_gamma != 0.0:
                r1_penalty = R1Penalty(real_img.detach(), D)
                d_loss += r1_penalty * (opts.r1_gamma * 0.5)

            if opts.r2_gamma != 0.0:
                r2_penalty = R2Penalty(fake_img.detach(), D)
                d_loss += r2_penalty * (opts.r2_gamma * 0.5)

            loss_D_list.append(d_loss.item())

            d_loss.backward()
            D_optimizer.step()

            if i % opts.critic_iter == 0:
                G.zero_grad()
                fake_score = D(fake_img)
                g_loss = softplus(-fake_score).mean()
                loss_G_list.append(g_loss.item())

                g_loss.backward()
                G_optimizer.step()
            
            bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(epoch, i+1, len(data_loader), loss_G_list[-1], loss_D_list[-1]))
        
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))
        writer.add_scalars("Loss in training", {"Generator": np.mean(loss_G_list), "Discriminator": np.mean(loss_D_list)}, epoch + 1)

        with torch.no_grad():
            test_img = G(fix_z)
            save_image(test_img, path.join(opts.img_dir, 'test%d_img.png' % (epoch+1)),normalize=True, nrow=ceil(sqrt(TEST_SIZE)))
            test_score = D(test_img)
            test_dloss = softplus(test_score).mean() + softplus(-test_score).mean()
            test_gloss = softplus(-test_score).mean()
            writer.add_scalars("Loss in testing", {"Generator": test_gloss.item(), "Discriminator": test_dloss.item()}, epoch + 1)
        
        if epoch == 0:
            save_image(real_img, path.join(opts.img_dir, 'real_img.png'), nrow=ceil(sqrt(real_img.size(0))), normalize=True)

        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
            "start_epoch": epoch + 1,
            'style': opts.dataset
        }

        # G_scheduler.step()
        # D_scheduler.step()

        torch.save(state, path.join(opts.model_dir, 'latest.pth'))
        if (epoch + 1) % opts.save_checkpoints == 0:
            torch.save(state, path.join(opts.model_dir, 'model%d.pth' % (epoch + 1)))

    writer.close()
    printInfo("Training complete!")








        

