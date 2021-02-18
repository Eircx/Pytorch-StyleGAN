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
from network_styleGAN import StyleGenerator, StyleDiscriminator

if __name__ == "__main__":
    opts = TrainOptions().parse()
    dataloader = None

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
    
    #Multi-GPU support
    if torch.cuda.is_available() > 1:
        printInfo("Multiple GPU: %d GPUs are available!" % torch.cuda.device_count())
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    G.to(opts.device)
    D.to(opts.device)

    G_optimizer = optim.Adam(G.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    D.optimizer = optim.Adam(D.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    G_scheduler = optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.99)
    D_scheduler = optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.99)

    #Training
    fix_z = torch.randn([opts.batch_size, 128]).to()




        

