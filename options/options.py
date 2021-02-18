import argparse
from os import path, makedirs
import torch

def printInfo(inputs):
    print("[ XTY's StyleGAN ] %s" % (inputs))

def printParameters(args_dict):
    printInfo("+++++++++++  Parameters ++++++++++++")
    for key in sorted(args_dict.keys()):
        printInfo("{:<20} : {}".format(key, args_dict[key]))
    printInfo("++++++++++++++++++++++++++++++++++++")

class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch StyleGAN Traning")
        parser.add_argument("--outf", type=str, default="train_result", help="folder to output models")
        parser.add_argument("--save-checkpoints", type=int, default=50, help="print frequency (default: 50)")
        parser.add_argument("--exp-id", type=str, default="1", help="record the id of experiment")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--epoches", type=int, default=250)
        parser.add_argument("--batch-size", type=int, default=100)
        parser.add_argument("--dataset", type=int, default=0, help="choose the dataset(0 : lfw, 1 : comic figures)")
        parser.add_argument("--cuda", default=0, type=int, help="choose which GPU to use in training")
        parser.add_argument("--resume", default=None, type=str, help='choose the pretrained model')
        self.opts = parser.parse_args()
        

    def parse(self):
        self.device = 'cuda:{}'.format(self.opts.cuda) if torch.cuda.is_available() else "cpu"
        self.opts.model_dir = path.join(path.dirname(path.dirname(__file__)), self.opts.outf, "models", self.opts.exp_id)
        self.opts.log_dir = path.join(path.dirname(path.dirname(__file__)), self.opts.outf, "logs", self.opts.exp_id)
        self.opts.img_dir = path.join(path.dirname(path.dirname(__file__)), self.opts.outf, "images", self.opts.exp_id)
        if not path.exists(self.opts.model_dir):
            makedirs(self.opts.model_dir)
        if not path.exists(self.opts.log_dir):
            makedirs(self.opts.log_dir)
        if not path.exists(self.opts.img_dir):
            makedirs(self.opts.img_dir)
        printParameters((self.opts.__dict__))
        return self.opts
        
        
