from torch.autograd import Variable, grad
import torch.autograd as autograd
import torch
import numpy as np

def R1Penalty(real_img, f):
    Variable()
    reals = Variable(real_img, requires_grad=True).to(real_img.device)
    real_score = f(reals).sum()
    real_grads = grad(real_score, reals)[0].view(reals.size(0), -1)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty


def R2Penalty(fake_img, f):
    fakes = Variable(fake_img, requires_grad=True).to(real_img.device)
    fake_score = f(fakes).sum()
    fake_grads = grad(fake_score, fakes)[0].view(fakes,size(0), -1)
    r2_penalty = torch.sum(torch.mul(fake_grads, fake_grads))
    return r2_penalty
