from __future__ import print_function
import argparse
import os
import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import dcgan
import numpy as np
np.random.seed(43)
from multiprocessing import Pool
import h5py


#Change workdir to where you want the files output
work_dir = os.path.expandvars('$PBS_O_WORKDIR/berea_test')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='3D')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
work_dir = os.path.expandvars('$PBS_O_WORKDIR/filters/')
opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1

def save_hdf5(tensor, weights, filename):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    tensor = tensor.cpu()
    ndarr = tensor.float().numpy()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=ndarr, compression="gzip")
	f.create_dataset('weights', data=weights.cpu().float().numpy(), compression='gzip')

def load_img(filepath):
    img = None
    with h5py.File(filepath, "r") as f:
        img = f['data'][()]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    torch_img = Tensor(img)
    torch_img = torch_img.div(255).sub(0.5).div(0.5)
    return torch_img


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netD = dcgan.DCGAN3D_D(opt.imageSize, nz, nc, ngf, ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netG = dcgan.DCGAN3D_G(opt.imageSize, nz, nc, ngf, ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

def counter_init(mod='layer_'):
    counter = 0
    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Variable. output.data is the Tensor we are interested
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))
        print('input[0]: ', type(input[0]))
        print('output: ', type(output))
        print('')
        print('output size:', output.data.size())
        
        try:
            print(self.weight.__class__.__name__, self.weight.data.__class__.__name__, self.parameters().__class__.__name__)
            print(self.weight)
            save_hdf5(output.data, self.weight.data, work_dir+mod+self.__class__.__name__+'_'+str(output.data.size())+'.hdf5')
        except:
            pass
    return printnorm

printnorm = counter_init()
for module in netD.modules():
    print(module)
    module.register_forward_hook(printnorm)

img = load_img("berea_1.hdf5")
img = img.cuda()
input = Variable(img)
netD = netD.cuda()
netD.forward(input)
print('forwarded')


printnorm = counter_init(mod='gen_')
for module in netG.modules():
    print(module)
    module.register_forward_hook(printnorm)
fixed_noise = Variable(torch.FloatTensor(1, nz, 9, 9, 9).normal_(0, 1).cuda())
netG = netG.cuda()
netG.forward(fixed_noise)
print('forwarded')