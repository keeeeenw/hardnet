#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-
"""
This is HardNet local patch descriptor. The training code is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this code, please cite 
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin 
"""

from __future__ import division, print_function

# uncomment to support plotting in headless mode
# this will not work with Jupyter
# import matplotlib
# matplotlib.use('Agg')

import sys
from copy import deepcopy
import math
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import PIL
from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from SOSLoss import loss_SOSNet
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm, cv2_scale, np_reshape
from Utils import str2bool
import utils.w1bs as w1bs
import torch.nn as nn
import torch.nn.functional as F

# ResNet improvements
from ResNetMono import resnet18, resnet34, resnet50, resnet101, reshardnet, reshardnetsmall, reshardnetsmall2, reshardnetstiny
# DenseNet improvements
from DenseNetMono import densenet121
# MoileNet improvements
from MobileNetMono import mobilenet_v2

from torchsummary import summary

class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum())/input.size(0)

class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, global_args, train=True, transform=None, batch_size = None,load_random_triplets = False, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.global_args = global_args
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = global_args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.global_args, self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(global_args, labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= global_args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if self.global_args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
        print("Initializing weights")
        self.features.apply(weights_init)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class ResHardNet(nn.Module):
    """ResHardNet model definition
    """
    def __init__(self, pretrained=False, model="reshardnet", dropout=0.0, initialization="hardnet"):
        super(ResHardNet, self).__init__()
        # by default resnet outputs 1000 classes

        if (model == "reshardnet"):
            print("Creating ResNet 18 Model")
            self.features = resnet18(pretrained=pretrained, progress=True, num_classes=128, dropout=dropout)
        elif (model == "reshardnet34"):
            print("Creating ResNet 34 Model")
            self.features = resnet34(pretrained=pretrained, progress=True, num_classes=128, dropout=dropout)
        elif (model == "reshardnet50"):
            print("Creating ResNet 50 Model")
            self.features = resnet50(pretrained=pretrained, progress=True, num_classes=128, dropout=dropout)
        elif (model == "reshardnet101"):
            print("Creating ResNet 101 Model")
            self.features = resnet101(pretrained=pretrained, progress=True, num_classes=128, dropout=dropout)
        elif (model == "reshardnetdefault"):
            print("Creating ResNet Hard")
            self.features = reshardnet(dropout=dropout)
        elif (model == "reshardnetdefaultsmall"):
            print("Creating ResNet Hard Small")
            self.features = reshardnetsmall(dropout=dropout, initialization=initialization)
        elif (model == "reshardnetdefaultsmall2"):
            print("Creating ResNet Hard Small")
            self.features = reshardnetsmall2(dropout=dropout)
        elif (model == "reshardnetdefaulttiny"):
            print("Creating ResNet Hard Tiny")
            self.features = reshardnetstiny(dropout=dropout, initialization=initialization)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class DenseHardNet(nn.Module):
    """DenseHardNet model definition
    """
    def __init__(self, pretrained=False, model="densenet121"):
        super(DenseHardNet, self).__init__()
        # by default desnet outputs 1000 classes
        print("Creating dense 121 Model")
        self.features = densenet121(pretrained=pretrained, progress=True, num_classes=128)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class MobileV2HardNet(nn.Module):
    """DenseHardNet model definition
    """
    def __init__(self, pretrained=False, model="mobilenet_v2"):
        super(MobileV2HardNet, self).__init__()
        # by default desnet outputs 1000 classes
        print("Creating MobileNet V2 Model")
        self.features = mobilenet_v2(pretrained=pretrained, progress=True, num_classes=128)
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

class TrainHardNet(object):
    def __init__(self, args=None):
        if args is None:
            # Training settings
            parser = argparse.ArgumentParser(description='PyTorch HardNet')
            
            # Add dummy arguments
            # https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter
            parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

            # Model options
            parser.add_argument('--w1bsroot', type=str,
                                default='data/sets/wxbs-descriptors-benchmark/code/',
                                help='path to dataset')
            parser.add_argument('--dataroot', type=str,
                                default='data/sets/',
                                help='path to dataset')
            parser.add_argument('--enable-logging',type=str2bool, default=True,
                                help='output to tensorlogger')
            parser.add_argument('--log-dir', default='data/logs/',
                                help='folder to output log')
            parser.add_argument('--model-dir', default='data/models/',
                                help='folder to output model checkpoints')
            parser.add_argument('--experiment-name', default= 'liberty_train/',
                                help='experiment path')
            parser.add_argument('--training-set', default= 'liberty',
                                help='Other options: notredame, yosemite')
            parser.add_argument('--loss', default= 'triplet_margin',
                                help='Other options: softmax, contrastive, qht')
            parser.add_argument('--batch-reduce', default= 'min',
                                help='Other options: average, random, random_global, L2Net')
            parser.add_argument('--num-workers', default= 0, type=int,
                                help='Number of workers to be created')
            parser.add_argument('--pin-memory',type=bool, default= True,
                                help='')
            parser.add_argument('--decor',type=str2bool, default = False,
                                help='L2Net decorrelation penalty')
            parser.add_argument('--anchorave', type=str2bool, default=False,
                                help='anchorave')
            parser.add_argument('--imageSize', type=int, default=32,
                                help='the height / width of the input image to network')
            parser.add_argument('--mean-image', type=float, default=0.443728476019,
                                help='mean of train dataset for normalization')
            parser.add_argument('--std-image', type=float, default=0.20197947209,
                                help='std of train dataset for normalization')
            parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                help='path to latest checkpoint (default: none)')
            parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                                help='manual epoch number (useful on restarts)')
            parser.add_argument('--epochs', type=int, default=10, metavar='E',
                                help='number of epochs to train (default: 10)')
            parser.add_argument('--anchorswap', type=str2bool, default=True,
                                help='turns on anchor swap')
            parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                                help='input batch size for training (default: 1024)')
            parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                                help='input batch size for testing (default: 1024)')
            parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                                help='how many triplets will generate from the dataset')
            parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                                help='the margin value for the triplet loss function (default: 1.0')
            parser.add_argument('--gor',type=str2bool, default=False,
                                help='use gor')
            parser.add_argument('--freq', type=float, default=10.0,
                                help='frequency for cyclic learning rate')
            parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                                help='gor parameter')
            parser.add_argument('--lr', type=float, default=10.0, metavar='LR',
                                help='learning rate (default: 10.0. Yes, ten is not typo)')
            parser.add_argument('--fliprot', type=str2bool, default=True,
                                help='turns on flip and 90deg rotation augmentation')
            parser.add_argument('--augmentation', type=str2bool, default=False,
                                help='turns on shift and small scale rotation augmentation')
            parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                                help='learning rate decay ratio (default: 1e-6')
            parser.add_argument('--wd', default=1e-4, type=float,
                                metavar='W', help='weight decay (default: 1e-4)')
            parser.add_argument('--optimizer', default='sgd', type=str,
                                metavar='OPT', help='The optimizer to use (default: SGD)')
            # Device options
            parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='enables CUDA training')
            parser.add_argument('--gpu-id', default='0', type=str,
                                help='id(s) for CUDA_VISIBLE_DEVICES')
            parser.add_argument('--seed', type=int, default=0, metavar='S',
                                help='random seed (default: 0)')
            parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                                help='how many batches to wait before logging training status')
            parser.add_argument('--model-variant', default='hardnet', type=str,
                                metavar='mv', help='The model to use (default: hardnet)')
            parser.add_argument('--pre-trained', default=False, type=str2bool,
                                metavar='pt', help='Use pretrained weights')
            parser.add_argument('--dropout', default=0.0, type=float,
                                metavar='dp', help='Dropout for various models')
            parser.add_argument('--change-lr', default=True, type=str2bool,
                                metavar='clr', help='Should change learning rate')
            parser.add_argument('--initialization', default='hardnet', type=str,
                                metavar='init', help='Initilaization method for selective models')

            self.args = parser.parse_args()
        else:
            self.args = args
        
        if self.args.model_variant.startswith("reshardnet"):
            self.model = ResHardNet(self.args.pre_trained, self.args.model_variant, self.args.dropout, self.args.initialization)
        elif self.args.model_variant.startswith("densenet"):
            self.model = DenseHardNet(self.args.pre_trained, self.args.model_variant)
        elif self.args.model_variant.startswith("mobilenet_v2"):
            self.model = MobileV2HardNet(self.args.pre_trained, self.args.model_variant)
        else:
            self.model = HardNet()

        self.suffix = '{}_{}_{}'.format(self.args.experiment_name, self.args.training_set, self.args.batch_reduce)

        self.triplet_flag = (self.args.batch_reduce == 'random_global') or self.args.gor

        self.dataset_names = ['liberty', 'notredame', 'yosemite']

        self.change_lr = self.args.change_lr

        self.test_on_w1bs = False
        # check if path to w1bs dataset testing module exists
        if os.path.isdir(self.args.w1bsroot):
            sys.path.insert(0, self.args.w1bsroot)
            import utils.w1bs as w1bs
            self.test_on_w1bs = True

        # set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
        # order to prevent any memory allocation on unused GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_id

        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        print (("NOT " if not self.args.cuda else "") + "Using cuda")

        if self.args.cuda:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True

        # create loggin directory
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)

        self.print_summary = True

    def create_loaders(self, load_random_triplets = False):

        test_dataset_names = copy.copy(self.dataset_names)
        test_dataset_names.remove(self.args.training_set)

        kwargs = {'num_workers': self.args.num_workers, 'pin_memory': self.args.pin_memory} if self.args.cuda else {}
        
        print("Creating loaders with the following arguments")
        print(kwargs)

        np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
        transform_test = transforms.Compose([
                transforms.Lambda(np_reshape64),
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.ToTensor()])
        transform_train = transforms.Compose([
                transforms.Lambda(np_reshape64),
                transforms.ToPILImage(),
                transforms.RandomRotation(5,PIL.Image.BILINEAR),
                transforms.RandomResizedCrop(32, scale = (0.9,1.0),ratio = (0.9,1.1)),
                transforms.Resize(32),
                transforms.ToTensor()])
        transform = transforms.Compose([
                transforms.Lambda(cv2_scale),
                transforms.Lambda(np_reshape),
                transforms.ToTensor(),
                transforms.Normalize((self.args.mean_image,), (self.args.std_image,))])
        if not self.args.augmentation:
            transform_train = transform
            transform_test = transform
        train_loader = torch.utils.data.DataLoader(
                TripletPhotoTour(self.args,
                                train=True,
                                load_random_triplets = load_random_triplets,
                                batch_size=self.args.batch_size,
                                root=self.args.dataroot,
                                name=self.args.training_set,
                                download=True,
                                transform=transform_train),
                                batch_size=self.args.batch_size,
                                shuffle=False, **kwargs)

        test_loaders = [{'name': name,
                        'dataloader': torch.utils.data.DataLoader(
                TripletPhotoTour(self.args,
                        train=False,
                        batch_size=self.args.test_batch_size,
                        root=self.args.dataroot,
                        name=name,
                        download=True,
                        transform=transform_test),
                            batch_size=self.args.test_batch_size,
                            shuffle=False, **kwargs)}
                        for name in test_dataset_names]

        return train_loader, test_loaders

    def train(self, train_loader, model, optimizer, epoch, logger, load_triplets  = False):
        print("Training model")
        # switch to train mode
        model.train()
        pbar = tqdm(enumerate(train_loader))
        for batch_idx, data in pbar:
            if load_triplets:
                data_a, data_p, data_n = data
            else:
                data_a, data_p = data

            if self.args.cuda:
                data_a, data_p  = data_a.cuda(), data_p.cuda()
                data_a, data_p = Variable(data_a), Variable(data_p)
            out_a = model(data_a)
            out_p = model(data_p)
            if load_triplets:
                data_n  = data_n.cuda()
                data_n = Variable(data_n)
                out_n = model(data_n)

            if self.args.loss == 'qht':
                loss = loss_SOSNet(out_a, out_p,
                                   batch_reduce=self.args.batch_reduce,
                                   no_cuda=self.args.no_cuda)
            else:
                if self.args.batch_reduce == 'L2Net':
                    loss = loss_L2Net(out_a, out_p, anchor_swap = self.args.anchorswap,
                            margin = self.args.margin, loss_type = self.args.loss)
                elif self.args.batch_reduce == 'random_global':
                    loss = loss_random_sampling(out_a, out_p, out_n,
                        margin=self.args.margin,
                        anchor_swap=self.args.anchorswap,
                        loss_type = self.args.loss)
                else:
                    loss = loss_HardNet(out_a, out_p,
                                    margin=self.args.margin,
                                    anchor_swap=self.args.anchorswap,
                                    anchor_ave=self.args.anchorave,
                                    batch_reduce = self.args.batch_reduce,
                                    loss_type = self.args.loss,
                                    no_cuda = self.args.no_cuda)

            if self.args.decor:
                loss += CorrelationPenaltyLoss()(out_a)
                
            if self.args.gor:
                loss += self.args.alpha*global_orthogonal_regularization(out_a, out_n)
            
            if self.print_summary:
                with torch.no_grad():
                    # We can only do it here because the input are only switched
                    # to cuda types above.
                    summary(model, input_size=(1, self.args.imageSize, self.args.imageSize))
                self.print_summary = False
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.change_lr:
                self.adjust_learning_rate(optimizer)
            if batch_idx % self.args.log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                        loss.item()))

        if (self.args.enable_logging):
            logger.log_value('loss', loss.item()).step()

        try:
            os.stat('{}{}'.format(self.args.model_dir,self.suffix))
        except:
            os.makedirs('{}{}'.format(self.args.model_dir,self.suffix))

        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                '{}{}/checkpoint_{}.pth'.format(self.args.model_dir,self.suffix,epoch))

    def test(self, test_loader, model, epoch, logger, logger_test_name):
        print("Testing model")
        # switch to evaluate mode
        model.eval()

        labels, distances = [], []

        pbar = tqdm(enumerate(test_loader))
        for batch_idx, (data_a, data_p, label) in pbar:

            if self.args.cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()

            with torch.no_grad():
                data_a, data_p, label = Variable(data_a), \
                                        Variable(data_p), Variable(label)
                out_a = model(data_a)
                out_p = model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1,1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)

            if batch_idx % self.args.log_interval == 0:
                pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                        100. * batch_idx / len(test_loader)))

        num_tests = test_loader.dataset.matches.size(0)
        labels = np.vstack(labels).reshape(num_tests)
        distances = np.vstack(distances).reshape(num_tests)

        fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
        print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

        if (self.args.enable_logging):
            logger.log_value(logger_test_name+' fpr95', fpr95)
        return

    def adjust_learning_rate(self, optimizer):
        """Updates the learning rate given the learning rate decay.
        The routine has been implemented according to the original Lua SGD optimizer
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0.
            else:
                group['step'] += 1.
            group['lr'] = self.args.lr * (
            1.0 - float(group['step']) * float(self.args.batch_size) / (self.args.n_triplets * float(self.args.epochs)))
        return

    def create_optimizer(self, model, new_lr):
        # setup optimizer
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=new_lr,
                                momentum=0.9, dampening=0.9,
                                weight_decay=self.args.wd)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=new_lr,
                                weight_decay=self.args.wd)
        else:
            raise Exception('Not supported optimizer: {0}'.format(self.args.optimizer))
        return optimizer

    def execute(self, train_loader, test_loaders, model, logger, file_logger):
        # print the experiment configuration
        print('\nparsed options:\n{}\n'.format(vars(self.args)))

        #if (self.args.enable_logging):
        #    file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(self.args)))

        if self.args.cuda:
            model.cuda()

        optimizer1 = self.create_optimizer(model.features, self.args.lr)

        # optionally resume from a checkpoint
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print('=> loading checkpoint {}'.format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.args.start_epoch = checkpoint['epoch']
                checkpoint = torch.load(self.args.resume)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('=> no checkpoint found at {}'.format(self.args.resume))
                
        
        start = self.args.start_epoch
        end = start + self.args.epochs
        for epoch in range(start, end):

            # iterate over test loaders and test results
            self.train(train_loader, model, optimizer1, epoch, logger, self.triplet_flag)
            for test_loader in test_loaders:
                self.test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])

            if self.test_on_w1bs:
                print("Saving test_on_w1bs results")
                patch_images = w1bs.get_list_of_patch_images(
                    DATASET_DIR=self.args.w1bsroot)
                desc_name = 'curr_desc'# + str(random.randint(0,100))
                
                self.descs_dir = self.log_dir + '/temp_descs/' #self.args.w1bsroot.replace('/code', "/data/out_descriptors")
                OUT_DIR = self.descs_dir.replace('/temp_descs/', "/out_graphs/")

                for img_fname in patch_images:
                    w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = self.args.cuda,
                                                mean_img=self.args.mean_image,
                                                std_img=self.args.std_image, out_dir = self.descs_dir)


                force_rewrite_list = [desc_name]
                w1bs.match_descriptors_and_save_results(DESC_DIR=self.descs_dir, do_rewrite=True,
                                                        dist_dict={},
                                                        force_rewrite_list=force_rewrite_list)
                print("descs_dir", self.descs_dir)
                print("OUT_DIR", OUT_DIR)
                print("Number of patch_images", len(patch_images))
                if(self.args.enable_logging):
                    w1bs.draw_and_save_plots_with_loggers(DESC_DIR=self.descs_dir, OUT_DIR=OUT_DIR,
                                            methods=["SNN_ratio"],
                                            descs_to_draw=[desc_name],
                                            logger=file_logger,
                                            tensor_logger = logger)
                else:
                    w1bs.draw_and_save_plots(DESC_DIR=self.descs_dir, OUT_DIR=OUT_DIR,
                                            methods=["SNN_ratio"],
                                            descs_to_draw=[desc_name])
            #randomize train loader batches
            train_loader, test_loaders2 = self.create_loaders(load_random_triplets=self.triplet_flag)

    def run(self):
        # set random seeds
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        self.log_dir = self.args.log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_dir = os.path.join(self.args.log_dir, self.suffix)
        self.descs_dir = os.path.join(self.log_dir, 'temp_descs')
        if self.test_on_w1bs:
            if not os.path.isdir(self.descs_dir):
                os.makedirs(self.descs_dir)
        logger, file_logger = None, None
        print("Creating Model")
        if(self.args.enable_logging):
            from Loggers import Logger, FileLogger
            logger = Logger(self.log_dir)
            #file_logger = FileLogger(./log/+self.suffix)
        print("Creating Loaders")
        train_loader, test_loaders = self.create_loaders(load_random_triplets = self.triplet_flag)
        
        print("Staring execution")
        self.execute(train_loader, test_loaders, self.model, logger, file_logger)

if __name__ == '__main__':
    runner = TrainHardNet()
    runner.run()
