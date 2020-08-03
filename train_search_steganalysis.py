import os
import sys
import time
import glob
import numpy as np
import random
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils

import cv2
import scipy.io as sio

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model_search_steganalysis import Network
from architect import Architect


parser = argparse.ArgumentParser("steganalysis")
parser.add_argument('-i', '--DATASET_INDEX', help='Path for loading dataset', type=str, default='')
parser.add_argument('-alg', '--STEGANOGRAPHY', help='embedding_algorithm', type=str, choices=['s-uniward','wow','hill'], default='s-uniward')
parser.add_argument('-rate', '--EMBEDDING_RATE', help='embedding_rate', type=str, choices=['0.1', '0.2', '0.3', '0.4'], default='0.4')

parser.add_argument('--BOSSBASE_COVER_DIR', help='The path to load BOSSBase_COVER_dataset', type=str, required=True) 
parser.add_argument('--BOSSBASE_STEGO_DIR', help='The path to load BOSSBase_STEGO_dataset', type=str, required=True)
parser.add_argument('--BOWS2_COVER_DIR', help='The path to load BOWS2_COVER_dataset', type=str, required=True) 
parser.add_argument('--BOWS2_STEGO_DIR', help='The path to load BOWS2_STEGO_dataset', type=str, required=True) 

parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
#parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=70, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=35, help='begin epoch for train the architecture hyper-parameter')

#parser.add_argument('--tmp_data_dir', type=str, default='/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args = parser.parse_args()

args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

 
CLASSES = 2


class AugData():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    rot = random.randint(0,3)


    data = np.rot90(data, rot, axes=[1, 2]).copy()


    if random.random() < 0.5:
      data = np.flip(data, axis=2).copy()

    new_sample = {'data': data, 'label': label}

    return new_sample

class ToTensor():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32)
    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'label': torch.from_numpy(label).long(),
    }

    return new_sample['data'], new_sample['label']
    
class MyDataset(Dataset):
  def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, transform=None):
    self.index_list = np.load(index_path)
    self.transform = transform

    self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
    self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.pgm'

    self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
    self.bows_stego_path = BOWS_STEGO_DIR + '/{}.pgm'

  def __len__(self):
    return self.index_list.shape[0]

  def __getitem__(self, idx):
    file_index = self.index_list[idx]

    if file_index <= 10000:
      cover_path = self.bossbase_cover_path.format(file_index)
      stego_path = self.bossbase_stego_path.format(file_index)
    else:
      cover_path = self.bows_cover_path.format(file_index - 10000)
      stego_path = self.bows_stego_path.format(file_index - 10000)


    cover_data = cv2.imread(cover_path, -1)
    stego_data = cv2.imread(stego_path, -1)
    #cover_data = sio.loadmat(cover_path)['img']
    #stego_data = sio.loadmat(stego_path)['img']


    data = np.stack([cover_data, stego_data])
    label = np.array([0, 1], dtype='int32')

    sample = {'data': data, 'label': label}

    if self.transform:
      sample = self.transform(sample)

    return sample

def initWeights(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    #logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    #dataset_dir = '/cache/'
    #pre.split_dataset(dataset_dir)
    #sys.exit(1)
  
   # dataset prepare
     
    train_transform = transforms.Compose([
      AugData(),
      ToTensor()
    ])

    valid_transform = transforms.Compose([
      ToTensor()
    ])
    
    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE 
 
    BOSSBASE_COVER_DIR = args.BOSSBASE_COVER_DIR
    BOSSBASE_STEGO_DIR = args.BOSSBASE_STEGO_DIR
    
    BOWS_COVER_DIR = args.BOWS2_COVER_DIR
    BOWS_STEGO_DIR = args.BOWS2_STEGO_DIR

    TRAIN_INDEX_PATH1 = 'index_list{}/bossbase_and_bows_train_index1.npy'.format(DATASET_INDEX)
    TRAIN_INDEX_PATH2 = 'index_list{}/bossbase_and_bows_train_index2.npy'.format(DATASET_INDEX)
    VALID_INDEX_PATH = 'index_list{}/bossbase_valid_index.npy'.format(DATASET_INDEX)

    train_data1 = MyDataset(TRAIN_INDEX_PATH1, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, train_transform)
    train_data2 = MyDataset(TRAIN_INDEX_PATH2, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, train_transform)
    valid_data = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, valid_transform)
 
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
  

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    num_train = len(train_data1)
    num_val = len(train_data2)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)
    
    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.apply(initWeights)
    
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    params = model.parameters()
     
    params_required = []
    for param_item in params:
        if param_item.requires_grad:
            params_required.append(param_item)
            
    optimizer = torch.optim.SGD(
        params_required,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
        
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
               lr=args.arch_learning_rate, betas=(0.5, 0.999), 
               weight_decay=args.arch_weight_decay)
    
    train_queue = torch.utils.data.DataLoader(
        train_data1, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    test_queue = torch.utils.data.DataLoader(
                        valid_data, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        pin_memory=True, 
                        num_workers=args.workers)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    #architect = Architect(model, args)
    lr=args.learning_rate
    for epoch in range(args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            print(optimizer) 
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)
        arch_param = model.module.arch_parameters()
        logging.info(F.softmax(arch_param[0], dim=-1))
        logging.info(F.softmax(arch_param[1], dim=-1))
        logging.info(F.softmax(arch_param[4], dim=-1))
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr,epoch)
        logging.info('Train_acc %f', train_acc)
        
        # validation
        if epoch>= 47:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            #test_acc, test_obj = infer(test_queue, model, criterion)
            logging.info('Valid_acc %f', valid_acc)
            #logging.info('Test_acc %f', test_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, valid_queue, model, optimizer, optimizer_a, criterion, lr,epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()

    params = model.module.parameters()
    
    params_required = []
    for param_item in params:
        if param_item.requires_grad:
            params_required.append(param_item)
        
    for step, (input, target) in enumerate(train_queue):
        model.train()

        shape = list(input.size())
        input = input.reshape(shape[0] * shape[1], *shape[2:])
        target = target.reshape(-1)
        
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        shape = list(input_search.size())
        input_search = input_search.reshape(shape[0] * shape[1], *shape[2:])
        target_search = target_search.reshape(-1)
        
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        
        if epoch >=args.begin:
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()
        #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(params_required, args.grad_clip)
#        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()


        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top2.avg)

    return top1.avg, objs.avg



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        
        shape = list(input.size())
        input = input.reshape(shape[0] * shape[1], *shape[2:])
        target = target.reshape(-1)
  
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 

