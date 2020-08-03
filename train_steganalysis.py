import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
#import pandas as pd
import cv2
import random
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch.autograd import Variable
from model import NetworkSteganalysis as Network


parser = argparse.ArgumentParser("training steganalysis")
parser.add_argument('-i', '--DATASET_INDEX', help='Path for loading dataset', type=str, default='')
parser.add_argument('-alg', '--STEGANOGRAPHY', help='embedding_algorithm', type=str, choices=['s-uniward','wow','hill'], default='s-uniward')
parser.add_argument('-rate', '--EMBEDDING_RATE', help='embedding_rate', type=str, choices=['0.1', '0.2', '0.3', '0.4'], default='0.4')

parser.add_argument('--BOSSBASE_COVER_DIR', help='The path to load BOSSBase_COVER_dataset', type=str, required=True) 
parser.add_argument('--BOSSBASE_STEGO_DIR', help='The path to load BOSSBase_STEGO_dataset', type=str, required=True)
parser.add_argument('--BOWS2_COVER_DIR', help='The path to load BOWS2_COVER_dataset', type=str, required=True) 
parser.add_argument('--BOWS2_STEGO_DIR', help='The path to load BOWS2_STEGO_dataset', type=str, required=True) 

parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay') 
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PC_DARTS_steganalysis', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
#parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')


args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 2

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



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
    DECAY_EPOCH = [80, 140, 180]
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()   
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------') 
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
    model.apply(initWeights)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    params = model.parameters()
    
    params_required = []
    for param_item in params:
        if param_item.requires_grad:
            params_required.append(param_item)

    optimizer = torch.optim.SGD(
        params_required,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
        
    train_transform = transforms.Compose([AugData(),ToTensor(),])
    valid_transform = transforms.Compose([ToTensor(),])
    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE

    BOSSBASE_COVER_DIR = args.BOSSBASE_COVER_DIR
    BOSSBASE_STEGO_DIR = args.BOSSBASE_STEGO_DIR
    
    BOWS_COVER_DIR = args.BOWS2_COVER_DIR
    BOWS_STEGO_DIR = args.BOWS2_STEGO_DIR

    TRAIN_INDEX_PATH = 'index_list{}/bossbase_and_bows_train_index.npy'.format(DATASET_INDEX)
    # TRAIN_INDEX_PATH = 'index_list/bossbase_train_index.npy'
    VALID_INDEX_PATH = 'index_list{}/bossbase_valid_index.npy'.format(DATASET_INDEX)
    # TEST_INDEX_PATH = 'index_list{}/bossbase_test_index.npy'.format(DATASET_INDEX)

    train_data = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, train_transform)
    valid_data = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, valid_transform)
    # test_data = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, eval_transform)

    train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    best_acc_top1 = 0
    best_acc_top2 = 0
    lr = args.learning_rate
    for epoch in range(args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0] 

        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top2, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top2: %f', valid_acc_top2)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top2 > best_acc_top2:
            best_acc_top2 = valid_acc_top2
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.save)        
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        shape = list(input.size())
        input = input.reshape(shape[0] * shape[1], *shape[2:])
        target = target.reshape(-1)

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R2: %f Duration: %ds BTime: %.3fs', 
                                    step, objs.avg, top1.avg, top2.avg, duration, batch_time.avg)

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
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R2: %f Duration: %ds', step, objs.avg, top1.avg, top2.avg, duration)

    return top1.avg, top2.avg, objs.avg


if __name__ == '__main__':
    main() 
