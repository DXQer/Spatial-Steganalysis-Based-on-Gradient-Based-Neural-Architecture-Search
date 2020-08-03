import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkSteganalysis as Network

#import pandas as pd
import cv2
import random
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("steganalysis")
#parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('-i', '--DATASET_INDEX', help='Path for loading dataset', type=str, default='')
parser.add_argument('-alg', '--STEGANOGRAPHY', help='embedding_algorithm', type=str, choices=['s-uniward','wow','hill'], default='s-uniward')
parser.add_argument('-rate', '--EMBEDDING_RATE', help='embedding_rate', type=str, choices=['0.1', '0.2', '0.3', '0.4'], default='0.4')

parser.add_argument('--BOSSBASE_COVER_DIR', help='The path to load BOSSBase_COVER_dataset', type=str, required=True) 
parser.add_argument('--BOSSBASE_STEGO_DIR', help='The path to load BOSSBase_STEGO_dataset', type=str, required=True)
parser.add_argument('--BOWS2_COVER_DIR', help='The path to load BOWS2_COVER_dataset', type=str, required=True) 
parser.add_argument('--BOWS2_STEGO_DIR', help='The path to load BOWS2_STEGO_dataset', type=str, required=True) 

parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default="0,1,2,3", help='gpu device id')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 2

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


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  gpus = [int(i) for i in args.gpu.split(',')]
#  torch.cuda.set_device(gpus)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = torch.nn.DataParallel(model)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  test_transform = transforms.Compose([ToTensor(),])
  
  DATASET_INDEX = args.DATASET_INDEX
  STEGANOGRAPHY = args.STEGANOGRAPHY
  EMBEDDING_RATE = args.EMBEDDING_RATE

  BOSSBASE_COVER_DIR = args.BOSSBASE_COVER_DIR
  BOSSBASE_STEGO_DIR = args.BOSSBASE_STEGO_DIR
    
  BOWS_COVER_DIR = args.BOWS2_COVER_DIR
  BOWS_STEGO_DIR = args.BOWS2_STEGO_DIR

 
  TEST_INDEX_PATH = 'index_list{}/bossbase_test_index.npy'.format(DATASET_INDEX)

  test_data = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, test_transform)

  test_queue = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj = infer(test_queue, model, criterion)
  print(test_acc)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top2 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    shape = list(input.size())
    input = input.reshape(shape[0] * shape[1], *shape[2:])
    target = target.reshape(-1)
    
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top2.update(prec2.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

