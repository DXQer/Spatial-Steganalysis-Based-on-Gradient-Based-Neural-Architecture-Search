import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from srm_filter_kernel import all_normalized_hpf_list
from MPNCOV.python import MPNCOV

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    for primitive in PRIMITIVES:
      op = OPS[primitive](C//2, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C//2, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//2, :, :]
    xtemp2 = x[ : ,  dim_2//2:, :, :]
    
    temp1 = sum(w.to(xtemp.device) * op(xtemp) for w, op in zip(weights, self._ops))
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)

    ans = channel_shuffle(ans,2) 

    return ans
    
    #return sum(w.to(x.device) * op(x) for w, op in zip(weights, self._ops))

class MixedHpf(nn.Module):
  def __init__(self):
    super(MixedHpf, self).__init__()

    self._hpfs = nn.ModuleList()
    self.tlu = TLU(5.0)

    filt_list = build_filters()

    for filter_ in filt_list:
      hpf_weight = nn.Parameter(torch.Tensor(filter_).view(1, 1, 5, 5), requires_grad=False)
      residual = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
      residual.weight = hpf_weight

      self._hpfs.append(residual)

  def forward(self, x, weights):
    result = []
    for w, hpf in zip(weights, self._hpfs):
      res = w.to(x.device) * self.tlu(hpf(x))
      result.append(res)
    output = torch.cat(result, dim=1)
    return output

    
class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights,weights2):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j].to(self._ops[offset+j](h, weights[offset+j]).device)*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      #s = channel_shuffle(s,4)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output

def build_filters():
    filters = []
    for hpf_item in all_normalized_hpf_list:
      row_1 = int((5 - hpf_item.shape[0])/2)
      row_2 = int((5 - hpf_item.shape[0])-row_1)
      col_1 = int((5 - hpf_item.shape[1])/2)
      col_2 = int((5 - hpf_item.shape[1])-col_1)
      hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
      filters.append(hpf_item)
      
    return filters
        

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=1, multiplier=1, stem_multiplier=1):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    
    self.stem = MixedHpf()
    

    self.stem1 = nn.Sequential(
#      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
      nn.ReLU(inplace=True),
    )

 
    C_prev_prev, C_prev, C_curr = 30, 30, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [1, 2]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

#    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(int(C_prev*(C_prev+1)//2), num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    hpf_weights = F.softmax(self.alphas_hpf, dim=-1)
    s0 = self.stem(input, hpf_weights)
    s1 = s0
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights,weights2)
#    out = self.global_pooling(s1)
    out = MPNCOV.CovpoolLayer(s1)
    out = MPNCOV.SqrtmLayer(out, 5)
    out = MPNCOV.TriuvecLayer(out)
    
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_filter = len(all_normalized_hpf_list)
    num_ops = len(PRIMITIVES)

    self.alphas_hpf = Variable(1e-3*torch.randn(num_filter).cuda(), requires_grad=True)
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
      self.alphas_hpf,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
            W[j,:] = W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    
    def _parse_hpf(weights):
      gene = []
      W = weights.copy()
      select = sorted(range(len(W)), key=lambda x: -W[x])[:16]

      return select

    gene_hpf = _parse_hpf(F.softmax(self.alphas_hpf, dim=-1).data.cpu().numpy())
    
    genotype = Genotype(
      hpf_select=gene_hpf,
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

