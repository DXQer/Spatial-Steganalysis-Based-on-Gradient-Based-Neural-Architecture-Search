import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path
import numpy as np

from srm_filter_kernel import all_normalized_hpf_list

from MPNCOV.python import MPNCOV

class MixedHpf(nn.Module):
  def __init__(self, genotype):
    super(MixedHpf, self).__init__()

    self._hpfs = nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False)
    self.tlu = TLU(5.0)

    filt_list = build_filters()

    hpf_weight = torch.Tensor(filt_list)

    hpf_selected = genotype.hpf_select
    hpf_weight_select = []

    for i in hpf_selected:
      hpf_weight_select.append(hpf_weight[i])
    
    hpf_weight_select = torch.cat(hpf_weight_select, dim=0)
    hpf_weight_selected = nn.Parameter(hpf_weight_select.view(16, 1, 5, 5), requires_grad=False)

    self._hpfs.weight = hpf_weight_selected

  def forward(self, x):
    output = self._hpfs(x)
    output = self.tlu(output)
    return output
    
class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)




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
        

class NetworkSteganalysis(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkSteganalysis, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self.drop_path_prob = 0.2
    self.stem0 = nn.Sequential( 
      MixedHpf(genotype),
      
    )

    self.stem1 = nn.Sequential(
#      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
      nn.ReLU(inplace=True),
    )

    C_prev_prev, C_prev, C_curr = 16, 16, C

    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [1, 2]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
#    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(int(C_prev*(C_prev+1)/2), num_classes)
 
  def forward(self, input):
    logits_aux = None
    s0 =  self.stem0(input)
    s1 = s0
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
#    out = self.global_pooling(s1)
    out = MPNCOV.CovpoolLayer(s1)
    out = MPNCOV.SqrtmLayer(out, 5)
    out = MPNCOV.TriuvecLayer(out)
    
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux
