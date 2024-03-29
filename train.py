#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/11 

# implementing EW-MTCNN in essay 基于情感轮和多任务卷积神经网络的图像情感分布学习 (赖金水，万中英，曾雪强)
# https://lkxb.jxnu.edu.cn/oa/pdfdow.aspx?Sid=202204006

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision.models as M
from torchvision.models.vgg import VGG19_BN_Weights, VGG
import matplotlib as mpl ; mpl.use('agg')
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import seaborn as sns

from data import Emotion6Dataset, DataLoader
from utils import *

# 按照论文设置
LR_FEAT = 0.001
LR_CLF = 0.01
LAMBDA = 0.7      
SIGMA = 0.6
EPOCHS = 20
BATCH_SIZE = 32


def EW_layer_savefig(mat:Tensor, title:str='EW_layer-weight'):
  fp = IMG_PATH / f'{title}.png'
  if fp.exists(): return
  ax = plt.gca()
  sns.heatmap(mat.cpu().numpy(), cbar=True, ax=ax)
  ax.set_xticklabels(Emotion6Dataset.class_names)
  ax.set_yticklabels(Emotion6Dataset.class_names)
  plt.suptitle(title)
  plt.savefig(fp, dpi=600)


def make_EW_layer_weight_original(sigma:float=SIGMA) -> Tensor:
  ''' Emotion6 label index 0~5: anger disgust fear joy/amusement sadness surprise/awe '''

  # 顺时针次序
  EMOTION_WHEEL_NODES = [
    'fear',
    'sadness',
    'disgust',
    'anger',
    'amusement',
    'contentment',
    'awe',
    'excitement',
  ]
  EMOTION_WHEEL_LEN = len(EMOTION_WHEEL_NODES)

  def dist_EW(emo1:str, emo2:str) -> int:
    if emo1 == emo2: return 0
    assert all([emo in EMOTION_WHEEL_NODES for emo in [emo1, emo2]])
    emo1_idx = EMOTION_WHEEL_NODES.index(emo1)
    emo2_idx = EMOTION_WHEEL_NODES.index(emo2)
    if emo1_idx > emo2_idx:       # assure incr
      emo1_idx, emo2_idx = emo2_idx, emo1_idx
    dist = emo2_idx - emo1_idx    # assert >0
    return min(dist, EMOTION_WHEEL_LEN - dist)

  EMOTION6_NODES = Emotion6Dataset.class_names
  EMOTION6_ALIAS = {    # Emotion6 => EmotionWheel
    'joy': 'amusement',
    'surprise': 'awe',
  }
  EMOTION6_LEN = len(EMOTION6_NODES)
  mat = torch.empty([EMOTION6_LEN, EMOTION6_LEN])
  for i, emo1 in enumerate(EMOTION6_NODES):
    for j, emo2 in enumerate(EMOTION6_NODES):
      emo1 = EMOTION6_ALIAS.get(emo1, emo1)
      emo2 = EMOTION6_ALIAS.get(emo2, emo2)
      mat[i, j] = np.exp(-dist_EW(emo1, emo2)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
  mat /= mat.sum(dim=1, keepdim=True)  # 行归一化作为权重

  EW_layer_savefig(mat, 'EW_layer-original')
  return mat


def get_corr_mat() -> Tensor:
  ''' 数据集导出的相关性矩阵 '''
  return torch.FloatTensor([
    [ 1.000000,  0.169003,  0.287530, -0.344558, -0.043495, -0.244552],
    [ 0.169003,  1.000000,  0.052773, -0.458298, -0.078017, -0.384233],
    [ 0.287530,  0.052773,  1.000000, -0.501457, -0.054263, -0.238941],
    [-0.344558, -0.458298, -0.501457,  1.000000, -0.457448,  0.438189],
    [-0.043495, -0.078017, -0.054263, -0.457448,  1.000000, -0.436899],
    [-0.244552, -0.384233, -0.238941,  0.438189, -0.436899,  1.000000],
  ])

def make_EW_layer_weight_corr() -> Tensor:
  mat = get_corr_mat()
  EW_layer_savefig(mat, 'EW_layer-corr')
  return mat

def make_EW_layer_weight_corr_softmax() -> Tensor:
  mat = get_corr_mat()
  mat = F.softmax(mat, dim=-1)
  EW_layer_savefig(mat, 'EW_layer-corr_softmax')
  return mat

def make_EW_layer_weight_corr_log_softmax() -> Tensor:
  mat = get_corr_mat()
  mat = F.log_softmax(mat, dim=-1)
  EW_layer_savefig(mat, 'EW_layer-corr_log_softmax')
  return mat


def get_model(n_class:int, init:str='original', mode:str='freeze') -> VGG:
  assert mode in ['freeze', 'unfreeze']
  from torch.nn import Linear

  use_EW = init != 'none'
  requires_grad = mode == 'unfreeze'

  # 预训练的VGG模型
  model = M.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1, dropout=0.5)
  # 倒数第二个Linear层，解释为图像表征向量
  layer: Linear = model.classifier[-4]
  new_layer = Linear(in_features=layer.in_features, out_features=n_class, bias=True)
  model.classifier[-4] = new_layer
  # 倒数第一个Linear层，替换为情感轮先验知识层
  if use_EW:
    layer: Linear = model.classifier[-1]
    new_layer = Linear(in_features=n_class, out_features=n_class, bias=False)
    if init == 'rand':
      pass
    elif init == 'eye':
      new_layer.weight.data = nn.Parameter(torch.eye(len(Emotion6Dataset.class_names)))
    else:
      init_fn = globals()[f'make_EW_layer_weight_{init}']
      new_layer.weight.data = nn.Parameter(init_fn())
    new_layer.requires_grad_(requires_grad)
    model.classifier[-1] = new_layer
  else:   # no EW_layer
    model.classifier[-1] = nn.Identity()
    model.classifier[-2] = nn.Identity()
    model.classifier[-3] = nn.Identity()
  return model


def train(args):
  seed_everything()

  train_dataset = Emotion6Dataset('train')
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

  model = get_model(train_dataset.n_class, args.init, args.mode).to(device)
  print(f'param_cnt: {sum([p.numel() for p in model.parameters()])}')
  print(f'  trainable: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
  print(f'  frozen: {sum([p.numel() for p in model.parameters() if not p.requires_grad])}')

  param_groups = [
    {'params': model.features  .parameters(), 'lr': LR_FEAT},              
    {'params': model.classifier.parameters(), 'lr': LR_CLF },
  ]
  optimizer = SGD(param_groups, momentum=0.9, weight_decay=5e-4)

  acc_train, kl_train = [], []
  step = 0
  for epoch in range(EPOCHS):
    ''' Train '''
    model.train()
    for X, Y, Z in train_loader:
      X, Y, Z = X.to(device), Y.to(device), Z.to(device)

      optimizer.zero_grad()
      output = model(X)
      loss_cls = F.cross_entropy(output, Y)
      loss_sdl = F.kl_div(F.log_softmax(output, dim=-1), Z, reduction='batchmean')
      loss = (1 - LAMBDA) * loss_cls + LAMBDA * loss_sdl
      loss.backward()
      optimizer.step()

      step += 1

      if step % 10 == 0:
        print(f'>> [step {step}] loss: {loss.item()}')


    ''' Eval '''
    tot, ok, kl = 0, 0, 0.0
    with torch.inference_mode():
      model.eval()
      for X, Y, Z in train_loader:
        X, Y, Z = X.to(device), Y.to(device), Z.to(device)

        output = model(X)   # [B, NC]

        ok += (torch.argmax(output, dim=-1) == Y).sum().item()
        kl_raw = F.kl_div(F.log_softmax(output, dim=-1), Z, reduction='none')
        kl += kl_raw.mean(dim=-1).sum().item()
        tot += len(Y)
  
      print(f'>> [Epoch: {epoch + 1}/{EPOCHS}] cls_acc: {ok / tot:.3%}, sdl_kl: {kl / tot:.7f}')

    acc_train.append(ok / tot)
    kl_train.append(kl / tot)

  if 'plot':
    plt.clf()
    plt.plot(acc_train, 'r', label='train accuracy')
    plt.legend()
    ax: Axes = plt.twinx()
    ax.plot(kl_train, 'b', label='train kl_div')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_PATH / f'{args.name}.png', dpi=600)

  if args.save:
    torch.save(model.state_dict(), LOG_PATH / f'{args.name}.pth')


if __name__ == '__main__':
  EW_WEIGHT_DEFS = [k[len('make_EW_layer_weight_'):] for k in globals() if k.startswith('make_EW_layer_weight_')]

  parser = ArgumentParser()
  parser.add_argument('-I', '--init', default='original', choices=['none', 'rand', 'eye'] + EW_WEIGHT_DEFS)
  parser.add_argument('-M', '--mode', default='freeze', choices=['freeze', 'unfreeze'])
  parser.add_argument('--save', action='store_true', help='save *.pth file')
  args = parser.parse_args()

  args.name = f'model-I={args.init}_M={args.mode}'

  train(args)
