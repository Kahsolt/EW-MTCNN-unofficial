import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from utils import *

DATA_PATH = BASE_PATH / 'data'
DATA_EMOTION6_PATH = DATA_PATH / 'Emotion6'

RESIZE = (224, 224)

# torchvision pretrained using ImageNet stats 
TV_MEAN = [0.485, 0.456, 0.406]
TV_STD  = [0.229, 0.224, 0.225]


class Emotion6Dataset(Dataset):

  class_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
  n_class = len(class_names)

  transform_train = T.Compose([
    T.RandomResizedCrop(RESIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=TV_MEAN, std=TV_STD),
  ])
  transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=TV_MEAN, std=TV_STD),
  ])

  def __init__(self, split:str='train', root:Path=DATA_EMOTION6_PATH):
    self.root = root
    self.split = split
    self.transform = self.transform_train if split == 'train' else self.transform_test
    self.metadata = pd.read_csv(root / 'ground_truth.txt', sep='\t')

  def __getitem__(self, idx:int):
    row: list = self.metadata.iloc[idx].to_list()
    fn: str = row[0]
    label_cls = self.class_names.index(fn.split('/')[0])
    pdist = np.asarray(row[3:9], dtype=np.float32)    # collect 6 dims, ignore 'neutral'
    label_sdl = pdist / pdist.sum()                   # re-norm as pdist
    image = load_pil(self.root / 'images' / fn)
    image = self.transform(image)
    return image, label_cls, label_sdl

  def __len__(self):
    return len(self.metadata)

  @staticmethod
  def show_stats():
    IMAGE_PATH = DATA_EMOTION6_PATH / 'images'
    ims = []
    for dp in tqdm((IMAGE_PATH.iterdir())):
      for fp in tqdm(list(dp.iterdir())):
        img = load_pil(fp)
        img = img.resize(RESIZE, resample=Resampling.BILINEAR)
        ims.append(pil_to_npimg(img))
    X = torch.from_numpy(np.stack(ims, axis=0))
    X = X.permute(0, 3, 1, 2)

    # [N=1980, C=3, H=224, W=224]
    print('X.shape', X.shape)
    # [0.4165, 0.3834, 0.3488]
    print('mean:', X.mean(axis=[0, 2, 3]))
    # [0.2936, 0.2805, 0.2850]
    print('std:',  X.std (axis=[0, 2, 3]))


if __name__ == '__main__':
  dataset = Emotion6Dataset()
  for img, lbl_cls, lbl_sdl in iter(dataset):
    print('img:', img)
    print('lbl_cls:', lbl_cls)
    print('lbl_sdl:', lbl_sdl, 'sum:', np.sum(lbl_sdl))
    break
