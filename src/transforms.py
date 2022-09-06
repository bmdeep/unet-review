
import torch
# from torch.utils.data import transform
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import numpy as np



class Resize(object):
    """Resize the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, msk = sample['img'], sample['msk']

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(img, (new_h, new_w))
        msk = resize(msk, (new_h, new_w), interpolation=InterpolationMode.NEAREST)
        
        return {'img': img, 'msk': msk}


class Normalize(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        img, msk = sample['img'], sample['msk']
        
        img = (img - img.min()) / (img.max() - img.min())
        msk = (msk - msk.min()) / (msk.max() - msk.min())

        return {'img': img, 'msk': msk}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        h, w = x.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[top: top + new_h, left: left + new_w]
        y = y[top: top + new_h, left: left + new_w]

        return {'x': x, 'y': y}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        # swap color axis because
        # numpy x: H x W x C
        # torch x: C x H x W
        x = x.transpose((2, 0, 1))
        return {'x': torch.from_numpy(x),
                'y': torch.from_numpy(y)}
