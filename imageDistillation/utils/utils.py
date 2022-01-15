import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization
from PIL import Image

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=5994, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class ImageTransform(object):
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])

    def __init__(self, arch_type="senet"):
        self.arch_type = arch_type
        if arch_type=="facenet":
            self.t = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                fixed_image_standardization
            ])
        else:
            self.t = transforms.Compose([
                transforms.Resize((256)),
                transforms.CenterCrop(224)
            ])

    def _transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def transform(self, path, img=None):
        if img is None:
            img = Image.open(path).convert('RGB')
        else:
            img = Image.fromarray(img).convert('RGB')
        img = self.t(img)

        if self.arch_type=="facenet":
            return img

        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3
        img = self._transform(img)
        return img.unsqueeze(0)

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def compute_loss(args, pred, gt):
    loss = torch.FloatTensor([0.0]).cuda()

    if args.kldiv_coeff:
        loss += args.kldiv_coeff * nn.KLDivLoss()(pred, gt)

    if args.mse_coeff:
        loss += args.mse_coeff * nn.MSELoss()(pred, gt)

    if args.l1_coeff:
        loss += args.l1_coeff * nn.L1Loss()(pred, gt)

    # if args.cosembed_coeff:
    #     loss += args.cosembed_coeff * nn.CosineEmbeddingLoss()(pred,gt,torch.FloatTensor([1.0]).cuda())

    return loss
