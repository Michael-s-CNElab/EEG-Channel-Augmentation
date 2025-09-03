import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from tf_utils import draw


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        #self.criterion = criterion
        self.criterion = nn.MSELoss()
        self.opt = opt

    def __call__(self, x, y, norm):
        #time.sleep(5)
        #print("SimpleLossCompute1:", x.shape, y.shape)
        #draw(3, x[0, :, :], "Gen_"+str(1))
        x = self.generator(x)
        #print("SimpleLossCompute2:", x.shape, y.shape)
        #draw(2, x[0, :, :], "Gen_(decode)" + str(4))
        #temp = y.permute(0, 2, 1)
        #draw(2, temp[0, :, :], "Gen(GT)_" + str(5))
        #draw(3, x, "Gen_" + str(2))

        # Compute the z-scores
        x_mean = torch.mean(x.permute(0, 2, 1), dim=0, keepdim=True)
        x_std = torch.std(x.permute(0, 2, 1), dim=0, keepdim=True)
        x = (x.permute(0, 2, 1) - x_mean) / (x_std + 1e-10)

        y_mean = torch.mean(y, dim=0, keepdim=True)
        y_std = torch.std(y, dim=0, keepdim=True)
        y = (y - y_mean) / (y_std + 1e-10)

        print("SimpleLossCompute3: ", y.shape)
        loss = self.criterion(x, y)
        #print("SimpleLossCompute3:", loss.item())
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm