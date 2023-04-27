import torch
from torch.autograd import Variable
from model_ops.resnet import *



class NN_Trainer(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.lr = kwargs['lr']
        self.momentum = kwargs['momentum']

    def build_model(self, num_classes=10):
        self.network = ResNet18(num_classes)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def evaluate(self, val_loader):
        self.network.eval()
        prec1_counter  = batch_counter = 0
        while val_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_image_batch, eval_label_batch = val_loader.next_batch(batch_size=self.eval_batch_size)
            X_batch, y_batch = Variable(eval_image_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            prec1_tmp = self.accuracy(output.detach(), eval_label_batch.long())
            prec1_counter += prec1_tmp
            batch_counter += 1
        prec1 = prec1_counter / batch_counter
        self._epoch_counter = val_loader.dataset.epochs_completed
        print('Evaluation Performance: Step:{} Precision@1: {}'.format(self.cur_step, prec1.numpy()[0]))