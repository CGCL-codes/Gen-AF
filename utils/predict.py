import os
import sys
import datetime
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.load_data import normalzie

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    return fileName

def accuracy(new_output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = new_output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def rob_test(args, model, test_loader):
    adv_test_dataset = adv_dataset()
    model.eval()
    # atk_model = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    atk_model = torchattacks.FGSM(model, eps=8 / 255)
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        adv_images = atk_model(images, labels)
        adv_test_dataset.append_data(adv_images, labels)

    adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=256, shuffle=True, num_workers=0)
    top1_accuracy = 0
    top5_accuracy = 0

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(adv_test_loader)):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            logits = model((normalzie(args, x_batch)))
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()

def test(args, model, test_loader):
    top1_accuracy = 0
    top5_accuracy = 0

    model.eval()
    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            logits = model((normalzie(args, x_batch)))
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
    return top1_accuracy.item(), top5_accuracy.item()