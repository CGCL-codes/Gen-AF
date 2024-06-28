import os
import torch
import argparse
import random
import json
import numpy as np
from utils.load_data import load_data, normalzie
from utils.predict import make_print_to_file, test
from copy import deepcopy
from tqdm import tqdm
from utils.drc import layer_robustness_contribution
from torch import nn
from pathlib import Path

def arg_parse():
    parser = argparse.ArgumentParser(description='Train downstream models using of the pre-trained encoder')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save', default='False')
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='deepclusterv2', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse'])
    args = parser.parse_args()
    return args

def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(normalzie(args, inputs))
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, correct / total * 100, model


def load_at_model(args):
    encoder_path = os.path.join('output', str(args.pre_dataset), 'aft_model', str(args.victim), str(args.dataset),
                                'encoder')
    checkpoint = [Path(encoder_path) / ckpt for ckpt in os.listdir(Path(encoder_path)) if ckpt.endswith("last.pth")][0]
    encoder = torch.load(checkpoint)

    f_path = os.path.join('output', str(args.pre_dataset), 'aft_model', str(args.victim), str(args.dataset), 'f')
    f_checkpoint = [Path(f_path) / ckpt for ckpt in os.listdir(Path(f_path)) if ckpt.endswith("last.pth")][0]

    F = torch.load(f_checkpoint)
    model = torch.nn.Sequential(encoder, F)
    return model

def main():
    args = arg_parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(profile="full")
    torch.cuda.synchronize()

    # Logging
    log_save_path = os.path.join('output', str(args.pre_dataset), 'log', 'down_test', "2aft_model", str(args.victim),
                                 str(args.dataset))
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    now_time = make_print_to_file(path=log_save_path)

    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    # Dump args
    with open(log_save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    uap_save_path_e = os.path.join('output', str(args.pre_dataset), '2aft_model', str(args.victim), str(args.dataset), 'encoder')
    uap_save_path_f = os.path.join('output', str(args.pre_dataset), '2aft_model', str(args.victim), str(args.dataset), 'f')

    if not os.path.exists(uap_save_path_e):
        os.makedirs(uap_save_path_e)

    if not os.path.exists(uap_save_path_f):
        os.makedirs(uap_save_path_f)

    # load data
    train_loader, test_loader = load_data(args.dataset, args.batch_size)

    print('Day: %s, Target encoder:%s, Downstream task:%s' % (now_time, args.victim, args.dataset))
    print("######################################  Test Attack! ######################################")

    print('==> Phase 1: Adversarial Fine-Tuning...')
    model = load_at_model(args)
    
    clean_acc_t1, clean_acc_t5 = test(args, model, test_loader)
    print('Clean downstream accuracy: %.4f%%' % (clean_acc_t1))

    print('==> Phase 2: Standard Fine-Tuning...')
    sorted_layer_robustness_contribution, non_robust_cd_layer_min_top10p = layer_robustness_contribution(deepcopy(model), epsilon=0.1)

    # Dump non_robust_cd_layer & value
    with open(log_save_path + '/layer.json', 'w') as fid:
        json.dump(sorted_layer_robustness_contribution, fid, indent=2)

    for name, param in model.named_parameters():
        param.requires_grad = False
        for n_key in non_robust_cd_layer_min_top10p:
            if n_key in name:
                param.requires_grad = True

    best_test_acc = 0
    for epoch in range(args.epochs):
        print("==> Epoch {}".format(epoch))
        print("==> Training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0008)
        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc, model = train(args, model, train_loader, optimizer, criterion)

        print("==> Train loss: {:.2f}, train acc: {:.2f}%".format(train_loss, train_acc))
        clean_acc_t1, clean_acc_t5 = test(args, model, test_loader)

        print('Epoch [%d/%d], Top1 train acc: %.4f, Top1 test acc: %.4f'
              % (epoch + 1, args.epochs, train_acc, clean_acc_t1))

        if args.save == 'True':
            if clean_acc_t1 > best_test_acc:
                best_test_acc = clean_acc_t1
                print('Best test acc: %.4f' % (best_test_acc))
                torch.save(model[0],
                           '{}/{}'.format(uap_save_path_e, str(args.victim) + '_' + str(args.pre_dataset) + '_' + str(
                               args.dataset) + '_last' + '.pth'))
                # save F
                torch.save(model[1],
                           '{}/{}'.format(uap_save_path_f, str(args.victim) + '_' + str(args.pre_dataset) + '_' + str(
                               args.dataset) + '_last' + '.pth'))

if __name__ == "__main__":
    main()
