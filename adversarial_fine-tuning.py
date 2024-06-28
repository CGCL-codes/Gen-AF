import os
import torch
import argparse
import time
import random
import json
import numpy as np
from tqdm import tqdm
from utils.load_model import load_victim
from utils.load_data import load_data, normalzie
from utils.predict import make_print_to_file, test, rob_test
from model.linear import NonLinearClassifier
from utils.gr import genetic_regularization

def arg_parse():
    parser = argparse.ArgumentParser(description='Genetic-Driven Dual-Track Adversarial Finetuning')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save', default='False')
    parser.add_argument('--pre_dataset', default='imagenet', choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='deepclusterv2', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse'])
    args = parser.parse_args()
    return args

def pgd_attack(model, x, y, loss_fn, epsilon, num_steps, step_size):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        # Calculate gradients
        loss.backward()
        x_adv = x_adv + step_size * x_adv.grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure pixel values are in [0, 1] range
        x_adv.detach_()
        x_adv.requires_grad_(True)
    return x_adv


def classify(args, encoder):
    data = args.dataset
    train_loader, test_loader = load_data(data, args.batch_size)

    uap_save_path_e = os.path.join('output', str(args.pre_dataset), 'aft_model', str(args.victim), str(args.dataset), 'encoder')
    uap_save_path_f = os.path.join('output', str(args.pre_dataset), 'aft_model', str(args.victim), str(args.dataset), 'f')

    if not os.path.exists(uap_save_path_e):
        os.makedirs(uap_save_path_e)

    if not os.path.exists(uap_save_path_f):
        os.makedirs(uap_save_path_f)

    # downstream task
    if args.dataset == 'imagenet':
        num_classes = 100
        args.epochs = 50
    elif args.dataset == 'gtsrb':
        num_classes = 43
    else:
        num_classes = 10

    F = NonLinearClassifier(feat_dim=512, num_classes=num_classes)
    F.cuda()

    encoder.cuda()
    model = torch.nn.Sequential(encoder, F)

    e_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001, weight_decay=0.0008)
    e_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=e_optimizer, gamma=0.96)

    f_optimizer = torch.optim.Adam(F.parameters(), lr=0.005, weight_decay=0.0008)
    f_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=f_optimizer, gamma=0.96)

    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            e_optimizer.zero_grad()
            f_optimizer.zero_grad()
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            epsilon = 0.03  # Maximum perturbation allowed
            num_steps = 10  # Number of PGD steps
            step_size = 0.007  # Step size for each step in PGD
            x_adv = pgd_attack(model, x_batch, y_batch, criterion, epsilon, num_steps, step_size)
            model.train()
            # Forward pass on the adversarial examples
            adv_logits = model((normalzie(args, x_adv)))
            clean_logits = model((normalzie(args, x_batch)))
            sample_weight = torch.nn.CrossEntropyLoss(reduction='none')(clean_logits, y_batch)
            tp_loss = genetic_regularization(clean_logits, adv_logits, y_batch, sample_weight)

            ft_loss = criterion(adv_logits, y_batch)
            loss = 20 * tp_loss + ft_loss
            print( f"Epoch: [{epoch+1}/{args.epochs}], TP Loss: {tp_loss.item():.4f}, FT Loss: {ft_loss.item():.4f}, Total Loss: {loss.item():.4f}")

            top1 = accuracy(adv_logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            loss.backward()
            e_optimizer.step()
            f_optimizer.step()

        end = time.time()
        clean_acc_t1, clean_acc_t5 = test(args, model, test_loader)
        adv_acc_t1, adv_acc_t5 = rob_test(args, model, test_loader)

        if args.save == 'True':
            best_adv_acc_t1 = 0
            best_clean_acc_t1 = 0
            if clean_acc_t1 > best_clean_acc_t1:
                best_clean_acc_t1 = clean_acc_t1
            if adv_acc_t1 > best_adv_acc_t1:
                best_adv_acc_t1 = adv_acc_t1
                # save encoder
                torch.save(encoder,
                           '{}/{}'.format(uap_save_path_e, str(args.victim) + '_' + str(args.pre_dataset) + '_' + str(
                               args.dataset) + '_last' + '.pth'))
                # save F
                torch.save(F,
                           '{}/{}'.format(uap_save_path_f, str(args.victim) + '_' + str(args.pre_dataset) + '_' + str(
                               args.dataset) + '_last' + '.pth'))
            print('Best test acc: %.4f, Best adv acc: %.4f'
                  % (best_clean_acc_t1, best_adv_acc_t1))

        e_lr_scheduler.step()
        f_lr_scheduler.step()

        top1_train_accuracy /= (counter + 1)
        print('Epoch [%d/%d], Top1 train acc: %.4f, Top1 test acc: %.4f, Top1 adv acc: %.4f, Time: %.4f'
              % (epoch + 1, args.epochs, top1_train_accuracy.item(), clean_acc_t1, adv_acc_t1, (end - start)))

    return clean_acc_t1, clean_acc_t5, adv_acc_t1, adv_acc_t5


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
    log_save_path = os.path.join('new_output', str(args.pre_dataset), 'log', 'down_test', "aft_model", str(args.victim),
                                 str(args.dataset))
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    now_time = make_print_to_file(path=log_save_path)

    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    # Dump args
    with open(log_save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    model = load_victim(args)

    print('Day: %s, Target encoder:%s, Downstream task:%s' % (now_time, args.victim, args.dataset))
    print("######################################  Test Attack! ######################################")

    clean_acc_t1, clean_acc_t5, adv_acc_t1, adv_acc_t5 = classify(args, model)
    print('Clean downstream accuracy: %.4f%%' % (clean_acc_t1))
    print('Adv downstream accuracy: %.4f%%' % (adv_acc_t1))


if __name__ == "__main__":
    main()
