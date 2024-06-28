import numpy as np
import argparse
import torch
import torch.nn as nn

def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)

def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)


def genetic_regularization(adv_results, ben_results, y, sample_weight=0, eps=0.0000001):
    (n, d) = adv_results.shape
    distance = 2.0
    tahn =  nn.Tanh()
    sample_weight = sample_weight.view(-1,n)
    sample_weight_matrix = (sample_weight+sample_weight.t())/32.0
    sample_weight_matrix = tahn(sample_weight_matrix)
    y  = y.view(-1,n)
    mask =1- (y==y.t()).float()
    mask[mask == 0] = -1
    distance = distance*mask*sample_weight_matrix
    adv_results_norm = torch.sqrt(torch.sum(adv_results ** 2, dim=1, keepdim=True))
    adv_results = adv_results / (adv_results_norm + eps)
    adv_results[adv_results != adv_results] = 0

    ben_results_norm = torch.sqrt(torch.sum(ben_results ** 2, dim=1, keepdim=True))
    ben_results = ben_results / (ben_results_norm + eps)
    ben_results[ben_results != ben_results] = 0

    model_similarity = torch.mm(adv_results, adv_results.transpose(0, 1))
    model_distance = 1 - model_similarity
    model_distance[range(n), range(n)] = 100000
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_distance = torch.clamp(model_distance, 0+eps, 2.0-eps)
    model_similarity = 1 - model_distance

    target_similarity = torch.mm(ben_results, ben_results.transpose(0, 1))
    target_distance = 1 - target_similarity
    target_distance[range(n), range(n)] = 100000
    p = torch.min(target_distance, dim=1)
    target_distance = target_distance - p[0].view(-1, 1)
    target_distance[range(n), range(n)] = 0
    target_distance = (1 - sample_weight_matrix) * target_distance + distance
    target_distance = torch.clamp(target_distance, 0+eps, 2.0-eps)
    target_similarity = 1 - target_distance

    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    loss = CE(target_similarity, model_similarity)
    return loss
