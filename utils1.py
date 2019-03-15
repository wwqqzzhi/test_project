from __future__ import print_function

import torch
from torch.autograd import Variable
import scipy.cluster.hierarchy as hc
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


# Compute the number of channels in a model
def compute_num_channels(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.shape[0]
    return total


# Compute the number of parameters in a model
def compute_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


# Compute the number of flops in a model
def compute_num_flops(model, use_cuda=True, size=32, multiply_adds=True):
    conv_flops = []
    linear_flops = []
    bn_flops = []
    relu_flops = []
    pooling_flops = []

    def conv_hook(self, inputs, outputs):
        batch_size, input_channels, input_height, input_width = inputs[0].size()
        output_channels, output_height, output_width = outputs[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        conv_flops.append(flops)

    def linear_hook(self, inputs, outputs):
        batch_size = inputs[0].size(0) if inputs[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        linear_flops.append(flops)

    def bn_hook(self, inputs, outputs):
        bn_flops.append(inputs[0].nelement() * 2)

    def relu_hook(self, inputs, outputs):
        relu_flops.append(inputs[0].nelement())

    def pooling_hook(self, inputs, outputs):
        batch_size, input_channels, input_height, input_width = inputs[0].size()
        output_channels, output_height, output_width = outputs[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        flops = kernel_ops * output_channels * output_height * output_width * batch_size
        pooling_flops.append(flops)

    def add_flops_counter_hooks(net):
        handles = []
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):
                handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, nn.Linear):
                handles.append(net.register_forward_hook(linear_hook))
            if isinstance(net, nn.BatchNorm2d):
                handles.append(net.register_forward_hook(bn_hook))
            if isinstance(net, nn.ReLU):
                handles.append(net.register_forward_hook(relu_hook))
            if isinstance(net, nn.MaxPool2d) or isinstance(net, nn.AvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook))
            return handles
        for c in childrens:
            handles += add_flops_counter_hooks(c)
        return handles

    handles = add_flops_counter_hooks(model.eval())
    inputs = Variable(torch.rand(3, 3, size, size))
    if use_cuda:
        inputs = inputs.cuda()
    outputs = model(inputs)
    for handle in handles:
        handle.remove()
    total_flops = (sum(conv_flops) + sum(linear_flops) + sum(bn_flops) + sum(relu_flops) + sum(pooling_flops))
    return total_flops / 3


def ssim(filter1, filter2, mean1, mean2, mean1_square, mean2_square, var1, var2):
    cov = (filter1 * filter2).mean(-1).mean(-1) - mean1 * mean2
    dist = (4 * mean1 * mean2 * cov / ((mean1_square + mean2_square) * (var1 + var2))).mean().item()
    return dist


def distance_matrix(filters):
    num_channels = filters.shape[0]
    Dist = []
    mean = filters.mean(-1).mean(-1)
    mean_square = mean.pow(2)
    std = torch.sqrt((filters * filters).mean(-1).mean(-1) - mean_square)
    filters.sub_(mean.view(mean.shape[0], mean.shape[1], 1, 1))
    filters.div_(std.view(std.shape[0], std.shape[1], 1, 1))
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            # dist = 1.0 - ssim(filters[i], filters[j], mean[i], mean[j], mean_square[i], mean_square[j], var[i], var[j])
            dist = (filters[i] - filters[j]).abs().mean().item()
            Dist.append(dist)
    assert len(Dist) == num_channels * (num_channels - 1) // 2
    return np.array(Dist, dtype=np.float32)


def hierarchical_clustering(filters, threshold=1.0):
    Dist = distance_matrix(filters)
    Z = hc.linkage(Dist, method='complete')
    clusters = hc.fcluster(Z, t=threshold, criterion='distance')
    return clusters, Dist


def create_mask(clusters, filters):
    mask = np.zeros(clusters.shape[0])
    num_clusters = int(clusters.max())
    print("Number of clusters: {:d}".format(num_clusters))
    for i in range(num_clusters):
        max = -np.inf
        index = -1
        for j in range(clusters.shape[0]):
            if clusters[j] == (i + 1):
                temp = filters[j].abs().sum().item()
                if temp > max:
                    max = temp
                    index = j
        mask[index] = 1.0
    return np.int64(mask > 0)
