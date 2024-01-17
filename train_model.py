import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import argparse

from pipelines.utils import set_seed, get_binary_metrics, get_regression_metrics
from dataset.EHRDataset import EhrDataset, pad_collate
from models.dlModel import DLModel


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--fill_lr', default=1e-2, type=float, help='learning rate of learnable features')
    parser.add_argument('--fill', action='store_true', default=False, help='enable learnable features')
    parser.add_argument('--fill_methods', default='', type=str, help='name of filled methods')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--save_path', default='../save_models_init/', help='path to save model checkpoint')
    parser.add_argument('--save_features_path', default='../save_features/', help='path to save updated features')
    parser.add_argument('--save_name', default=None, help='name to save model checkpoint')
    parser.add_argument('--load_model', default=None, help='checkpoint path to load model')
    parser.add_argument('--data', default='challenge', type=str, help='name of dataset')
    parser.add_argument('--model', default='mhagru', type=str, help='name of model')
    parser.add_argument('--task', default='outcome', type=str, help='name of task')
    parser.add_argument('--device', default='0', type=str, help='index of device')
    parser.add_argument('--distrain', action='store_true', default=False, help='disable train pipeline')
    parser.add_argument('--num_layers', default=1, type=int, help='num_layers for transformer and GRU')
    parser.add_argument('--freeze_layers', default=0, type=int, help='freeze_layers for transformer and GRU')
    parser.add_argument('--init_features', default='median', type=str, help='Method of initializing features')
    parser.add_argument('--sample_rate', default=1, type=float, help='sample dataset')
    parser.add_argument('--miss_rate', default=1, type=float, help='miss rate')
    parser.add_argument('--seed', default=2023, type=int, help='random seed')
    return parser.parse_args()


def get_loss(y_pred, y_true, task_name):
    if task_name == "outcome":
        loss = F.binary_cross_entropy(y_pred, y_true)
    elif task_name == "los":
        loss = F.mse_loss(y_pred, y_true)
    return loss


def adjust_mask_ratio(mask, target_ratio):
    current_ratio = np.mean(mask)  # 计算当前矩阵中1的比例
    if current_ratio >= target_ratio:
        return mask  # 如果当前比例已经达到或超过目标比例，则不需要调整
    num_to_convert = int(np.sum(mask) * (target_ratio - current_ratio) / current_ratio)  # 需要转换的0的数量
    zero_indices = np.where(mask == 0)  # 找到矩阵中值为0的位置
    random_indices = np.random.choice(len(zero_indices[0]), num_to_convert, replace=False)  # 随机选择需要转换的位置
    # 将选定位置的0转换为1
    mask[zero_indices[0][random_indices], zero_indices[1][random_indices]] = 1
    return mask


def train_step(model, optimizer, train_loader, device, fill, trainable_features=None, task_name='outcome'):
    model.train()
    # sum_values = []
    # miss_values = []
    # miss_rate = []
    for step, (train_x_nor, train_x_nor_mask, train_y, train_lens, train_pid) in enumerate(train_loader):
        optimizer.zero_grad()
        train_x_nor = train_x_nor.to(device)
        train_x_nor_mask = train_x_nor_mask.to(device)
        train_y = train_y.to(device)
        # print(train_y)

        if args.miss_rate != 1:
            train_x_nor_mask = train_x_nor_mask.cpu().detach().numpy()
            train_x_nor_mask = adjust_mask_ratio(train_x_nor_mask, args.miss_rate)
            train_x_nor_mask = torch.from_numpy(train_x_nor_mask).to(device)

        # sum_values.append(train_x_nor_mask.shape[0] * train_x_nor_mask.shape[1] * train_x_nor_mask.shape[2])
        # miss_values.append(torch.sum(train_x_nor_mask).cpu().detach().numpy())
        # miss_rate.append(torch.sum(train_x_nor_mask).cpu().detach().numpy() / (train_x_nor_mask.shape[0] * train_x_nor_mask.shape[1] * train_x_nor_mask.shape[2]))
        if fill:
            trainable_features_expand = trainable_features.expand(train_x_nor.shape)
            train_x_nor[train_x_nor_mask == 1] = trainable_features_expand[train_x_nor_mask == 1]
        else:
            if args.miss_rate != 1:
                trainable_features_expand = trainable_features.expand(train_x_nor.shape)
                train_x_nor[train_x_nor_mask == 1] = trainable_features_expand[train_x_nor_mask == 1]

        if model.backbone_name == 'concare':
            train_pred, decov_loss = model(train_x_nor, train_lens)
            loss = get_loss(train_pred, train_y, task_name) + 10 * decov_loss
        else:
            train_pred = model(train_x_nor, train_lens)
            loss = get_loss(train_pred, train_y, task_name)
        loss.backward()
        optimizer.step()
    # print(np.sum(miss_values), np.sum(sum_values), np.sum(miss_values) / np.sum(sum_values), np.max(miss_rate), np.min(miss_rate))


def val_step(model, val_loader, device, fill, trainable_features=None, task_name='outcome'):
    model.eval()
    # sum_values = []
    # miss_values = []
    # miss_rate = []
    with torch.no_grad():
        val_pred_list = []
        val_y_list = []
        for step, (val_x_nor, val_x_nor_mask, val_y, val_lens, val_pid) in enumerate(val_loader):
            val_x_nor = val_x_nor.to(device)
            val_x_nor_mask = val_x_nor_mask.to(device)
            val_y = val_y.to(device)
            # print(val_y)
            if args.miss_rate != 1:
                val_x_nor_mask = val_x_nor_mask.cpu().detach().numpy()
                val_x_nor_mask = adjust_mask_ratio(val_x_nor_mask, args.miss_rate)
                val_x_nor_mask = torch.from_numpy(val_x_nor_mask).to(device)

            # sum_values.append(val_x_nor_mask.shape[0] * val_x_nor_mask.shape[1] * val_x_nor_mask.shape[2])
            # miss_values.append(torch.sum(val_x_nor_mask).cpu().detach().numpy())
            # miss_rate.append(torch.sum(val_x_nor_mask).cpu().detach().numpy() / (val_x_nor_mask.shape[0] * val_x_nor_mask.shape[1] * val_x_nor_mask.shape[2]))
            if fill:
                trainable_features_expand = trainable_features.expand(val_x_nor.shape)
                val_x_nor[val_x_nor_mask == 1] = trainable_features_expand[val_x_nor_mask == 1]
            else:
                if args.miss_rate != 1:
                    trainable_features_expand = trainable_features.expand(val_x_nor.shape)
                    val_x_nor[val_x_nor_mask == 1] = trainable_features_expand[val_x_nor_mask == 1]

            if model.backbone_name == 'concare':
                val_pred, _ = model(val_x_nor, val_lens)
            else:
                val_pred = model(val_x_nor, val_lens)
            val_pred_list.append(val_pred)
            val_y_list.append(val_y)
        val_pred_list = torch.cat(val_pred_list)
        val_y_list = torch.cat(val_y_list)
        if task_name == 'outcome':
            val_performance = get_binary_metrics(val_pred_list, val_y_list)
        elif task_name == 'los':
            val_performance = get_regression_metrics(val_pred_list, val_y_list)
    # print(val_performance)
    # print(np.sum(miss_values), np.sum(sum_values), np.sum(miss_values) / np.sum(sum_values), np.max(miss_rate), np.min(miss_rate))
    return val_performance


args = parse_args()
set_seed(args.seed)


if args.data == 'cdsl':
    from dataset.cdslDataset import *
elif args.data == 'mimic':
    from dataset.mimicivDataset import *

train_dataset = EhrDataset(train_x, train_x_mask, train_y, train_pid, args.task, args.fill_methods, args.sample_rate)
val_dataset = EhrDataset(val_x, val_x_mask, val_y, val_pid, args.task, args.fill_methods)
test_dataset = EhrDataset(test_x, test_x_mask, test_y, test_pid, args.task, args.fill_methods)

print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=pad_collate, num_workers=8)
val_loader = data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, collate_fn=pad_collate, num_workers=8)
test_loader = data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=pad_collate, num_workers=8)

print(len(train_loader), len(val_loader), len(test_loader))

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() == True else 'cpu')
hidden_dim = {'mhagru': 32, 'transformer': 32, 'concare': 32, 'rnn':32, 'gru': 32, 'lstm': 32, 'retain': 32, 'safari': 32, 'm3care': 32}
model = DLModel(backbone_name=args.model, 
				task_name=args.task, 
				demo_dim=len(demographic_features), 
				input_dim=len(demographic_features)+len(labtest_features),
				hidden_dim=hidden_dim[args.model],
				output_dim=1,
                num_layers=args.num_layers).to(device)

if args.load_model:
    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint['net'])
if args.fill:
    if args.init_features == 'median':
        trainable_features = nn.Parameter(torch.tensor(default_fill[demographic_features + labtest_features].values, dtype=torch.float32), requires_grad=False).to(device)
    elif args.init_features == 'zero':
        trainable_features = np.zeros(default_fill[demographic_features + labtest_features].values.shape)
        trainable_features = nn.Parameter(torch.tensor(trainable_features, dtype=torch.float32), requires_grad=False).to(device)
    elif args.init_features == 'random':
        trainable_features = np.random.rand(*default_fill[demographic_features + labtest_features].values.shape)
        trainable_features = nn.Parameter(torch.tensor(trainable_features, dtype=torch.float32), requires_grad=False).to(device)
    trainable_features.requires_grad = True
    save_features = [trainable_features.cpu().detach().numpy()]
    params = [
        {'params': model.parameters(), 'lr': args.lr},
        {'params': [trainable_features], 'lr': args.fill_lr}
    ]
    optimizer = torch.optim.AdamW(params)
else:
    trainable_features = nn.Parameter(torch.tensor(default_fill[demographic_features + labtest_features].values, dtype=torch.float32), requires_grad=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

def outperformance(cur_performance, best_performance, task_name):
    if task_name == 'outcome':
        if cur_performance['auroc'] >= best_performance:
            return True
    if task_name == 'los':
        if cur_performance['mse'] <= best_performance:
            return True
    return False

def get_performance(cur_performance, task_name):
    if task_name == 'outcome':
        return cur_performance['auroc']
    if task_name == 'los':
        return cur_performance['mse']

if args.save_name is None:
    if args.fill:
        save_checkpoint_path = os.path.join(args.save_path, '{}-{}-{}-filling.pth'.format(args.data, args.model, args.task))
    else:
        save_checkpoint_path = os.path.join(args.save_path, '{}-{}-{}.pth'.format(args.data, args.model, args.task))
else:
    save_checkpoint_path = os.path.join(args.save_path, '{}.pth'.format(args.save_name))

best_epoch = 0
if not args.distrain:
    for epoch in range(args.epoch):
        if args.fill:
            train_step(model, optimizer, train_loader, device, args.fill, trainable_features, task_name=args.task)
            save_features.append(trainable_features.cpu().detach().numpy())
            val_performance = val_step(model, val_loader, device, args.fill, trainable_features, task_name=args.task)
            test_performance = val_step(model, test_loader, device, args.fill, trainable_features, task_name=args.task)
        else:
            train_step(model, optimizer, train_loader, device, args.fill, trainable_features, task_name=args.task)
            val_performance = val_step(model, val_loader, device, args.fill, trainable_features, task_name=args.task)
            test_performance = val_step(model, test_loader, device, args.fill, trainable_features, task_name=args.task)
        if epoch == 0:
            best_performance = get_performance(val_performance, args.task)
        if outperformance(val_performance, best_performance, args.task):
            print('#' * 10, epoch, '#' * 10)
            print(val_performance)
            print(test_performance)
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
                }
            if args.fill:
                state['trainable_features'] = trainable_features
            torch.save(state, save_checkpoint_path)
            best_performance = get_performance(val_performance, args.task)

            best_epoch = epoch

if args.distrain:
    save_checkpoint_path = args.load_model

checkpoint = torch.load(save_checkpoint_path)
model.load_state_dict(checkpoint['net'])
if args.fill:
    trainable_features = checkpoint['trainable_features']
    test_performance = val_step(model, test_loader, device, args.fill, trainable_features, task_name=args.task)
else:
    test_performance = val_step(model, test_loader, device, args.fill, trainable_features, task_name=args.task)

print(test_performance)
