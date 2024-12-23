import argparse
import os
import pandas as pd
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

import utils
from model import Model
import math

import torchvision


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["lr"] = lr * args.lr


def train(args, epoch, net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = (
        0.0,
        0,
        tqdm(data_loader),
    )
    for step, data_tuple in enumerate(train_bar, start=epoch * len(train_bar)):
        if args.lr_shed == "cosine":
            adjust_learning_rate(args, train_optimizer, data_loader, step)
        (pos_1, pos_2), _ = data_tuple
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _, out_1, drop_1 = net(pos_1)
        _, out_2, drop_2 = net(pos_2)

        out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
        out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)
        Drop_1_norm = (drop_1 - drop_1.mean(dim=0)) / drop_1.std(dim=0)
        Drop_2_norm = (drop_2 - drop_2.mean(dim=0)) / drop_2.std(dim=0)

        c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_o = on_diag + lmbda * off_diag

        c_D1 = torch.matmul(out_1_norm.T, Drop_1_norm) / batch_size
        on_diag = torch.diagonal(c_D1).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_D1).pow_(2).sum()
        loss_d1 = on_diag + lmbda * off_diag

        c_D2 = torch.matmul(out_2_norm.T, Drop_2_norm) / batch_size
        on_diag = torch.diagonal(c_D2).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_D2).pow_(2).sum()
        loss_d2 = on_diag + lmbda * off_diag

        loss_f = (loss_o + loss_d1 + loss_d2) / 3


        loss = loss_f
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description(
            "Train Epoch: [{}/{}] lr: {:.3f}x10-3 Loss: {:.4f} lmbda:{:.4f} bsz:{} f_dim:{} dataset: {}".format(
                epoch,
                epochs,
                train_optimizer.param_groups[0]["lr"] * 1000,
                total_loss / total_num,
                lmbda,
                batch_size,
                feature_dim,
                dataset,
            )
        )
    return total_loss / total_num


def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc="Feature extracting"):
            (data, _), target = data_tuple
            target_bank.append(target)
            feature, out, drop = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out, drop = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(
                feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices
            )
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0
            )
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1),
                dim=1,
            )

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                    epoch,
                    epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100,
                )
            )
    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        help="Dataset: cifar10, cifar100, tiny_imagenet, stl10",
        choices=["cifar10", "cifar100", "tiny_imagenet", "stl10"],
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        help="Backbone architecture",
        choices=["resnet50", "resnet18"],
    )
    parser.add_argument(
        "--feature_dim", default=1024, type=int, help="Feature dim for embedding vector"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="The base string of the pretrained model path",
    )
    parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
        help="Temperature used in softmax (kNN evaluation)",
    )
    parser.add_argument(
        "--k",
        default=200,
        type=int,
        help="Top k most similar images used to predict the label",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="Base learning rate")
    parser.add_argument(
        "--lr_shed",
        default="cosine",
        choices=["step", "cosine"],
        type=str,
        help="Learning rate scheduler: step / cosine",
    )

    parser.add_argument(
        "--lmbda",
        default=0.0051,
        type=float,
        help="Lambda that controls the on- and off-diagonal terms",
    )

    # GPU id (just for record)
    parser.add_argument("--gpu", dest="gpu", type=int, default=0)

    args = parser.parse_args()

    dataset = args.dataset
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    lmbda = args.lmbda

    if dataset == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root="~/cifar10",
            train=True,
            transform=utils.CifarPairTransform(train_transform=True),
            download=True,
        )
        memory_data = torchvision.datasets.CIFAR10(
            root="~/cifar10",
            train=True,
            transform=utils.CifarPairTransform(train_transform=False),
            download=True,
        )
        test_data = torchvision.datasets.CIFAR10(
            root="~/cifar10",
            train=False,
            transform=utils.CifarPairTransform(train_transform=False),
            download=True,
        )
    elif dataset == "cifar100":
        train_data = torchvision.datasets.CIFAR100(
            root="~/cifar100",
            train=True,
            transform=utils.CifarPairTransform(train_transform=True),
            download=True,
        )
        memory_data = torchvision.datasets.CIFAR100(
            root="~/cifar100",
            train=True,
            transform=utils.CifarPairTransform(train_transform=False),
            download=True,
        )
        test_data = torchvision.datasets.CIFAR100(
            root="~/cifar100",
            train=False,
            transform=utils.CifarPairTransform(train_transform=False),
            download=True,
        )
    elif dataset == "stl10":
        train_data = torchvision.datasets.STL10(
            root="~/stl10",
            split="train+unlabeled",
            transform=utils.StlPairTransform(train_transform=True),
            download=True,
        )
        memory_data = torchvision.datasets.STL10(
            root="~/stl10",
            split="train",
            transform=utils.StlPairTransform(train_transform=False),
            download=True,
        )
        test_data = torchvision.datasets.STL10(
            root="~/stl10",
            split="test",
            transform=utils.StlPairTransform(train_transform=False),
            download=True,
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True
    )

    model = Model(feature_dim, dataset, args.arch).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    if args.lr_shed == "step":
        m = [args.epochs - a for a in [50, 25]]
        scheduler = MultiStepLR(optimizer, milestones=m, gamma=0.2)
    c = len(memory_data.classes)

    results = {"train_loss": [], "test_acc@1": [], "test_acc@5": []}
    save_name_pre = "{}_{}_{}_{}".format(
        lmbda, feature_dim, batch_size, dataset
    )
    run_id_dir = os.path.join(
        args.save_path, dataset
    )
    if not os.path.exists(run_id_dir):
        print("Creating directory {}".format(run_id_dir))
        os.mkdir(run_id_dir)

    for epoch in range(1, epochs + 1):
        train_loss = train(
            args, epoch, model, train_loader, optimizer
        )
        if args.lr_shed == "step":
            scheduler.step()
            
        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                "{}/{}_model_{}.pth".format(
                    run_id_dir,save_name_pre, epoch
                ),
            )
    # wandb.finish()
