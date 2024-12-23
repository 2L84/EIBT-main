import argparse
import torch
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from model import Model
import math
import torchvision

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


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

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-NN")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Best-cifar10.pth",
        help="The base string of the pretrained model path",
    )
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

    # model setup and optimizer config
    model = Model(feature_dim, dataset, args.arch).cuda()
    model.load_state_dict(
        torch.load(
            args.model_path
        )
    )

    c = len(memory_data.classes)

    test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)

    print(test_acc_1)
    print(test_acc_5)
