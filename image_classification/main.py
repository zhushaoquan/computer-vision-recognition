from config import opt
import os
import models
from torch import nn
import torch
from utils.visualize import Visualizer
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataloder import IcvDataset
from dataset.transform_data import standard_data
from utils.hyper_parameter import *


# from torchvision import models

def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'label', 'score'])
        writer.writerows(results)


def idx2label(label_index_dict, value):
    for key, val in label_index_dict.items():
        if isinstance(value, list):
            print('error')
        else:
            if val == value:
                return key


@torch.no_grad()
def test():
    if opt.load_model_dir is not None:
        # step1 : load model
        model = getattr(models, opt.model)().eval()
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 10),
        )
        # 加载模型
        checkpoint = torch.load(opt.load_model_dir)
        model.load_state_dict(checkpoint["state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # step2: data
        test_data_list = standard_data(opt.train_data_dir, 'test')
        test_dataloader = DataLoader(IcvDataset(test_data_list, train=False, test=True), batch_size=1,
                                     shuffle=False,
                                     num_workers=opt.num_workers)
        results = []
        for i, (input, filepath) in enumerate(tqdm(test_dataloader)):
            input = input.to(device)
            output = model(input)
            # _, indices = torch.sort(output, descending=True)
            _, indices = torch.max(output, 1)
            probability = F.softmax(output, dim=1)[0] * 100
            # label = output.max(dim=1)[1].detach().tolist()
            # for idx in indices[0][0:1]:
            #     print(idx2label(opt.label_index_dict, idx.item()), probability[idx].item())
            output_label = idx2label(opt.label_index_dict, indices[0].item())
            out_score = probability[indices[0]].item()
            filepath = filepath[0]
            results.append([filepath, output_label, out_score])

        write_csv(results, opt.result_file)


@torch.no_grad()
def val(model, dataloader, criterion, device):
    model.eval()
    val_losses = AverageMeter()
    val_top1 = AverageMeter()
    for item, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        # loss and acc
        precious = accuracy(output, target, topk=(1,))
        val_losses.update(loss.item(), input.size(0))
        val_top1.update(precious[0].item(), input.size(0))
    model.train()
    return val_losses, val_top1


def train():
    vis = Visualizer(opt.env, port=opt.vis_port)
    # step1 : load model
    model = getattr(models, opt.model)(pretrained=True)
    # 加载预训练模型,微调或者特征提取
    model = init_extract_model(model, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # step2: data
    train_data_list = standard_data(opt.train_data_dir, 'train')
    val_data_list = standard_data(opt.train_data_dir, 'val')
    train_dataloader = DataLoader(IcvDataset(train_data_list), batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(IcvDataset(val_data_list, train=False), batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers)

    # step3: criterion and optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # 每100个epoch 下降 lr=lr*gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    # step4: define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()

    # step5.1: some parameters for K-fold and restart model
    start_epoch = 0
    best_top1 = 50

    # step5.2: restart the training process
    # PyTorch 保存断点checkpoints 的格式为 .tar文件扩展名格式
    if opt.resum_model_dir is not None:
        checkpoint = torch.load(opt.resum_model_dir)
        start_epoch = checkpoint["epoch"]
        best_top1 = checkpoint["best_top1"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["state_dict"])

    # 在恢复训练时，需要调用 model.train() 以确保所有网络层处于训练模式
    model.train()

    # step6 : train
    for epoch in range(start_epoch, opt.max_epoch):
        # lr 下降
        scheduler.step(epoch)
        lr = get_learning_rate(optimizer)
        train_losses.reset()
        train_top1.reset()
        for iter, (input, target) in enumerate(train_dataloader):
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # forword
            output = model(input)
            loss = criterion(output, target)
            precious = accuracy(output, target, topk=(1,))
            # loss and acc
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precious[0].item(), input.size(0))
            # backword
            loss.backward()
            optimizer.step()
        val_loss, val_top1 = val(model, val_dataloader, criterion, device)

        is_best = val_top1.avg > best_top1
        best_top1 = max(val_top1.avg, best_top1)

        print("epoch : {}/{}".format(epoch, opt.max_epoch))
        print("train-->loss:{},acc:{}".format(train_losses.avg, train_top1.avg))
        print("val-->loss:{},acc:{}".format(val_loss.avg, val_top1.avg))

        vis.plot_many({'train_loss':train_losses.avg,'val_loss':val_loss.avg})
        # vis.plot('train_loss', train_losses.avg)
        # vis.plot('val_accuracy', val_top1.avg)

        vis.log(
            "epoch:{epoch},lr:{lr},train_loss:{train_loss},val_loss:{val_loss},train_acc:{train_acc},val_acc:{val_acc}".format(
                epoch=epoch, train_loss=train_losses.avg, val_loss=str(val_loss.avg), train_acc=str(train_top1.avg),
                val_acc=str(val_top1.avg), lr=lr))

        if epoch % 10 == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "model": opt.model,
                "state_dict": model.state_dict(),
                "best_top1": best_top1,
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss.avg,
            }, opt.save_model_dir, is_best, epoch)


if __name__ == '__main__':
    train()
