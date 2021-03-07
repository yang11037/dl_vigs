import os

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data

from Dataset import ClsDataset
from eval import eval_cls

import pandas as pd


def train_cls(net, epochs, batch_size, lr, gpu, patch_size):
    data_root = "/home/lqy/桌面/sofa_test/"
    dir_checkpoint = "data/cp/resnet/"
    if os.path.exists(dir_checkpoint) == 0:
        os.mkdir(dir_checkpoint)
    train_df = pd.read_csv("./data/train_new.csv")
    valid_df = pd.read_csv("./data/valid.csv")
    test_df = pd.read_csv("./data/test.csv")
    train_dataset = ClsDataset(data_root, train_df, "train", patch_size)
    valid_dataset = ClsDataset(data_root, valid_df, "train", patch_size)
    test_dataset = ClsDataset(data_root, test_df, "train", patch_size)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    min_loss = 100

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_df), len(valid_df), str(gpu)))

    for epoch in range(epochs):
        print('Starting epoch {}/{}'.format(epoch+1, epochs))
        net.train()

        epoch_loss = 0
        i = 0
        for i, (imgs, labels) in enumerate(train_loader):
            if gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()

                logits = net(imgs)
                labels = labels.view(-1)

                loss = criterion(logits, labels.long())
                print("\tbatch {}, positive {}, loss: {}".format(i, torch.sum(labels), loss.item()))

                epoch_loss = epoch_loss + loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scheduler.step(epoch_loss)
        print('Epoch finished ! Loss: {}'. \
              format(epoch_loss / (i + 1)))

        acc, recall, precision, specificity, auc, th, losses = \
            eval_cls(net, valid_loader, gpu)
        print('acc: {}, recall: {}, precision: {}, specificity: {}, auc: {}, th: {}, loss: {}'. \
              format(acc, recall, precision, specificity, auc, th, losses))

        if losses < min_loss:
            torch.save(net,
                       dir_checkpoint + 'CP{}.pt'.format(epoch + 1))
            print('Checkoutpoint {} saved !\n'.format(epoch + 1))
            min_loss = losses

        acc, recall, precision, specificity, auc, th, losses = \
            eval_cls(net, test_loader, gpu)
        print("test set:")
        print('acc: {}, recall: {}, precision: {}, specificity: {}, auc: {}, th: {}, loss: {}'. \
              format(acc, recall, precision, specificity, auc, th, losses))
        print()


if __name__ == '__main__':
    pretrained_path = "data/cp/resnet50-19c8e357.pth"
    patch_size = (224, 224)
    epochs = 100
    batch_size = 48
    lr = 0.001
    classes = 2

    net = torchvision.models.resnet50(pretrained=False)
    net.load_state_dict(torch.load(pretrained_path))
    net.fc = nn.Linear(2048, classes)

    gpu = torch.cuda.is_available()
    if gpu:
        net.cuda()

    train_cls(net=net,
              epochs=epochs,
              batch_size=batch_size,
              lr=lr,
              gpu=gpu,
              patch_size=patch_size)