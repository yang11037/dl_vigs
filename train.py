import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data

from Dataset import ClsDataset

import pandas as pd

def train_cls(net, epochs, batch_size, lr, gpu, patch_size):
    data_root = "/home/lqy/桌面/sofa_test/"
    train_df = pd.DataFrame("./data/train.csv")
    valid_df = pd.DataFrame("./data/valid.csv")
    test_df = pd.DataFrame("./data/test.csv")
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

    for epoch in range(epochs):
        print('Starting epoch {}/{}'.format(epoch+1, epochs))
        net.train()

        epoch_loss = 0





if __name__ == '__main__':
    pretrained_path = "data/cp/resnet50-19c8e357.pth"
    patch_size = (224, 224)

    net = torchvision.models.resnet50(pretrained=False)
    net.fc = (2048, 2)

    net.load_state_dict(torch.load(pretrained_path))
    gpu = torch.cuda.is_available()
    if gpu:
        net.cuda()

