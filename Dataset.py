# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2021/3/4 5:10 PM
@Author: Qinyang Lu
"""
import os
import cv2

import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

np.random.seed(0)


class ClsDataset(data.Dataset):
	def __init__(self, data_dir, data_df, mode, patch_size):
		self.data_dir = data_dir
		self.data_df = data_df
		self.mode = mode
		self.patch_size = patch_size

	def __getitem__(self, index):
		img_name = self.data_df.iloc[index]["img_name"] + ".jpg"
		img_path = os.path.join(self.data_dir, img_name)
		img = cv2.imread(img_path)
		img_crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# x_left = self.data_df.iloc[index]["x"]
		# y_top = self.data_df.iloc[index]["y"]
		# w = self.data_df.iloc[index]["w"]
		# h = self.data_df.iloc[index]["h"]
		#
		# img_crop = img[y_top:y_top + h, x_left:x_left + w, :]

		# 保证为方形
		rows = img_crop.shape[0]
		columns = img_crop.shape[1]
		if rows != columns:
			max_len = max(rows, columns)
			img_new = np.zeros((max_len, max_len, 3))
			img_new[:rows, :columns, 3] = img_crop
			img_crop = img_new

		transformer = transforms.Compose([transforms.ToTensor(),
										  transforms.Normalize
										  (mean=[0.485, 0.456, 0.406],
										   std=[0.229, 0.224, 0.225])])
		img_crop = transformer(img_crop)
		img_crop = torch.reshape(img_crop, (3,) + tuple(self.patch_size))

		if self.mode == "inference":
			return img_crop
		else:
			label = self.data_df.iloc[index]["label"]
			return img_crop, label

	def __len__(self):
		return len(self.data_df)


if __name__ == '__main__':
	data_df = pd.read_csv("./data.csv", sep=',')
	dataset = ClsDataset("./", data_df, "train", (224, 224))
	dataloader = data.DataLoader(dataset, batch_size=1)
	for img, label in dataloader:
		img = img.permute(0, 2, 3, 1)[0, :, :, :].numpy()
		img *= 255
		cv2.imshow("1", img)
		cv2.waitKey(0)