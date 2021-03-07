# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2021/3/4 5:10 PM
@Author: Qinyang Lu
"""
import os
import cv2
import logging

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
		try:
			img_crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except Exception as e:
			logging.exception(e)
			print(img_path)
			exit()

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
			img_new[:rows, :columns, :] = img_crop
			img_crop = img_new
		img_crop = cv2.resize(img_crop, self.patch_size)

		transformer = transforms.Compose([transforms.ToTensor()])#,
										  # transforms.Normalize
										  # (mean=[0.485, 0.456, 0.406],
										  #  std=[0.229, 0.224, 0.225])])
		img_crop = transformer(img_crop)
		img_crop = img_crop.float()

		if self.mode == "inference":
			return img_crop
		else:
			label = self.data_df.iloc[index]["label"]
			return img_crop, label

	def __len__(self):
		return len(self.data_df)


if __name__ == '__main__':
	# data_df = pd.read_csv("./data/train.csv", sep=',')
	# dataset = ClsDataset("/home/lqy/桌面/sofa_test/", data_df, "train", (224, 224))
	# dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)
	# for i, (img, label) in enumerate(dataloader):
	# 	print(i)
	# 	print(img.shape)
	# 	print(label)
	# 	exit()

	img = cv2.imread("/home/lqy/桌面/sofa_test/antarius-44-square-arm-loveseat-m1-w003217065_0.jpg")
	print(img)