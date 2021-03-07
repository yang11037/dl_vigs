import json
import random
import os

import cv2
import pandas as pd

img_root = "/home/lqy/桌面/sofa_test/"
with open("/home/lqy/桌面/img_info.json", "r") as f:
    info_list = json.load(f)

# statistic
# statistic = {}
# i = 0
# sum_ = 0
# for info in info_list:
#     i += 1
#     print(i)
#     tags = info["tags"]
#     j = 0
#     for tag in tags:
#         tag_splits = tag.split("&&")
#         if tag_splits[0] == "style":
#             j += 1
#             tag_name = tag_splits[-1]
#             if tag_name in statistic:
#                 statistic[tag_name] += 1
#             else:
#                 statistic[tag_name] = 0
#     if j > 1:
#         sum_ += 1
# print(sum_)
# print(statistic)

# 看某个tag_name在各个网站有多少
statistic = {"-m1-": 0, "-m2-": 0, "-m3-": 0, "-m4-": 0, "-m5-": 0, "-m6-": 0, "-m7-": 0, "-m8-": 0, "-m9-": 0, "-m10-": 0,}
for info in info_list:
    tags = info["tags"]
    filename = info["filename"]
    cur_key = None
    for key in statistic:
        if key in filename:
            cur_key = key
            break
    for tag in tags:
        if tag == "design&&recliner":
            if cur_key is not None:
                statistic[cur_key] += 1
print(statistic)

# make set
# positives = []
# negatives = []
# sum_ = 0
# # flag = 0
# for info in info_list:
#     tags = info["tags"]
#     i = [0, 0]
#     for tag in tags:
#         if tag == "design&&recliner":
#             i[0] = 1
#         if "design" in tag:
#             i[1] += 1
#     if i[1] == 1 and i[0] == 1:
#         positives.append(info["filename"])
#         # sum_ += 1
#         # if flag == 1:
#         #     continue
#         # print("unique")
#         # img = cv2.imread(img_root + info["filename"] + ".jpg")
#         # cv2.imshow("1", img)
#         # cv2.waitKey(0)
#         # flag = 1
#     # if i[0] == 0:
#         negatives.append(info["filename"])
#     #     if flag == 0:
#     #         continue
#     #     print("other")
#     #     img = cv2.imread(img_root + info["filename"] + ".jpg")
#     #     cv2.imshow("2", img)
#     #     cv2.waitKey(0)
#     #     flag = 0
# data_sum = len(positives)
# random.shuffle(positives)
# random.shuffle(negatives)
#
# train_num = int(0.8 * data_sum)
# test_num = int(0.1 * data_sum)
# train_names = []
# train_labels = []
# valid_names = []
# valid_labels = []
# test_names = []
# test_labels = []
#
# train_names.extend(positives[:train_num])
# train_labels.extend([1] * train_num)
# train_names.extend(negatives[:train_num])
# train_labels.extend([0] * train_num)
#
# valid_names.extend(positives[train_num:train_num+test_num])
# valid_labels.extend([1] * test_num)
# valid_names.extend(negatives[train_num:train_num+test_num])
# valid_labels.extend([0] * test_num)
#
# test_names.extend(positives[train_num+test_num:])
# test_labels.extend([1] * len(positives[train_num+test_num:]))
# test_names.extend(negatives[train_num+test_num:train_num + test_num * 2])
# test_labels.extend([0] * len(negatives[train_num+test_num:train_num + test_num * 2]))
#
# df = {}
# df["img_name"] = train_names
# df["label"] = train_labels
# df = pd.DataFrame(df)
# print(len(df))
# df.to_csv("data/train.csv", index=False)
#
# df = {}
# df["img_name"] = valid_names
# df["label"] = valid_labels
# df = pd.DataFrame(df)
# print(len(df))
# df.to_csv("data/valid.csv", index=False)
#
# df = {}
# df["img_name"] = test_names
# df["label"] = test_labels
# df = pd.DataFrame(df)
# print(len(df))
# df.to_csv("data/test.csv", index=False)

# 删除为空的图
# train_df = pd.read_csv("./data/train_new.csv")
# valid_df = pd.read_csv("./data/valid.csv")
# test_df = pd.read_csv("./data/test.csv")
#
# w_i = 0
# drop_list = []
# for i in range(len(train_df)):
#     print(i)
#     img_name = train_df.iloc[i]["img_name"] + ".jpg"
#     img_path = os.path.join(img_root, img_name)
#     img = cv2.imread(img_path)
#     if img is None:
#         w_i += 1
#         # drop_list.append(i)
# # train_df.drop(index=drop_list, inplace=True)
# # train_df.to_csv("./data/train_new.csv", index=False)
# print(w_i)


# 比较正负例图像
# positives = train_df[train_df["label"] == 1]["img_name"].tolist()
# negatives = train_df[train_df["label"] == 0]["img_name"].tolist()
# i = 1
# while i > 0:
#     index = int(i / 2)
#     if index >= len(negatives) or index >= len(positives):
#         break
#     if i % 2 == 0:
#         img_name = negatives[index]
#         img = cv2.imread(os.path.join(img_root, img_name+".jpg"))
#         cv2.imshow("0", img)
#     else:
#         img_name = positives[index]
#         img = cv2.imread(os.path.join(img_root, img_name+".jpg"))
#         cv2.imshow("1", img)
#     cv2.waitKey(0)
#     i += 1