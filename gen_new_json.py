import json

with open("/home/lqy/桌面/img_info_cache.json", "r") as f:
    info_list = json.load(f)
i = 0
new_list = []
for info in info_list:
    i += 1
    print(i)
    if info["is_downloaded"] is True:
        new_list.append(info)

print(len(new_list))
with open("/home/lqy/桌面/img_info_valid.json", "w") as f:
    json.dump(new_list, f)