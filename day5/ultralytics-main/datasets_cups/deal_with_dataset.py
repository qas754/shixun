import os
import shutil
from sklearn.model_selection import train_test_split
import random
import copy

# 设置数据文件夹路径
gray_dir = "./images"
label_dir = "./labels"

# 获取image和label文件夹中的所有文件名
gray_files = os.listdir(gray_dir)
label_files = os.listdir(label_dir)

# 确保image和label文件夹中的文件数量相同
assert len(label_files) == len(gray_files), "Number of image and labels files must be t he same!"

# 将文件名组合为一个列表
# label_files = copy.copy(gray_files)
# for i in range(len(label_files)):
#     label_files[i] = label_files[i].replace(".jpg", ".txt")

files = list(zip(gray_files, label_files))
random.shuffle(files)

# 划分数据为训练集和测试集（这里还包括验证集，但你可以根据需要调整比例）
train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)  # 假设30%为测试集
valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)  # 剩下的50%中，再取50%为验证集
print("测试集长度：" + str(len(test_files)))
print("训练集长度：" + str(len(train_files)))
print("验证集长度：" + str(len(valid_files)))

# 创建目录（如果它们不存在）
for split in ['train', 'test', 'val']:

    os.makedirs(os.path.join(gray_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# 移动文件到相应的目录
def move_files(file_list, split):
    for gray, lbl in file_list:

        shutil.move(os.path.join(gray_dir, gray), os.path.join(gray_dir, split, gray))
        shutil.move(os.path.join(label_dir, lbl), os.path.join(label_dir, split, lbl))

move_files(train_files, 'train')
move_files(valid_files, 'val')
move_files(test_files, 'test')

print("Data split completed!")
