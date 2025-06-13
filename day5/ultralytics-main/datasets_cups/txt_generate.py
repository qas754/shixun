import os
# 最后一步:生成数据集的train.txt val.txt test.txt

def write_file_paths_to_txt(directory, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                f.write(file_path + '\n')

            # 调用函数，为每个文件夹写入文件路径

write_file_paths_to_txt(r'D:\python.learm\shixun\day5\ultralytics-main\datasets_cups\images\train', 'train.txt')
write_file_paths_to_txt(r'D:\python.learm\shixun\day5\ultralytics-main\datasets_cups\images\test', 'test.txt')
write_file_paths_to_txt(r'D:\python.learm\shixun\day5\ultralytics-main\datasets_cups\images\val', 'val.txt')
