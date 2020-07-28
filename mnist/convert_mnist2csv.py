# -*- coding:utf-8 -*-
# @time :2020/7/23
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

'''
将mnist数据集转换为csv格式便于查看
mnist数据集下载地址：http://yann.lecun.com/exdb/mnist/
下载解压之后，运行这个文件即可转换
-------
'''


def convert(img_path, label_path, target_path, n):
    img = open(img_path, "rb")
    target = open(target_path, "w")
    label = open(label_path, "rb")


    ## 开头要先让指针滑动一部分
    # 图像文件的前16个字节是头, 包含了4个字节的幻数, 4个字节表示图像数量
    # 4个字节表示单个图像的行数, 4个字节表示单个图像的列数.
    # 标记文件的前8个字节是头, 包含了4个字节的幻数, 4个字节表示标记数量
    img.read(16)
    label.read(8)
    images = []
    # s1 = label.read(1)
    # print(s1, ord(s1))
    for i in range(n):
        image = [ord(label.read(1))]
        for j in range(28*28):
            image.append(ord(img.read(1)))
        images.append(image)

    for image in images:
        target.write(",".join(str(pix) for pix in image)+"\n")

    img.close()
    target.close()
    label.close()

if __name__ == '__main__':
    import os
    home = os.path.expanduser('~')
    path = home + '/ML/mnist/'
    convert(path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte",
            path + "mnist_train.csv", 60000)
    convert(path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte",
            path + "mnist_test.csv", 10000)


    convert(path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte",
            path + "mnist_train_samples.csv", 200)
    convert(path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte",
            path + "mnist_test_samples.csv", 10)

    # import pandas as pd
    # test_data = pd.read_csv('./mnist_test.csv')
    # print(test_data.shape)
    # print(test_data.head())
