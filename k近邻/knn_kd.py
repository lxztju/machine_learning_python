# -*- coding:utf-8 -*-
# @time :2020/7/24
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


'''
实现一个KNN算法
构建kd树来搜索最近邻
----------------
距离的度量依然采用欧式距离
'''



import time
import logging

from collections import namedtuple
from math import sqrt


def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    # 存放数据及标记的list
    dataArr = [];
    labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        labelArr.append(int(curLine[0]))
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)

    # 返回data和label
    return dataArr, labelArr


# kd-tree每个结点中主要包含的数据结构如下
class KdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree:
    '''
    对于输入空间构建KD树
    '''
    def __init__(self, data):
        # data为list格式的数据
        k = len(data[0])  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集dataset创建KdNode
            if not data_set:  # 数据集为空
                return None
            # 对于输入的列表版找第split维进行排序
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # 找到中位数的索引
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates

            # 递归的创建kd树
            return KdNode(
                median,
                split,
                CreateNode(split_next, data_set[:split_pos]),  # 创建左子树
                CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点





class KnnKd():
    def __init__(self, kd, traindataArr, trainlabelArr):
        self.kd = kd
        # 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
        self.result = namedtuple("Result_tuple",
                        "nearest_point  nearest_dist  nodes_visited")
        self.data_label_dict = {''.join([str(j) for j in traindataArr[i]]): trainlabelArr[i] for i in range(len(trainlabelArr)) }



    def find_nearest(self, point):
        '''
        # 对构建好的kd树进行搜索，寻找与目标点最近的样本点：
        :param point: 待查找的某个节点
        :return: 返回对应的类别
        '''
        k = len(point)  # 数据维度

        def travel(kd_node, target, max_dist):
            '''
            递归在kd树中进行搜索，对应的point
            :param kd_node: kd树的节点
            :param target: 待查找的节点
            :param max_dist:  以待查找节点为圆心的超球的半径
            :return: 返回最终的numed_tuple
            '''
            if kd_node is None:
                return self.result([0] * k, float("inf"),
                              0)  # python中用float("inf")和float("-inf")表示正负无穷

            nodes_visited = 1

            s = kd_node.split  # 进行分割的维度
            pivot = kd_node.dom_elt  # 进行分割的“轴”

            if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
                nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
                further_node = kd_node.right  # 同时记录下右子树
            else:  # 目标离右子树更近
                nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
                further_node = kd_node.left

            temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

            nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
            dist = temp1.nearest_dist  # 更新最近距离

            nodes_visited += temp1.nodes_visited

            if dist < max_dist:
                max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

            temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
            if max_dist < temp_dist:  # 判断超球体是否与超平面相交
                return self.result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

            # ----------------------------------------------------------------------
            # 计算目标点与分割点的欧氏距离
            temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

            if temp_dist < dist:  # 如果“更近”
                nearest = pivot  # 更新最近点
                dist = temp_dist  # 更新最近距离
                max_dist = dist  # 更新超球体半径

            # 检查另一个子结点对应的区域是否有更近的点
            temp2 = travel(further_node, target, max_dist)

            nodes_visited += temp2.nodes_visited
            if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
                nearest = temp2.nearest_point  # 更新最近点
                dist = temp2.nearest_dist  # 更新最近距离

            return self.result(nearest, dist, nodes_visited)

        res =  travel(self.kd.root, point, float("inf"))  # 从根节点开始递归
        return self.data_label_dict[''.join([str(j)for j in res.nearest_point])]



if __name__ == '__main__':

    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='knnkd.log',
                        filemode='w')  # filemode默认为a，追加信息到日志文件，指定为‘w'，重新写入 文件，之前的文件信息丢失
    # 定义一个handler来将信息输出到控制台，StreamHandler与FileHandler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 设置在控制台输出格式[-
    formatter = logging.Formatter('%(asctime)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    #  将handler加入到根记录器
    logging.getLogger('').addHandler(console)

    # 根记录器输出信息
    logging.info('This is an info message.')


    start = time.time()


    # mnist数据集的存储位置
    import os
    home = os.path.expanduser('~')
    train_path = home + '/ML/mnist/mnist_train.csv'
    test_path = home + '/ML/mnist/mnist_train.csv'



    # 读取训练与测试集
    logging.info('Loading data....')

    traindataArr, trainlabelArr = loadData(train_path)
    testdataArr, testlabelArr = loadData(test_path)
    logging.info('Loading data done.')



    # 构建KD树
    logging.info('Building Kd Tree...')
    kd = KdTree(traindataArr)

    knnkd = KnnKd(kd, traindataArr, trainlabelArr)
    logging.info('Classify one image.....')

    print(knnkd.find_nearest(testdataArr[0]), testlabelArr[0])


    end = time.time()

    # logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end-start), 4))
