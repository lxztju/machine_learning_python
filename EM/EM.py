# -*- coding:utf-8 -*-
# @time :2020/8/3
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


'''
构建两个高斯分布来混合来模拟生成的数据

---------------------------------------
第一个高斯分布
alpha=0.3, mu=0, sigma=1

第二个高斯分布

alpha=0.7, mu=1, sigma=3
'''


import numpy as np
import logging
import time


def loadData(*args):
    '''
    传入一个参数的列表，然后模拟高斯混合产生数据
    :param args: 输入的列表分别为[alpha0, mu0, sigma0, alpha1, mu1, sigma1]
    :return:  返回高斯混合boing生成的数据
    '''
    print(args)
    alpha0, mu0, sigma0, alpha1, mu1, sigma1 = args[0]

    # 生成数据的长度
    length = 1000
    # 第一个高斯模型产生的数据
    data1 = np.random.normal(mu0, sigma0, int(length*alpha0))

    #第二个高斯模型产生的数据
    data2 = np.random.normal(mu1, sigma1, int(length*alpha1))
    #所有的数据接起来放在一起
    dataArr = np.append(data1, data2)
    # 打乱数据
    np.random.shuffle(dataArr)
    return dataArr



class EM:
    def __init__(self, alpha0, mu0, sigma0, alpha1, mu1, sigma1, dataArr):
        '''
        高斯混合模型的参数
        :param alpha0: 第一个模型的生成概率
        :param mu0: 第一个高斯模型的均值
        :param sigma0: 第一个高斯模型的标准差
        :param alpha1: 第二个模型的生成概率
        :param mu1:  第二个模型的均值
        :param sigma1: 第三个模型的标准差
        '''
        self.alpha0 = alpha0
        self.mu0  = mu0
        self.sigma0 = sigma0
        self.alpha1 = alpha1
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.dataArr = dataArr
        self.iter = 200
        self.train()


    def getGamma(self, mu, sigma):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp( -1 * ((self.dataArr - mu) * (self.dataArr - mu)) / (2 * sigma ** 2))

    def E_step(self):
        gamma0 = self.alpha0 * self.getGamma(self.mu0, self.sigma0)

        gamma1 = self.alpha1 * self.getGamma(self.mu1, self.sigma1)

        sum_ = gamma0 + gamma1
        return gamma0/sum_, gamma1/sum_


    def M_step(self):
        gamma0, gamma1 = self.E_step()
        # print(sum(gamma0))
        self.mu0 = sum(gamma0 * self.dataArr) / sum(gamma0)
        self.mu1 = sum(gamma1 * self.dataArr) / sum(gamma1)

        self.alpha0 = sum(gamma0) / self.dataArr.size
        self.alpha1 = sum(gamma1) / self.dataArr.size

        # print(self.alpha0, self.alpha1)
        self.sigma0 = np.sqrt(sum(gamma0 * (self.dataArr - self.mu0)*(self.dataArr - self.mu0) ) / sum(gamma0))
        self.sigma1 = np.sqrt(sum(gamma1 * (self.dataArr - self.mu1)*(self.dataArr - self.mu1) ) / sum(gamma1))



    def train(self):


        for i in range(self.iter):
            self.M_step()
            # print(self.alpha0, self.mu0 , self.sigma0)


if __name__ == '__main__':

    parameters = [0.3, 0, 1, 0.7, 1, 3]
    dataArr = loadData(parameters)
    # print(dataArr.shape)
    em = EM(0.5, 0, 1, 0.5, 1, 2, dataArr)
    print(em.alpha0, em.mu0, em.sigma0, em.alpha1, em.alpha1, em.mu1, em.sigma1)































