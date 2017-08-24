# coding=utf-8
import csv
import os
import pickle
import cPickle
from math import ceil
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import operator

'''
函数说明：计算根节点的“信息熵”、以及其他输入数据的“信息熵”
输入：data（DataFrame类型变量，最后一列是 label，最后一列之前有多少列没有要求）
返回值：“信息熵”
'''
from math import log
def root_information_gain(data):
    # 计算每种label的数目，然后放在字典中
    label_dict = {} # 建立一个字典，里面的索引是“label做了unique后的数字”，对应的值是这种label的数目！
    for i in data['label'].unique():
        label_dict[i] = len(data[data['label'] == i])

    # 计算信息增益
    gain = 0.0 # 信息增益最终保存在这个变量里面
    length = len(data)  # 得到数据集长度，用于一会计算概率
    for i in label_dict.keys():
        prob = (label_dict[i] * 1.0)/length
        gain += -(prob * log(prob,2))
    return gain

'''
函数说明：计算输入数据集中，下一次该选择哪个特征来分裂数据（这个只可以处理 离散型特征）
输入：data（DataFrame类型变量，最后一列是 label，最后一列之前“全部”是N列特征列，我们就是要从这N列特征中选择下一次该选择哪个特征来分裂）
返回值：1个list变量，里面存有两个值，第一个是 选择好的特征名，第二个是 对应的信息增益
'''
def nextChooseWhichFeature_lisan(data):
    fea_list = list(data.columns)
    fea_list.remove('label') #存储所有的“特征列名”
    root_gain = root_information_gain(data) #根节点信息熵有多大
    # 定义两个变量，一个用于临时存储最好的特征的“信息增益值”，一个用于保存对应的“特征名”
    gain_best_gain = 0.0
    gain_best_name = ""

    # 用for循环计算每一个特征的信息熵
    for i in fea_list:
        # 用来保存当前特征 i 的“信息熵”
        gain = 0.0
        for j in data[i].unique():
            # 得到特征 == i的时候，其中一种取值为j的 训练样本数
            i_j = data[data[i] == j]
            prob = (len(i_j)*1.0) / len(data)
            gain += prob * root_information_gain(i_j)
        # 信息增益 = 根节点信息熵root_gain - gain
        info_gain = root_gain - gain
        # 更新gain_best_gain gain_bset_index

        # # print("fea = ",i," info_gain = ",info_gain)
        if(info_gain > gain_best_gain):
            gain_best_gain = info_gain
            gain_best_name = i

    return [gain_best_name,gain_best_gain]


'''
函数说明：计算输入数据集中，下一次该选择哪个特征来分裂数据（这个只可以处理 连续性特征）
输入：data（DataFrame类型变量，最后一列是 label，最后一列之前“全部”是N列特征列，我们就是要从这N列特征中选择下一次该选择哪个特征来分裂）
返回值：最好的特征“name”
'''
def nextChooseWhichFeature_lianxu(data):
    fea_list = list(data.columns)
    fea_list.remove('label') #存储所有的“特征列名”
    root_gain = root_information_gain(data) #根节点信息熵有多大
    # 一共有多少个样本
    how_many = len(data)
    # 定义一个dict变量，里面的每一个元素是 (特征，这个特征对应的信息增益)
    fea_gain = {}

    # 用for循环计算每一个特征的信息熵
    for i in fea_list:
        # gain 用来保存当前特征 i 的“信息增益”；yuzhi 用来保存特征 i 对应最佳 信息增益 时候的 阈值
        gain = 0.0
        yuzhi = 0.0
        # 将特征i列，进行从小到大排序，将排序后的值放在一个list中
        i_paixu = list(data[i])
        i_paixu.sort()
        # 找出来 i_paixu 中每两个值间的“中位数”
        i_paixu_minus_1 = []
        for ii in range(0,len(i_paixu)-1):
            i_paixu_minus_1.append( (i_paixu[ii] + i_paixu[ii+1])/2.0 )
        #“连续特征”和“离散特征”的计算信息增益的方式不一样
        for j in i_paixu_minus_1:
            # 先找出来大于等于 j 的 “一批样本”
            bigger_than_j = data[data[i] >= j]
            if(len(bigger_than_j) == 0):
                gain_pos = 0
                bigger_than_j_pos = bigger_than_j
                bigger_than_j_neg = bigger_than_j
            else:
                # 那批样本里，正负样本各有多少
                bigger_than_j_pos = bigger_than_j[bigger_than_j['label'] == 1]
                bigger_than_j_neg = bigger_than_j[bigger_than_j['label'] == 0]
                # 计算出来 信息熵（这块有除法prob_1 prob_2的计算，但是不存在 除数小于0的情况）
                prob_1 = len(bigger_than_j_pos) * 1.0 / (len(bigger_than_j_pos) + len(bigger_than_j_neg))
                prob_2 = len(bigger_than_j_neg) * 1.0 / (len(bigger_than_j_pos) + len(bigger_than_j_neg))
                if(prob_1 == 0 and prob_2 != 0):
                    gain_pos = -1 * (prob_2 * log(prob_2, 2))
                elif(prob_1 != 0 and prob_2 == 0):
                    gain_pos = -1 * (prob_1 * log(prob_1, 2))
                else:
                    gain_pos = -1 * (prob_1 * log(prob_1, 2) + prob_2 * log(prob_2, 2))

            # 再同上面的逻辑，处理出来 小于 j 的“一批样本”
            smaller_than_j = data[data[i] < j]
            if(len(smaller_than_j) == 0):
                gain_neg = 0
                smaller_than_j_pos = smaller_than_j
                smaller_than_j_neg = smaller_than_j
            else:
                smaller_than_j_pos = smaller_than_j[smaller_than_j['label'] == 1]
                smaller_than_j_neg = smaller_than_j[smaller_than_j['label'] == 0]
                prob_1 = len(smaller_than_j_pos) * 1.0 / (len(smaller_than_j_pos) + len(smaller_than_j_neg))
                prob_2 = len(smaller_than_j_neg) * 1.0 / (len(smaller_than_j_pos) + len(smaller_than_j_neg))
                if (prob_1 == 0 and prob_2 != 0):
                    gain_neg = -1 * (prob_2 * log(prob_2, 2))
                elif (prob_1 != 0 and prob_2 == 0):
                    gain_neg = -1 * (prob_1 * log(prob_1, 2))
                else:
                    gain_neg = -1 * (prob_1 * log(prob_1, 2) + prob_2 * log(prob_2, 2))

            # 接下来可以用公式计算出来 这个特征造成的“信息增益”
            a = (len(bigger_than_j_pos) + len(bigger_than_j_neg)) * 1.0 / how_many * gain_pos
            b = (len(smaller_than_j_pos) + len(smaller_than_j_neg)) * 1.0 / how_many * gain_neg
            gain_pos_and_neg = root_gain - ( a  +  b )

            # # print("fea = ",i,"yuzhi = ",j, "gain_pos_and_neg = ",gain_pos_and_neg)
            # 如果j阈值计算出来的“信息增益”大于 gain，就交换两个值。for j in i_paixu_minus_1: 这个for循环结束后，gain 中 存储的是特征i最好的一个信息增益
            if(gain_pos_and_neg > gain):
                gain = gain_pos_and_neg
                yuzhi = j
        fea_gain[i] = (gain,yuzhi)

    fea_gain_sorted = sorted(fea_gain.iteritems(), key=operator.itemgetter(1), reverse=True)

    return fea_gain_sorted[0]



'''
函数说明：分裂结束的时候，程序遍历完所有划分数据集的属性，但是每一个分支下的所有实例如果不具有同一个“标签”，就得用“投票选择”。下面这个函数就是这个
输入：data（list类型，每一个索引对应的值是 这个实例的分类）
返回值：“投票选择”思想，投出来的“分类”
'''
def vote(data):
    label_fenlei = {} #定义一个字典，索引是每一个分类，对应的值是每一个分类的数量
    # 初始化那个label_fenlei
    for i in data:
        if i not in label_fenlei.keys():
            label_fenlei[i] = 0
        label_fenlei[i] += 1
    # 返回“分类数量”最多的一个“分类”
    label_fenlei_sorted = sorted(label_fenlei.iteritems(), key=operator.itemgetter(1), reverse=True)
    return label_fenlei_sorted[0][0]


'''
函数说明：创建一棵树，具体的分裂信息 用 一个字典嵌套一个字典 表示出来
输入：data（DataFrame类型变量，最后一列是 label，最后一列之前“全部”是N列特征列）
返回值：一个字典（表示树的分裂情况）
补充说明：分裂停止的条件 1：要么一个结点所有的实例都属于了一个分类 2：要么就是达到了预先设定好的树深（或者 遍历完了所有划分数据集的属性）
'''
def create_tree(data):
    # 如果“所有样本”的类别都相同就停止划分。并且返回唯一的“label”
    if(len(data['label'].unique()) == 1):
        return list(data['label'].unique())[0]
    # 如果在递归过程中，data中只剩下一个特征，那就用“投票”的思想，返回多数占的label
    if(len(list(data.columns)) == 2):
        major_label = vote(data[list(data.columns)[0]])
        return major_label

    # 接下来的难点在于如何从 所有特征中（连续、离散）选择出来最好的一个特征来接着进行分裂
    # 根据 data中每一列数据的“dtype”来判断 这个特征是什么类型的特征。。
    # 定义两个变量，一个里面存储 data中所有的“离散特征”，一个里面存储 data中所有的“离散特征”
    dispersed_feas = []
    continuity_feas = []
    col = list(data.columns)
    col.remove('label')
    for i in col:
        if (data[i].dtype == 'float64'):
            continuity_feas.append(i)
        else:
            dispersed_feas.append(i)
    dispersed_feas.append('label')
    continuity_feas.append('label')

    '''当把所有的 特征列 分开到 dispersed_feas continuity_feas 中后，还要检查他俩的长度，以防哪个种类的特征已经没有了，造成函数输入为Null，造成崩溃'''
    if( len(dispersed_feas) == 1 and len(continuity_feas) > 1 ):
        b = nextChooseWhichFeature_lianxu(data[continuity_feas])
        best_fea_name = b[0]
    elif( len(dispersed_feas) > 1 and len(continuity_feas) == 1 ):
        a = nextChooseWhichFeature_lisan(data[dispersed_feas])
        best_fea_name = a[0]
    else:
        # 将离散型特征 组成的数据 输入到“它对应的特征选择函数”中，连续特征的数据同样
        a = nextChooseWhichFeature_lisan(data[dispersed_feas])  # b是一个这样子的变量：  [gain_best_name,gain_best_gain]
        b = nextChooseWhichFeature_lianxu(data[continuity_feas])  # a是一个这样子的变量：  ('tiandu', (25,0.36))

        if(a[1] > b[1][0]):
            best_fea_name = a[0]
        else:
            best_fea_name = b[0]

    # 定义一个字典，里面的“索引是”特征名，里面存储的也是我们的“树结构”
    tree = {best_fea_name : {}}

    '''选择好了best_fea_name后，首先看它是什么类型的变量，再用不同的逻辑进行分裂'''
    if (data[best_fea_name].dtype == 'float64'):
        # 如果特征是“连续性特征”的话，我们就将数据分成两部分，依次处理
        data_small_than_j = data[data[best_fea_name] < b[1][1]]
        del data_small_than_j[best_fea_name]
        tree[best_fea_name]['small_'+str(b[1][1])] = create_tree(data_small_than_j)

        data_big_than_j = data[data[best_fea_name] >= b[1][1]]
        del data_big_than_j[best_fea_name]
        tree[best_fea_name]['big_'+str(b[1][1])] = create_tree(data_big_than_j)
    else:
        # 当它是“离散型特征”的时候
        # 首先看这个特征有几个unique()，数据集就会被分割成几部分。其中每一部分都可以接着进行分裂，所以只要把 每一部分的数据做好格式
        # 就可以输入到 create_tree 这个函数中，进行递归运算
        for i in data[best_fea_name].unique():
            # 做上面说的 几部分 数据 中的一部分！
            part_data = data[data[best_fea_name] == i] # 先找出来 data中 best_fea_name == i的数据
            del part_data[best_fea_name] # 再删除 best_fea_name 这列，因为这列的取值完全一样，不用再进行分裂
            tree[best_fea_name][i] = create_tree(part_data)

    # 最后将 tree 这个结构返回即可
    return tree


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('test.txt')
    del data['id']

    a = create_tree(data)
    print(a)
















































