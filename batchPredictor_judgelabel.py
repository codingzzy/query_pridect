# -*- coding: utf-8 -*-
import pdb
import sys
from parse import LRModel, load_model_list
from scipy.sparse import lil_matrix
import numpy as np
import time
from numpy.matlib import repmat


def get_judge_data_batch(filename, data_size, feature_size):
    fobj = open(filename)
    label_list = []
    features = []
    cnt = 0
    features = lil_matrix((data_size, feature_size))
    for line in fobj:
        if cnt%10000 == 0:
            print "loader process feature %d"%cnt
        line = line.strip()
        line_list = line.split(' ')
        label_list.append(int(line_list[0]))
        for i in range(1,len(line_list)):
            feaobj = line_list[i].split(":")
            features[cnt, int(feaobj[0])-1]= int(feaobj[1])
        cnt += 1
    return(label_list, features)


def linear_predict_batch(coeff_vec, intercept, query_vec):
    margin = query_vec.dot(coeff_vec)+intercept
    score = 1.0/(1.0+np.exp(-margin))
    return score




def gen_score_mat_batch(features, model_list):
    score_matrix = []
    cnt = 0
    start = time.time()
    for model in model_list:
        score = linear_predict_batch(model.coeff_vec, model.intercept, features)
        score_matrix.append(score)
    x = np.array(score_matrix)
    y = x.transpose()
    end = time.time()
    return y 

def multi_classifier_batch(score_matrix):
    x = np.argsort(score_matrix, kind='mergesort')
    y = np.sort(score_matrix, kind='mergesort')
    return (y,x)

def judgeTopK_batch(score_matrix, res_matrix, ground_label=[], k=12, threshold=0.5): 
    if (res_matrix.shape[0] != len(ground_label)) and len(ground_label) != 0:
        return -1
    pdb.set_trace()#待会查

    right_cnt_list = []
    topK_label = []
    for i in range(0,k):
        right_cnt_list.append(0)

    row_cnt , col_cnt = score_matrix.shape

    for i in range(0, score_matrix.shape[0]):
        flag = False
        idxr = -1
        tmp = []
        for j in range(0,k):
            if score_matrix[i, col_cnt - j - 1] < threshold:
                tmp.append(-1)
                break
            if len(ground_label) > 0:
                if (res_matrix[i, col_cnt - j - 1] == ground_label[i]):
                    flag = True
                    idxr = j
                    break
            tmp.append(res_matrix[i, col_cnt - j - 1])
        topK_label.append(tmp)

        if flag:
            for j in range(k-1,idxr-1, -1):
                right_cnt_list[j] += 1

    confuse_mat = np.zeros( (col_cnt,col_cnt) )

    if len(ground_label) != 0:
        for i in range(0, score_matrix.shape[0]):
            if score_matrix[i, col_cnt  - 1] < threshold:
                continue
            confuse_mat[ground_label[i], res_matrix[i, col_cnt  - 1]] += 1



    return right_cnt_list, confuse_mat, topK_label




if __name__ == "__main__":
    label_list, features = get_judge_data_batch('../gen_data/train.lp',int(sys.argv[1]), int(sys.argv[2]))
    model_path = "../queryBinClass/data2/"
    model_number = int(sys.argv[3])
    filelist = []
    for i in range(0,model_number):
        filelist.append(model_path+"pmmlModel_class"+str(i)+".pmml")#存放n个线性分类函数，n表示类别数
    model_list = load_model_list(filelist)#获得分类模型，包含向量和intercept
    score_mat = gen_score_mat_batch(features, model_list)
    score_mat, res_mat = multi_classifier_batch(score_mat)
    right_list, confuse_mat, topK_label = judgeTopK_batch(score_mat, res_mat, label_list, min(model_number, 5), 0.5)
    label_map = load_label_map('../gen_data/label.map')
    classes_map = load_classes_map('../origin_data/classes_map.txt')
    content_list = origin_content('../origin_data/query1_map.txt')
    output_predict_result('../gen_data/final_result.txt', topK_result, label_list, label_map, classes_map, content_list)

    print right_list 
    print len(label_list)
    label_dict = {}
    for label in label_list:
        if not label_dict.has_key(label):
            label_dict[label] = 0
        label_dict[label] += 1
    print label_dict
    print confuse_mat
    np.savetxt('../gen_data/test.mat',confuse_mat)

