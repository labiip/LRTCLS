#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:46:25 2021

@author: Jiani Ma
"""
import os
os.chdir("/home/jiani.ma/LRTCLS/supplementary/")
os.getcwd()
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from scipy import interp
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import roc_curve, auc
import argparse
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from sklearn.model_selection import KFold
from numpy import linalg as LA
import tensorly as tl
np.set_printoptions(suppress = True)
np.random.seed(0)
# from hottbox.core import Tensor
# from hottbox.algorithms.decomposition import HOOI
from tensorly.decomposition import tucker

def tensorAndmatrix(mat1, mat2, mat3, mat4, v, m, n):
    tensor = np.zeros((v,m,n))
    tensor[0] = mat1
    tensor[1] = mat2
    tensor[2] = mat3
    tensor[3] = mat4
    tensor[np.isnan(tensor)]=0
    unfolding_mat = np.hstack((mat1,mat2,mat3,mat4))    
    unfolding_mat[np.isnan(unfolding_mat)] = 0
    return tensor, unfolding_mat 

def unfolding2slice(mat, mask, v, n):
    mat1 = mat[:,0:n]
    mat2 = mat[:,n:2*n]
    mat3 = mat[:,2*n:3*n]
    mat4 = mat[:,3*n:4*n]
    mask1 = mask[:,0:n]
    mask2 = mask[:,n:2*n]
    mask3 = mask[:,2*n:3*n]
    mask4 = mask[:,3*n:4*n]
    return mat1, mat2, mat3, mat4, mask1, mask2, mask3, mask4

def W_update(Z,Y,pai,Z_mask,rol):
    coef1 = 1/rol 
    coef2= 2/(2+rol)
    W_part = 2 * coef1 * np.multiply(Z_mask,Z) + coef1 *pai + Y 
    W = W_part - coef2 * np.multiply(Z_mask,W_part)
    return W

def X_tensor_update(Y_tensor, theta_tensor, rank_arr, rol): 
    ndarr = Y_tensor - theta_tensor/rol
    Y_theta_tensor = tl.tensor(ndarr)
    result = tucker(Y_theta_tensor, rank=rank_arr, init='random', tol=10e-5)
    core = result.core
    factors = result.factors 
    X_tmp1 = tl.tenalg.mode_dot(core,factors[0],mode = 0)
    X_tmp2 = tl.tenalg.mode_dot(X_tmp1,factors[1],mode=1) 
    X_tensor = tl.tenalg.mode_dot(X_tmp2, factors[2],mode=2)
    return X_tensor 

def SnLaplacianMatrix(X,n): 
    W = np.zeros((n,n))
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = np.dot(X[:,i],X[:,j])
    d = np.sum(W,axis = 1)   #row sum 
    D = np.diag(d) 
    # normlized_d = 1.0/np.sqrt(d)
    # normlized_D = np.diag(normlized_d)
    snL = D - W 
    # snL_tmp = np.dot(normlized_D, L) 
    # snL = np.dot(snL_tmp, normlized_D)
    return snL 

def Y_update(theta, pai, X, W, snL, rol, alpha, n): 
    Y_part_one = theta - pai + rol*X +rol*W
    Y_part_two_tmp = alpha*(snL.T + snL) + 2* rol * np.identity(n)
    Y_part_two = LA.inv(Y_part_two_tmp)
    Y = np.dot(Y_part_one,Y_part_two)
    return Y 

def theta_update(X,Y,last_theta,delta): 
    theta = last_theta + delta*(X-Y)
    return theta

def pai_update(Y,W,last_pai,delta):
    pai = last_pai + delta*(Y-W)
    return pai

def convergence(X,Y,W):
    lossXY = LA.norm(X-Y)
    lossYW = LA.norm(X-W)
    return lossXY, lossYW

def Matrix2Vector(mat):
    vector = mat.flatten()
    vector[np.isnan(vector)] = 0
    return vector 


if __name__=="__main__":
    m6A_disease_circrna = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=0).values  #131*1338
    m6A_disease_mirna = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=1).values  #131*1338
    m6A_disease_rbp = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=2).values  #131*1338
    m6A_disease_splice = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=3).values  #131*1338
    m6A_disease_circrna_vector = Matrix2Vector(m6A_disease_circrna)
    m6A_disease_mirna_vector = Matrix2Vector(m6A_disease_mirna)
    m6A_disease_rbp_vector = Matrix2Vector(m6A_disease_rbp)
    m6A_disease_splice_vector = Matrix2Vector(m6A_disease_splice)
    m = m6A_disease_circrna.shape[0]        #131
    n = m6A_disease_circrna.shape[1]        #1338
    v = 4
    alpha = 4    
    rol = 8
    delta = 0.8
    rank_arr = [3,60,400]
    threshold1 = 1
    fake_m6a_disease = m6A_disease_circrna
    fake_m6a_disease[np.isnan(fake_m6a_disease)] = 0
    fake_m6a_disease = np.abs(fake_m6a_disease)    
    fake_m6a_disease_vector = fake_m6a_disease.flatten()    
    known_samples_index = np.where(fake_m6a_disease_vector == 1)[0]   #是fake_m6a_disease_vector中已知元素的索引
    kf = KFold(n_splits =10, shuffle=True)      #10 fold
    train_all=[]    
    test_all=[]     
    for train_ind,test_ind in kf.split(known_samples_index):  
        train_all.append(train_ind)    #known_samples_index的索引
        test_all.append(test_ind)      #known_samples_index的索引
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_int in range(10):
        print('fold_int:',fold_int)
        train_id = train_all[fold_int]
        train_index = known_samples_index[train_id]
        train_mask_vector = np.zeros(len(fake_m6a_disease_vector))
        train_mask_vector[train_index] = 1   #to mask the train element
        circrna_train_vector = np.multiply(m6A_disease_circrna_vector,train_mask_vector)
        mirna_train_vector = np.multiply(m6A_disease_mirna_vector,train_mask_vector)
        rbp_train_vector = np.multiply(m6A_disease_rbp_vector,train_mask_vector)
        splice_train_vector = np.multiply(m6A_disease_splice_vector,train_mask_vector)
        Z_v1 = circrna_train_vector.reshape(m,n)
        Z_v2 = mirna_train_vector.reshape(m,n)
        Z_v3 = rbp_train_vector.reshape(m,n)
        Z_v4 = splice_train_vector.reshape(m,n)
        Z1_mask = Z2_mask = Z3_mask = Z4_mask = train_mask_vector.reshape(m,n)
        last_W_tensor = last_Y_tensor = last_X_tensor = last_theta_tensor = last_pai_tensor = np.random.random((v,m,n))
        current_W_tensor = current_Y_tensor = current_X_tensor = current_theta_tensor = current_pai_tensor = np.zeros((v,m,n))
        snL_tensor = np.zeros((v,n,n))
        for iteration in range(300):
            current_W_tensor[0] = W_update(Z_v1,last_Y_tensor[0],last_pai_tensor[0],Z1_mask,rol)
            current_W_tensor[1] = W_update(Z_v2,last_Y_tensor[1],last_pai_tensor[1],Z2_mask,rol)
            current_W_tensor[2] = W_update(Z_v3,last_Y_tensor[2],last_pai_tensor[2],Z3_mask,rol)
            current_W_tensor[3] = W_update(Z_v4,last_Y_tensor[3],last_pai_tensor[3],Z4_mask,rol)
            snL_tensor[0] = SnLaplacianMatrix(last_X_tensor[0],n)
            snL_tensor[1] = SnLaplacianMatrix(last_X_tensor[1],n)
            snL_tensor[2] = SnLaplacianMatrix(last_X_tensor[2],n)
            snL_tensor[3] = SnLaplacianMatrix(last_X_tensor[3],n)
            current_Y_tensor[0] = Y_update(last_theta_tensor[0], last_pai_tensor[0], last_X_tensor[0], current_W_tensor[0], snL_tensor[0], rol, alpha, n)
            current_Y_tensor[1] = Y_update(last_theta_tensor[1], last_pai_tensor[1], last_X_tensor[1], current_W_tensor[1], snL_tensor[1], rol, alpha, n)
            current_Y_tensor[2] = Y_update(last_theta_tensor[2], last_pai_tensor[2], last_X_tensor[2], current_W_tensor[2], snL_tensor[2], rol, alpha, n)
            current_Y_tensor[3] = Y_update(last_theta_tensor[3], last_pai_tensor[3], last_X_tensor[3], current_W_tensor[3], snL_tensor[3], rol, alpha, n)
            current_X_tensor = X_tensor_update(current_Y_tensor, last_theta_tensor, rank_arr, rol)
            current_theta_tensor = theta_update(current_X_tensor,current_Y_tensor,last_theta_tensor,delta)
            current_pai_tensor = pai_update(current_Y_tensor,current_W_tensor,last_pai_tensor,delta)
            loss1 = LA.norm(current_X_tensor-current_Y_tensor)
            print("loss1:",loss1)
            if (loss1 < threshold1):
                break
            else: 
                last_W_tensor = current_W_tensor
                last_Y_tensor = current_Y_tensor
                last_X_tensor = current_X_tensor
                last_theta_tensor = current_theta_tensor
                last_pai_tensor = current_pai_tensor 
        
        X_mat1 = current_X_tensor[0]
        X_mat2 = current_X_tensor[1]
        X_mat3 = current_X_tensor[2]
        X_mat4 = current_X_tensor[3] 
        
        X_vector1 = X_mat1.flatten()
        X_vector2 = X_mat2.flatten()
        X_vector3 = X_mat3.flatten()
        X_vector4 = X_mat4.flatten()
               
        test_id = test_all[fold_int]
        test_index = known_samples_index[test_id]
        
        circrna_labels = m6A_disease_circrna_vector[test_index]
        circrna_scores = X_vector1[test_index]
        
        mirna_labels = m6A_disease_mirna_vector[test_index]
        mirna_scores = X_vector2[test_index]
        
        rbp_labels = m6A_disease_rbp_vector[test_index]
        rbp_scores = X_vector3[test_index]
        
        splice_labels = m6A_disease_splice_vector[test_index]
        splice_scores = X_vector4[test_index]
        
        labels = np.hstack((circrna_labels, mirna_labels,rbp_labels,splice_labels))
        scores = np.hstack((circrna_scores, mirna_scores, rbp_scores, splice_scores))
        
        fpr,tpr,threshold=roc_curve(labels,scores,pos_label=1)
        interp_y=interp(mean_fpr, fpr, tpr)     
        tprs.append(interp_y)      
        tprs[-1][0]=0.0
        roc_auc = auc(fpr, tpr)   #roc_auc of each fold
        print('roc',roc_auc)
        aucs.append(roc_auc)
        plt.plot(fpr,tpr,lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.4f)' % (fold_int, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)   #auc of mean
    std_auc = np.std(aucs)        #5 auc      
    print('mean_auc',mean_auc)
    print('std_auc',std_auc) 
    #integration all ROCs 
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title("ROC for LRTCLS CVpair")
    plt.savefig("D:/m6A_tensor_work/Methods_code/LRTCLS/CV_pair_experiments/full_views/CVpair_ROC.pdf")
    



            
