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

def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-alpha', '--alpha_topofallfeature', type=float, nargs='?', default=1.0,help='Performing k-fold for cross validation.') 
    parser.add_argument('-rol', '--rol_topofallfeature', type=float, nargs='?', default=1.0,help='Performing k-fold for cross validation.') 
    parser.add_argument('-threshold1','--threshold1_topofallfeature', type=float, nargs='?', default=0.01,help='Performing k-fold for cross validation.')
    return parser.parse_args() 


if __name__=="__main__":
    args=parse_args()
    m6A_disease_circrna = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=0).values  #131*1338
    m6A_disease_mirna = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=1).values  #131*1338
    m6A_disease_rbp = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=2).values  #131*1338
    m6A_disease_splice = pd.read_excel("./m6A_event_disease.xlsx",header=None,sheet_name=3).values  #131*1338
    m = m6A_disease_circrna.shape[0]        #131
    n = m6A_disease_circrna.shape[1]        #1338
    v = 4
     
    alpha = args.alpha_topofallfeature
    rol = args.rol_topofallfeature
    threshold1 = args.threshold1_topofallfeature
    
    delta = 1
    rank_arr = [3,100,100]
       
    Z, Z_unfolding = tensorAndmatrix(m6A_disease_circrna, m6A_disease_mirna, m6A_disease_rbp, m6A_disease_splice, v, m,n)  #tensor:4*131*1338, X:131*5352
    Z_vector = Z_unfolding.flatten()   #701112

    """
    10-fold for postive samples in Z_vector 
    """
    known_pos_index = np.where(Z_vector == 1)[0]
    pos_kf = KFold(n_splits = 10, shuffle=True)
    pos_train_all = [] 
    pos_test_all= []
    for pos_train_ind, pos_test_ind in pos_kf.split(known_pos_index):
        pos_train_all.append(pos_train_ind)    
        pos_test_all.append(pos_test_ind)
    """
    10-fold for negtive samples in X_vector
    """
    known_neg_index = np.where(Z_vector ==-1)[0]
    neg_kf = KFold(n_splits =10, shuffle=True)
    neg_train_all = []
    neg_test_all = []
    for neg_train_ind, neg_test_ind in neg_kf.split(known_neg_index):
        neg_train_all.append(neg_train_ind)
        neg_test_all.append(neg_test_ind)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_int in range(10):
        """
        train matrix and mask : train_mat, train_mask_mat
        """
        print("fold_int:",fold_int)
        pos_train_id = pos_train_all[fold_int]            #index of known_pos_index 
        Z_pos_train_id = known_pos_index[pos_train_id]    #index in Z_vector
        Z_pos_train_list = np.zeros_like(Z_vector)
        Z_pos_train_list[Z_pos_train_id] = 1
        
        neg_train_id = neg_train_all[fold_int]
        Z_neg_train_id = known_neg_index[neg_train_id]
        Z_neg_train_list = np.zeros_like(Z_vector)
        Z_neg_train_list[Z_neg_train_id] = -1
        
        train_list = Z_pos_train_list + Z_neg_train_list
        train_mat = train_list.reshape((Z_unfolding.shape[0],Z_unfolding.shape[1]))
        train_mask_mat = np.abs(train_mat)
        """
        test list and test matrix : X_pos_test_mask_list, X_pos_test_mat,X_pos_test_mask; 
                                    X_neg_test_list, X_neg_test_mat, X_neg_test_mask
        """
        pos_test_id = pos_test_all[fold_int]
        Z_pos_test_id = known_pos_index[pos_test_id]
        Z_pos_test_mask_list = np.zeros_like(Z_vector)
        Z_pos_test_mask_list[Z_pos_test_id] = 1
        Z_pos_test_mask = Z_pos_test_mask_list.reshape((Z_unfolding.shape[0],Z_unfolding.shape[1]))

        neg_test_id = neg_test_all[fold_int]
        Z_neg_test_id = known_neg_index[neg_test_id]
        Z_neg_test_mask_list = np.zeros_like(Z_vector)
        Z_neg_test_mask_list[Z_neg_test_id] = 1 
        Z_neg_test_mask = Z_neg_test_mask_list.reshape((Z_unfolding.shape[0],Z_unfolding.shape[1]))
        """
        algorithm： 
        input: train_mat, train_mask_mat
        """
        Z_v1, Z_v2, Z_v3, Z_v4, Z1_mask, Z2_mask,Z3_mask,Z4_mask = unfolding2slice(train_mat, train_mask_mat, v, n)
        """
        initialize 
        """
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
        
        X_unfolding_mat = np.hstack((X_mat1,X_mat2,X_mat3,X_mat4))
        
        X_unfolding_vector = X_unfolding_mat.flatten()
        
        pos_samples_index = np.where(Z_pos_test_mask_list==1)[0]
        neg_samples_index = np.where(Z_neg_test_mask_list==1)[0]
        
        pos_samples = X_unfolding_vector[pos_samples_index]
        neg_samples = X_unfolding_vector[neg_samples_index]
        
        pos_sample_labels = np.ones_like(pos_samples)
        neg_sample_labels = np.zeros_like(neg_samples)     
        
        scores = np.hstack((pos_samples,neg_samples))
        labels = np.hstack((pos_sample_labels,neg_sample_labels))
        fpr,tpr,threshold=roc_curve(labels,scores,pos_label=1)
        interp_y=np.interp(mean_fpr, fpr, tpr)     
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
    plt.xlim([0.00, 1.05])
    plt.ylim([0.00, 1.05])
    plt.title("ROC for LRTCLS CVtriple")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("LRTCLS_CVevent_ROC.pdf")

            