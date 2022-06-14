#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
import os
os.getcwd()  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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
from tensorly.decomposition import tucker

def tensorAndmatrix(mat1, mat2, mat3, mat4, v, m, n):
    tensor = np.zeros((v,m,n))
    tensor[0] = mat1
    tensor[1] = mat2
    tensor[2] = mat3
    tensor[3] = mat4
    return tensor 

def Matrix2Vector(mat):
    vector = mat.flatten()
    vector[np.isnan(vector)] = 0
    return vector 

def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-rol', '--rol_topofallfeature', type=float, nargs='?', default=40,help='Performing k-fold for cross validation.') 
    parser.add_argument('-thres','--thres_topofallfeature', type=float, nargs='?', default=10,help='Performing k-fold for cross validation.')     
    return parser.parse_args() 

def X_tensor_update(W,S_two_tensor,rol,rank_arr):
    ndarr = W - (1/rol) * S_two_tensor 
    WS_tensor = tl.tensor(ndarr)
    result = tucker(WS_tensor,rank=rank_arr,init = 'random', tol=10e-5)
    core = result.core
    factors = result.factors
    X_tmp1 = tl.tenalg.mode_dot(core, factors[0],mode=0)
    X_tmp2 = tl.tenalg.mode_dot(X_tmp1,factors[1],mode=1)
    X_tensor = tl.tenalg.mode_dot(X_tmp2,factors[2],mode=2)
    return X_tensor 

def W_update(Z,S_two,X,Z_mask,rol):
    coff1 = 1/rol
    coff2 = 2/(2+rol)
    ct = 2 * coff1 * np.multiply(Z_mask,Z) + coff1 * S_two + X 
    lt = coff2 * np.multiply(Z_mask, ct)
    W = ct - lt 
    return W 

def S_two_update(last_S_two,X,W,delta):
    current_S = last_S_two + delta * (X-W)
    return current_S 

def convergence(X,W):
    lossXY = LA.norm(X-W)
    return lossXY

if __name__ == "__main__":
    args=parse_args()    
    rol = args.rol_topofallfeature
    thres = args.thres_topofallfeature
   
    m6A_disease_circrna = pd.read_excel("/home/jiani.ma/m6a_disease_dataset/m6A_circRNA_disease.xlsx", header=None).values  # 131*1338
    m6A_disease_mirna = pd.read_excel("/home/jiani.ma/m6a_disease_dataset/m6A_miRNA_disease.xlsx", header=None).values  # 131*1338
    m6A_disease_rbp = pd.read_excel("/home/jiani.ma/m6a_disease_dataset/m6A_RBP_disease.xlsx", header=None).values  # 131*1338
    m6A_disease_splice = pd.read_excel("/home/jiani.ma/m6a_disease_dataset/m6A_splice_disease.xlsx", header=None).values  # 131*1338

    m6A_disease_circrna_vector = Matrix2Vector(m6A_disease_circrna)
    m6A_disease_mirna_vector = Matrix2Vector(m6A_disease_mirna)
    m6A_disease_rbp_vector = Matrix2Vector(m6A_disease_rbp)
    m6A_disease_splice_vector = Matrix2Vector(m6A_disease_splice)
    
    m = m6A_disease_circrna.shape[0]  # 131
    n = m6A_disease_circrna.shape[1]  # 1338
    v = 4    
    delta = 5
    rank_arr = [3, 100, 100]
    fake_m6a_disease = m6A_disease_circrna
    fake_m6a_disease[np.isnan(fake_m6a_disease)] = 0
    fake_m6a_disease = np.abs(fake_m6a_disease)
    fake_m6a_disease_vector = fake_m6a_disease.flatten()
    known_samples_index = np.where(fake_m6a_disease_vector == 1)[0]  # 是fake_m6a_disease_vector中已知元素的索引
    
    kf = KFold(n_splits=10, shuffle=True)  # 10 fold
    train_all = []
    test_all = []
    for train_ind, test_ind in kf.split(known_samples_index):
        train_all.append(train_ind)  # known_samples_index的索引
        test_all.append(test_ind)  # known_samples_index的索引
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)    
    for fold_int in range(10):
        train_id = train_all[fold_int]
        train_index = known_samples_index[train_id]
        train_mask_vector = np.zeros(len(fake_m6a_disease_vector))
        train_mask_vector[train_index] = 1  # to mask the train element
        circrna_train_vector = np.multiply(m6A_disease_circrna_vector, train_mask_vector)
        mirna_train_vector = np.multiply(m6A_disease_mirna_vector, train_mask_vector)
        rbp_train_vector = np.multiply(m6A_disease_rbp_vector, train_mask_vector)
        splice_train_vector = np.multiply(m6A_disease_splice_vector, train_mask_vector)
    
        Z_v1 = circrna_train_vector.reshape(m, n)
        Z_v2 = mirna_train_vector.reshape(m, n)
        Z_v3 = rbp_train_vector.reshape(m, n)
        Z_v4 = splice_train_vector.reshape(m, n)        
        Z_tensor = tensorAndmatrix(Z_v1, Z_v2, Z_v3, Z_v4, v, m, n)        
        
        Z1_mask = Z2_mask = Z3_mask = Z4_mask = train_mask_vector.reshape(m, n)        
        Z_mask_tensor = tensorAndmatrix(Z1_mask, Z2_mask, Z3_mask, Z4_mask, v, m, n)
        
        last_W_tensor = last_X_tensor = last_S_two_tensor = np.random.random((v,m,n))
        current_W_tensor = current_X_tensor = current_S_two_tensor = np.zeros((v,m,n))
        snL_tensor = np.zeros((v,n,n))                        
        for iteration in range(200):             
            current_X_tensor = X_tensor_update(last_W_tensor,last_S_two_tensor,rol,rank_arr)
            current_W_tensor[0] = W_update(Z_v1,last_S_two_tensor[0],current_X_tensor[0],Z1_mask,rol)
            current_W_tensor[1] = W_update(Z_v2,last_S_two_tensor[1],current_X_tensor[1],Z2_mask,rol)
            current_W_tensor[2] = W_update(Z_v3,last_S_two_tensor[2],current_X_tensor[2],Z3_mask,rol)
            current_W_tensor[3] = W_update(Z_v4,last_S_two_tensor[3],current_X_tensor[3],Z4_mask,rol)
            
            current_S_two = S_two_update(last_S_two_tensor,current_X_tensor,current_W_tensor,delta)
            
            loss = convergence(current_X_tensor,current_W_tensor)
            print("loss:",loss)
            if (loss < thres):
                break
            else: 
                last_W_tensor = current_W_tensor
                last_X_tensor = current_X_tensor
                last_S_two_tensor = current_S_two_tensor
                
        X_mat1 = last_X_tensor[0]
        X_mat2 = last_X_tensor[1]
        X_mat3 = last_X_tensor[2]
        X_mat4 = last_X_tensor[3]        
        
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
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title("LRTC_CVpair")
    plt.savefig("./LRTC.pdf")                
                
                
        
        