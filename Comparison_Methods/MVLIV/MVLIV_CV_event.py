# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:21:32 2021
@author: Jiani Ma
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd 
import numpy as np
import matplotlib as mpl
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
np.random.seed(0)

"""
unfolding matrice and tensor  
"""
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

def Ui_update(Zi,W):
    W_inner = np.dot(W,W.T)
    W_inner_inv = LA.inv(W_inner)
    Ui_tmp = np.dot(Zi,W.T)
    Ui = np.dot(Ui_tmp,W_inner_inv)
    return Ui

def Zi_update(Ui,W,Yi,Xi,mu,mask):
    Zi_befo = np.dot(Ui,W) - np.multiply(mask,Yi) + mu*np.multiply(mask,Xi)
    Zi_behi_tmp = np.dot(Ui,W)-Yi + mu*Xi
    Zi_behi = (mu/(mu+1)) * np.multiply(mask,Zi_behi_tmp)
    Zi = Zi_befo - Zi_behi
    return Zi

def W_update(U1,U2,U3,U4,Z1,Z2,Z3,Z4):
    U_inner = np.dot(U1.T,U1) + np.dot(U2.T,U2) + np.dot(U2.T,U2) + np.dot(U3.T,U3)+ np.dot(U4.T,U4) 
    U_inner_inv = LA.inv(U_inner)
    U_Z_inner = np.dot(U1.T,Z1) + np.dot(U2.T,Z2) + np.dot(U3.T,Z3) + np.dot(U4.T,Z4)
    W = np.dot(U_inner_inv,U_Z_inner)
    return W

def Yi_update(lastYi, delta, Xi, Zi, mask): 
    Yi = lastYi + delta * np.multiply(mask,(Zi-Xi))
    return Yi 

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
                                                                                                                                    
def converge_condition(lastZi, Xi, currentZi, mask):
    loss1_up = LA.norm(currentZi-lastZi)
    loss1_do = LA.norm(currentZi)
    loss1 = loss1_up/loss1_do
    loss2_up = LA.norm(np.multiply(mask,(currentZi-Xi))) 
    loss2_do = LA.norm(np.multiply(mask,Xi)) 
    loss2 = loss2_up / loss2_do  
    return loss1, loss2 
"""
define updating function
"""

if __name__=="__main__":
    m6A_disease_circrna = pd.read_excel("./m6A_circRNA_disease.xlsx",header=None).values  #131*1338 
    m6A_disease_mirna = pd.read_excel("./m6A_miRNA_disease.xlsx",header=None).values  #131*1338 
    m6A_disease_rbp = pd.read_excel("./m6A_RBP_disease.xlsx",header=None).values  #131*1338 
    m6A_disease_splice = pd.read_excel("./m6A_splice_disease.xlsx",header=None).values  #131*1338
    
    m = m6A_disease_circrna.shape[0]
    n = m6A_disease_circrna.shape[1]
    v = 4
    k = 90        #rank
    mu = 0.5
    delta = 0.001
    threshold1 = 0.1      #0.015
    threshold2 = 0.7          #0.4
    tensor, X = tensorAndmatrix(m6A_disease_circrna, m6A_disease_mirna, m6A_disease_rbp, m6A_disease_splice, v, m,n)  #tensor:4*131*1338, X:131*5352
    X_vector = X.flatten()   #701112
    """
    10-fold for postive samples in X_vector 
    """
    known_pos_index = np.where(X_vector == 1)[0]   
    pos_kf = KFold(n_splits = 10, shuffle=True)
    pos_train_all = [] 
    pos_test_all= []
    for pos_train_ind, pos_test_ind in pos_kf.split(known_pos_index): 
        pos_train_all.append(pos_train_ind)
        pos_test_all.append(pos_test_ind)
    """
    10-fold for negtive samples in X_vector
    """
    known_neg_index = np.where(X_vector ==-1)[0]
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
        pos_train_id = pos_train_all[fold_int]
        X_pos_train_id = known_pos_index[pos_train_id]
        X_pos_train_list = np.zeros_like(X_vector)
        X_pos_train_list[X_pos_train_id] = 1
        
        neg_train_id = neg_train_all[fold_int]
        X_neg_train_id = known_neg_index[neg_train_id]
        X_neg_train_list = np.zeros_like(X_vector)
        X_neg_train_list[X_neg_train_id] = -1
        
        train_list = X_pos_train_list + X_neg_train_list
        train_mat = train_list.reshape((X.shape[0],X.shape[1]))  # size 701112 ---> 131 * (1338*4) 
        train_mask_mat = np.abs(train_mat)
        """
        test list and test matrix : X_pos_test_mask_list, X_pos_test_mat,X_pos_test_mask; 
                                    X_neg_test_list, X_neg_test_mat, X_neg_test_mask
        """
        pos_test_id = pos_test_all[fold_int]
        X_pos_test_id = known_pos_index[pos_test_id]
        X_pos_test_list = np.zeros_like(X_vector)
        X_pos_test_list[X_pos_test_id] = 1
        X_pos_test_mask_list = X_pos_test_list
        X_pos_test_mat = X_pos_test_mask_list.reshape((X.shape[0],X.shape[1]))
        X_pos_test_mask = X_pos_test_mask_list.reshape((X.shape[0],X.shape[1]))
        
        neg_test_id = neg_test_all[fold_int]
        X_neg_test_id = known_neg_index[neg_test_id]
        X_neg_test_list = np.zeros_like(X_vector)
        X_neg_test_list[X_neg_test_id] = 1 
        X_neg_test_mask_list = np.abs(X_neg_test_list)
        X_neg_test_mat = X_neg_test_list.reshape((X.shape[0],X.shape[1]))
        X_neg_test_mask = X_neg_test_mask_list.reshape((X.shape[0],X.shape[1]))
        """
        algorithm： 
        input: train_mat, train_mask_mat
        """
        X1, X2, X3, X4, X_mask1, X_mask2, X_mask3, X_mask4 = unfolding2slice(train_mat, train_mask_mat, v, n)
            
        lastU1 = lastU2 = lastU3 = lastU4 = np.random.random((m,k))
        lastZ1 = lastZ2 = lastZ3 = lastZ4 = np.random.random((m,n))
        lastY1 = lastY2 = lastY3 = lastY4 = np.random.random((m,n))
        lastW = np.random.random((k,n))

        for i in range(100):
            currentZ1 = Zi_update(lastU1,lastW,lastY1,X1,mu,X_mask1)            
            currentZ2 = Zi_update(lastU2,lastW,lastY2,X2,mu,X_mask2)
            currentZ3 = Zi_update(lastU3,lastW,lastY3,X3,mu,X_mask3)
            currentZ4 = Zi_update(lastU4,lastW,lastY4,X4,mu,X_mask4)
            
            currentU1 = Ui_update(currentZ1,lastW)           
            currentU2 = Ui_update(currentZ2,lastW)
            currentU3 = Ui_update(currentZ3,lastW)
            currentU4 = Ui_update(currentZ4,lastW)
            
            currentW = W_update(currentU1,currentU2,currentU3,currentU4,currentZ1,currentZ2,currentZ3,currentZ4)
            currentY1 = Yi_update(lastY1, delta, X1, currentZ1, X_mask1)
            currentY2 = Yi_update(lastY2, delta, X2, currentZ2, X_mask2)
            currentY3 = Yi_update(lastY3, delta, X3, currentZ3, X_mask3)
            currentY4 = Yi_update(lastY4, delta, X4, currentZ4, X_mask4)
            """
            calculate loss
            """
            los1_arr = np.zeros(4)
            los2_arr = np.zeros(4)
            
            los1_arr[0], los2_arr[0] = converge_condition(lastZ1, X1, currentZ1, X_mask1)
            los1_arr[1], los2_arr[1] = converge_condition(lastZ2, X2, currentZ2, X_mask2)
            los1_arr[2], los2_arr[2] = converge_condition(lastZ3, X3, currentZ3, X_mask3)
            los1_arr[3], los2_arr[3] = converge_condition(lastZ4, X4, currentZ4, X_mask4)
            
            los1 = np.max(los1_arr)
            los2 = np.max(los2_arr)
            
            if (los1 < threshold1) and (los2 < threshold2):
                break 
            else:
                lastZ1 = currentZ1 
                lastZ2 = currentZ2
                lastZ3 = currentZ3
                lastZ4 = currentZ4
                lastU1 = currentU1 
                lastU2 = currentU2
                lastU3 = currentU3
                lastU4 = currentU4 
                lastW = currentW
                lastY1 = currentY1
                lastY2 = currentY2
                lastY3 = currentY3
                lastY4 = currentY4
        
        Z_unfolding_mat1 = np.hstack((currentZ1, currentZ2))
        Z_unfolding_mat2 = np.hstack((Z_unfolding_mat1,currentZ3))
        Z_unfolding_mat = np.hstack((Z_unfolding_mat2,currentZ4))
        Z_vector = Z_unfolding_mat.flatten()
        
        
        pos_sample_index = np.where(X_pos_test_mask_list==1)[0]
        neg_sample_index = np.where(X_neg_test_mask_list==1)[0]
        
        pos_samples = Z_vector[pos_sample_index]
        neg_samples = Z_vector[neg_sample_index]
        
        pos_sample_labels = np.ones_like(pos_samples)
        neg_sample_labels = np.zeros_like(neg_samples)
        
        scores = np.hstack((pos_samples,neg_samples))
        labels = np.hstack((pos_sample_labels,neg_sample_labels))
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
    for k in range(len(mean_tpr)):
        print("mean_tpr:",mean_tpr[k])

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
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title("MVLIV CVevent")
    #plt.show()
    plt.savefig("./ROC_for_MVLIV_CVevent.pdf")
        
        

          