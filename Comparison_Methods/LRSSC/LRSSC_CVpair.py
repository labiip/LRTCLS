#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:18:46 2020
@author: Jiani
"""
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
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


def Combine_X(XSD_train_matrix,XSS,XSD_train_mask,XDD):
    XSS_mask = np.ones_like(XSS)
    XDD_mask = np.ones_like(XDD)
    X_row1 = np.hstack((XSS,XSD_train_matrix))
    X_row2 = np.hstack((XSD_train_matrix.T,XDD))
    X = np.vstack((X_row1,X_row2))
    X_mask_row1 = np.hstack((XSS_mask,XSD_train_mask))
    X_mask_row2 = np.hstack((XSD_train_mask.T,XDD_mask))
    X_mask = np.vstack((X_mask_row1,X_mask_row2))
    return X, X_mask

def svt(A,tol):
    U,s,VT = LA.svd(A,full_matrices=False)
    shrink_s = s - tol
    shrink_s[shrink_s<0]=0
    S = np.diag(shrink_s)
    svtm_tmp = np.dot(U,S)
    svtm = np.dot(svtm_tmp,VT)
    return svtm

def X_update(Y2,mu,M,A,X0,X0_mask):
    X1 = Y2 + mu*M
    invmat = (1+mu)*np.identity(A.shape[0])+np.dot(A,A.T)-A.T-A
    X2 = LA.inv(invmat)
    X_tmp = np.dot(X1,X2)
    X_tmp_arr = X_tmp.flatten() #value
    X0_arr = X0.flatten()    #value
    X0_mask_arr = X0_mask.flatten() #mask
    exist_id = np.nonzero(X0_mask_arr)
    X_tmp_arr[exist_id] = X0_arr[exist_id]
    X_tmp_arr[X_tmp_arr>1]=1
    X_tmp_arr[X_tmp_arr<0]=0
    X = X_tmp_arr.reshape((X0.shape[0],X0.shape[1]))
    return X

def M_update(mu,X,Y2):
    tol = 1/mu
    mat = X - tol*Y2
    M = svt(mat,tol)
    return M
    
def C_update(A,Y1,lamda,mu):
    thresh = lamda/mu
    mat = A + (1/mu) * Y1 
    C1 = mat - thresh
    C1[C1<0] = 0
    C2 = mat + thresh
    C2[C2>0] = 0 
    C = C1 + C2
    return C

def A_update(X,mu,Y1,C):
    invmat = np.dot(X.T,X) + mu* np.identity(X.shape[1])
    A1 = LA.inv(invmat)
    A2 = np.dot(X.T,X) - Y1 + mu*C
    A = np.dot(A1,A2)    
    return A


def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-lamda', '--lamda_topofallfeature', type=float, nargs='?', default=0.25,help='Performing k-fold for cross validation.')
    parser.add_argument('-mu', '--mu_topofallfeature', type=float, nargs='?', default=0.25,help='Performing k-fold for cross validation.')
    return parser.parse_args() 


if __name__=="__main__":
    args=parse_args()
    lamda = args.lamda_topofallfeature
    mu = args.mu_topofallfeature

    alpha = 0.5
    err_th = 0.02
    delta = 0.01

    XSD_path = "./m6A_circRNA_disease.xlsx"
    XSD = pd.read_excel(XSD_path,header=None).values
    XSD_vector = XSD.flatten()
    
    known_pos_index = np.where(XSD_vector == 1)[0]   
    pos_kf = KFold(n_splits = 10, shuffle=True)
    pos_train_all = [] 
    pos_test_all= []
    for pos_train_ind, pos_test_ind in pos_kf.split(known_pos_index): 
        pos_train_all.append(pos_train_ind)    
        pos_test_all.append(pos_test_ind)
    
    known_neg_index = np.where(XSD_vector ==-1)[0]
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
        print('fold_int',fold_int)
        pos_train_id = pos_train_all[fold_int]            
        XSD_pos_train_id = known_pos_index[pos_train_id]    
        XSD_pos_train_list = np.zeros_like(XSD_vector)
        XSD_pos_train_list[XSD_pos_train_id] = 1
        
        neg_train_id = neg_train_all[fold_int]
        XSD_neg_train_id = known_neg_index[neg_train_id]
        XSD_neg_train_list = np.zeros_like(XSD_vector)
        XSD_neg_train_list[XSD_neg_train_id] = -1
        
        XSD_train_list = XSD_pos_train_list + XSD_neg_train_list
        XSD_train_matrix = XSD_train_list.reshape((XSD.shape[0],XSD.shape[1]))
        XSD_train_mask = np.abs(XSD_train_matrix)
        
        pos_test_id = pos_test_all[fold_int]
        XSD_pos_test_id = known_pos_index[pos_test_id]
        XSD_pos_test_mask_list = np.zeros_like(XSD_vector)
        XSD_pos_test_mask_list[XSD_pos_test_id] = 1
        #XSD_pos_test_mask = XSD_pos_test_mask_list.reshape((XSD.shape[0],XSD.shape[1]))

        neg_test_id = neg_test_all[fold_int]
        XSD_neg_test_id = known_neg_index[neg_test_id]
        XSD_neg_test_mask_list = np.zeros_like(XSD_vector)
        XSD_neg_test_mask_list[XSD_neg_test_id] = 1 
        #XSD_neg_test_mask = XSD_neg_test_mask_list.reshape((XSD.shape[0],XSD.shape[1]))
        
        X0 = XSD_train_matrix
        X0_mask = XSD_train_mask
    
        lastX = np.random.random((X0.shape[0],X0.shape[1]))
        lastM =np.random.random((X0.shape[0],X0.shape[1]))
        lastY2 = np.random.random((X0.shape[0],X0.shape[1]))
        
        lastC = np.random.random((X0.shape[1],X0.shape[1]))
        lastY1 = np.random.random((X0.shape[1],X0.shape[1]))
        lastA = np.random.random((X0.shape[1],X0.shape[1]))

        for i in range(200): 
            #print("iteration",i)
            currentX = X_update(lastY2,mu,lastM,lastA,X0,X0_mask)
            currentM = M_update(mu,currentX,lastY2)
            currentA = A_update(currentX,mu,lastY1,lastC)
            currentC = C_update(currentA,lastY1,lamda,mu)
            currentY1 = lastY1 + delta*(currentA-currentC)
            currentY2 = lastY2 + delta*(currentM-currentX)
            err_X = np.max(abs(currentX-lastX))
            err_M = np.max(abs(currentM-lastM))
            err_XM = np.max(abs(currentX-currentM))            
            print("err_X",err_X)
            if (err_X < err_th) and (err_M < err_th) and (err_XM < err_th):
                break
            else:
                lastX = currentX
                lastM = currentM
                lastA = currentA
                lastC = currentC
                lastY1 = currentY1
                lastY2 = currentY2
        
        X_unfolding_vector = currentX.flatten()    
        
        pos_samples_index = np.where(XSD_pos_test_mask_list==1)[0]
        neg_samples_index = np.where(XSD_neg_test_mask_list==1)[0]
        
        pos_samples = X_unfolding_vector[pos_samples_index]
        neg_samples = X_unfolding_vector[neg_samples_index]
        
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
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title("LRSSC_CVpair")            
    plt.savefig("./ROC_for_LRSSC_CVpair.pdf")
