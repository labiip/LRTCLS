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

def Matrix2Vector(mat):
    vector = mat.flatten()
    vector[np.isnan(vector)] = 0
    return vector 

                                                                                                                                    
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
    threshold1 = 0.2     #0.015
    threshold2 = 0.9        #0.4
    # tensor, X = tensorAndmatrix(m6A_disease_circrna, m6A_disease_mirna, m6A_disease_rbp, m6A_disease_splice, v, m,n)  #tensor:4*131*1338, X:131*5352
    # X_vector = X.flatten()   #701112
        
    m6A_disease_circrna_vector = Matrix2Vector(m6A_disease_circrna)
    m6A_disease_mirna_vector = Matrix2Vector(m6A_disease_mirna)
    m6A_disease_rbp_vector = Matrix2Vector(m6A_disease_rbp)
    m6A_disease_splice_vector = Matrix2Vector(m6A_disease_splice)
        
    fake_m6a_disease = m6A_disease_circrna
    fake_m6a_disease[np.isnan(fake_m6a_disease)] = 0
    fake_m6a_disease = np.abs(fake_m6a_disease)    
    fake_m6a_disease_vector = fake_m6a_disease.flatten()    
    known_samples_index = np.where(fake_m6a_disease_vector == 1)[0]   
    
    kf = KFold(n_splits =10, shuffle=True)      #10 fold
    train_all=[]    
    test_all=[]     
    for train_ind,test_ind in kf.split(known_samples_index):  
        train_all.append(train_ind)    
        test_all.append(test_ind)     
  
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_int in range(10):        
        train_id = train_all[fold_int]
        train_index = known_samples_index[train_id]
        train_mask_vector = np.zeros(len(fake_m6a_disease_vector))
        train_mask_vector[train_index] = 1   #to mask the train element
        
        circrna_train_vector = np.multiply(m6A_disease_circrna_vector,train_mask_vector)
        mirna_train_vector = np.multiply(m6A_disease_mirna_vector,train_mask_vector)
        rbp_train_vector = np.multiply(m6A_disease_rbp_vector,train_mask_vector)
        splice_train_vector = np.multiply(m6A_disease_splice_vector,train_mask_vector)
        
        X1 = circrna_train_vector.reshape(m,n)
        X2 = mirna_train_vector.reshape(m,n)
        X3 = rbp_train_vector.reshape(m,n)
        X4 = splice_train_vector.reshape(m,n)
        
        X_mask1 = X_mask2 = X_mask3 = X_mask4 = train_mask_vector.reshape(m,n)
        
        
        lastU1 = lastU2 = lastU3 = lastU4 = np.random.random((m,k))
        lastZ1 = lastZ2 = lastZ3 = lastZ4 = np.random.random((m,n))
        lastY1 = lastY2 = lastY3 = lastY4 = np.random.random((m,n))
        lastW = np.random.random((k,n))

        for i in range(200):
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
        
        Z_vector1 = currentZ1.flatten()
        Z_vector2 = currentZ2.flatten()
        Z_vector3 = currentZ3.flatten()
        Z_vector4 = currentZ4.flatten()

        
        test_id = test_all[fold_int]
        test_index = known_samples_index[test_id]
        
        circrna_labels = m6A_disease_circrna_vector[test_index]
        circrna_scores = Z_vector1[test_index]
        
        mirna_labels = m6A_disease_mirna_vector[test_index]
        mirna_scores = Z_vector2[test_index]
        
        rbp_labels = m6A_disease_rbp_vector[test_index]
        rbp_scores = Z_vector3[test_index]
        
        splice_labels = m6A_disease_splice_vector[test_index]
        splice_scores = Z_vector4[test_index]
        
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
    mean_auc = auc(mean_fpr, mean_tpr)   
    std_auc = np.std(aucs)            
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
    plt.title("MVLIV CVpair")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("./ROC_for_MVLIV_CVpair.pdf")

        
        

          