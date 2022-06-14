# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 20:16:30 2021
@author: Jiani Ma

"""
 import numpy as np 
import tensorly as tl 
from numpy import linalg as LA
from tensorly import unfold
import pandas as pd
from tensorly import fold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def Z_build(mat1,mat2,mat3,mat4,v,m,n):    
    Z_view_mat = np.hstack((mat1,mat2,mat3,mat4))
    Z_view_mat[np.isnan(Z_view_mat)]=0
    Z_vector = Z_view_mat.flatten()    
    return Z_view_mat, Z_vector

def matrix2tensor(mat1,mat2,mat3,mat4,v,m,n):
    tensor = np.zeros((v,m,n))
    tensor[0] = mat1
    tensor[1] = mat2
    tensor[2] = mat3
    tensor[3] = mat4
    tensor[np.isnan(tensor)]=0
    tensor_tl = tl.tensor(tensor)
    return tensor_tl

def M_mode(X_unfolding,alpha,beta):
    tol = alpha/beta
    U,s,VT = LA.svd(X_unfolding,full_matrices=False)
    shrink_s = s - tol
    shrink_s[shrink_s<0]=0
    S = np.diag(shrink_s)
    M_mode = np.dot(U,S)
    M_unfolding = np.dot(M_mode,VT)
    #print("shrink_s",shrink_s)
    return M_unfolding

def unfolding2slice(mat, v, n):
    mat1 = mat[:,0:n]
    mat2 = mat[:,n:2*n]
    mat3 = mat[:,2*n:3*n]
    mat4 = mat[:,3*n:4*n]   
    return mat1, mat2, mat3, mat4

def X_tensor_update(M0,M1,M2,v,m,n,alpha_li,beta_li,train_mat,train_mask_mat):    
    M0_tensor = fold(M0,0,(v,m,n))
    M1_tensor = fold(M1,1,(v,m,n))
    M2_tensor = fold(M1,2,(v,m,n))
    beta_M = beta_li[0]*M0_tensor + beta_li[1]*M1_tensor + beta_li[2]*M2_tensor
    beta_sum = sum(beta_li)
    pre_X_tensor = beta_M/beta_sum 
    X_unfolding_mat = np.hstack((pre_X_tensor[0],pre_X_tensor[1],pre_X_tensor[2],pre_X_tensor[3]))
    X_vector = X_unfolding_mat.flatten()
    train_vector = train_mat.flatten()
    pos_id = np.where(train_vector==1)[0]
    neg_id = np.where(train_vector==-1)[0]
    X_vector[pos_id]=1
    X_vector[neg_id]=-1    
    X_mat = X_vector.reshape((train_mat.shape[0],train_mat.shape[1]))
    Xview1, Xview2, Xview3, Xview4 = unfolding2slice(X_mat, v, n)
    X_tensor = matrix2tensor(Xview1, Xview2, Xview3, Xview4,v,m,n)
    X_tensor = tl.tensor(X_tensor) 
    return X_tensor

def Matrix2Vector(mat):
    vector = mat.flatten()
    vector[np.isnan(vector)] = 0
    return vector 

if __name__=="__main__":
    v = 4 
    alpha_li = [2,2,2]
    beta_li = [0.5,0.5,0.5]
    threshold1 = 25
    """
    consist Z tensor, and Z_unfolding
    """      
    m6A_disease_circrna = pd.read_excel("./m6A_circRNA_disease.xlsx",header=None).values  #131*1338 
    m6A_disease_mirna = pd.read_excel("./m6A_miRNA_disease.xlsx",header=None).values  #131*1338 
    m6A_disease_rbp = pd.read_excel("./m6A_RBP_disease.xlsx",header=None).values  #131*1338 
    m6A_disease_splice = pd.read_excel("./m6A_splice_disease.xlsx",header=None).values  #131*1338

    """
    transform the matrix with missing entries to vector without missing data
    """    
    m6A_disease_circrna_vector = Matrix2Vector(m6A_disease_circrna)
    m6A_disease_mirna_vector = Matrix2Vector(m6A_disease_mirna)
    m6A_disease_rbp_vector = Matrix2Vector(m6A_disease_rbp)
    m6A_disease_splice_vector = Matrix2Vector(m6A_disease_splice)

    
    m = m6A_disease_circrna.shape[0]
    n = m6A_disease_circrna.shape[1]  
    """
    find the mask of the train matrix 
    """
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
        print("fold:",fold_int)
        train_id = train_all[fold_int]
        train_index = known_samples_index[train_id]
        train_mask_vector = np.zeros(len(fake_m6a_disease_vector))
        train_mask_vector[train_index] = 1   #to mask the train element
        
        circrna_train_vector = np.multiply(m6A_disease_circrna_vector,train_mask_vector)
        mirna_train_vector = np.multiply(m6A_disease_mirna_vector,train_mask_vector) 
        rbp_train_vector = np.multiply(m6A_disease_rbp_vector,train_mask_vector)
        splice_train_vector = np.multiply(m6A_disease_splice_vector,train_mask_vector)
        
        circrna_train_mat = circrna_train_vector.reshape((m,n))
        mirna_train_mat = mirna_train_vector.reshape((m,n))
        rbp_train_mat = rbp_train_vector.reshape((m,n))
        splice_train_mat = splice_train_vector.reshape((m,n))
        
        train_mat = np.hstack((circrna_train_mat,mirna_train_mat,rbp_train_mat,splice_train_mat))
        train_mask_mat = np.abs(train_mat)
        
        Z_tensor = np.zeros((v,m,n))
        Z_tensor[0] = circrna_train_mat
        Z_tensor[1] = mirna_train_mat
        Z_tensor[2] = rbp_train_mat
        Z_tensor[3] = splice_train_mat
        Z_tl = tl.tensor(Z_tensor)
        
        Z_unfold0 = unfold(Z_tl,0)
        Z_unfold1 = unfold(Z_tl,1)
        Z_unfold2 = unfold(Z_tl,2)
        
        lastM0 = np.random.random((Z_unfold0.shape[0],Z_unfold0.shape[1]))
        lastM1 = np.random.random((Z_unfold1.shape[0],Z_unfold1.shape[1]))
        lastM2 = np.random.random((Z_unfold2.shape[0],Z_unfold2.shape[1]))
        lastX_tensor = np.random.random((v,m,n))
       
        for i in range(200):
            currentX_tensor = X_tensor_update(lastM0,lastM1,lastM2,v,m,n,alpha_li,beta_li,train_mat,train_mask_mat)
            currentX_unfolding0 = unfold(currentX_tensor,0)
            currentX_unfolding1 = unfold(currentX_tensor,1)
            currentX_unfolding2 = unfold(currentX_tensor,2)
           
            currentM0 = M_mode(currentX_unfolding0,alpha_li[0],beta_li[0])
            currentM1 = M_mode(currentX_unfolding1,alpha_li[1],beta_li[1])
            currentM2 = M_mode(currentX_unfolding2,alpha_li[2],beta_li[2])
            
            loss1 = LA.norm(currentX_tensor-lastX_tensor)
            print("loss1:",loss1) 
            
            if (loss1 < threshold1):
                break
            else: 
                lastM0 = currentM0
                lastM1 = currentM1
                lastM2 = currentM2
                lastX_tensor = currentX_tensor
            
        X_mat1 = currentX_tensor[0]
        X_mat2 = currentX_tensor[1]
        X_mat3 = currentX_tensor[2]
        X_mat4 = currentX_tensor[3] 
        
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
    plt.title("siLRTC CVpair")
    plt.savefig("./ROC_for_siLRTC_CVpair.pdf")
            
            
            
            
            
        
        
        
        
        
        
    
    
    