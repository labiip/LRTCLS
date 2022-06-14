#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
from sklearn.model_selection import KFold
from numpy import linalg as LA
np.set_printoptions(suppress = True)
np.random.seed(0)

def Matrix2Vector(mat):
    vector = mat.flatten()
    vector[np.isnan(vector)] = 0
    return vector 

def SnLaplacianMatrix(X,n): 
    W = np.zeros((n,n))
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = np.dot(X[:,i],X[:,j])
    d = np.sum(W,axis = 1)   #row sum 
    D = np.diag(d) 
    snL = D - W 
    return snL 

def X_update(rol,Z,Z_mask,Y,S_one): 
    coff1 = 1/rol 
    coff2 = 2/(2+rol)
    ct = coff1 * np.multiply(Z_mask,Z) + Y - coff1 * S_one
    X = ct - coff2 * np.multiply(Z_mask,ct)
    return X
   
def Y_update(X,S_one,alpha,rol,snL,n):
    pt = S_one + rol * X
    lt_inv = alpha * (snL.T + snL) + rol * np.identity(n)
    lt = LA.inv(lt_inv)
    Y = np.dot(pt,lt)
    return Y

def S_one_update(X,Y,delta,last_S):
    S = last_S + delta*(X-Y)
    return S 

def convergence(X,Y):
    lossXY = LA.norm(X-Y)
    return lossXY

def tensorAndmatrix(mat1, mat2, mat3, mat4, v, m, n):
    tensor = np.zeros((v,m,n))
    tensor[0] = mat1
    tensor[1] = mat2
    tensor[2] = mat3
    tensor[3] = mat4
    return tensor 

def parse_args():    
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-alpha', '--alpha_topofallfeature', type=float, nargs='?', default=2,help='Performing k-fold for cross validation.') 
    parser.add_argument('-rol', '--rol_topofallfeature', type=float, nargs='?', default=0.5,help='Performing k-fold for cross validation.') 
    parser.add_argument('-threshold1','--threshold1_topofallfeature', type=float, nargs='?', default=100,help='Performing k-fold for cross validation.')     
    return parser.parse_args() 

if __name__ == "__main__":
    
    args=parse_args()
    alpha = args.alpha_topofallfeature
    rol = args.rol_topofallfeature
    threshold1 = args.threshold1_topofallfeature
    
    m6A_disease_circrna = pd.read_excel("./m6A_circRNA_disease.xlsx", header=None).values  # 131*1338
    m6A_disease_mirna = pd.read_excel("./m6A_miRNA_disease.xlsx", header=None).values  # 131*1338
    m6A_disease_rbp = pd.read_excel("./m6A_RBP_disease.xlsx", header=None).values  # 131*1338
    m6A_disease_splice = pd.read_excel("./m6A_splice_disease.xlsx", header=None).values  # 131*1338

    m6A_disease_circrna_vector = Matrix2Vector(m6A_disease_circrna)
    m6A_disease_mirna_vector = Matrix2Vector(m6A_disease_mirna)
    m6A_disease_rbp_vector = Matrix2Vector(m6A_disease_rbp)
    m6A_disease_splice_vector = Matrix2Vector(m6A_disease_splice)
    
    m = m6A_disease_circrna.shape[0]  # 131
    n = m6A_disease_circrna.shape[1]  # 1338
    v = 4    
    delta = 1.2
    rank_arr = [3, 100, 100]
    fake_m6a_disease = m6A_disease_circrna
    fake_m6a_disease[np.isnan(fake_m6a_disease)] = 0
    fake_m6a_disease = np.abs(fake_m6a_disease)
    fake_m6a_disease_vector = fake_m6a_disease.flatten()
    known_samples_index = np.where(fake_m6a_disease_vector == 1)[0]  
    
    kf = KFold(n_splits=10, shuffle=True)  # 10 fold
    train_all = []
    test_all = []
    for train_ind, test_ind in kf.split(known_samples_index):
        train_all.append(train_ind)  
        test_all.append(test_ind)  #
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)    
    
    for fold_int in range(10):
        print("fold:",fold_int)
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
        
        last_Y_tensor = last_X_tensor = last_S_one_tensor = np.random.random((v,m,n))
        current_Y_tensor = current_X_tensor = current_S_one_tensor = np.zeros((v,m,n))
        snL_tensor = np.zeros((v,n,n))                        
        
        for iteration in range(200):             
            current_X_tensor = X_update(rol,Z_tensor,Z_mask_tensor,last_Y_tensor,last_S_one_tensor)                
            snL_tensor[0] = SnLaplacianMatrix(last_Y_tensor[0],n)
            snL_tensor[1] = SnLaplacianMatrix(last_Y_tensor[1],n)
            snL_tensor[2] = SnLaplacianMatrix(last_Y_tensor[2],n)
            snL_tensor[3] = SnLaplacianMatrix(last_Y_tensor[3],n)
     
            current_Y_tensor[0] = Y_update(current_X_tensor[0],last_S_one_tensor[0],alpha,rol,snL_tensor[0],n)
            current_Y_tensor[1] = Y_update(current_X_tensor[1],last_S_one_tensor[1],alpha,rol,snL_tensor[1],n)
            current_Y_tensor[2] = Y_update(current_X_tensor[2],last_S_one_tensor[2],alpha,rol,snL_tensor[2],n)
            current_Y_tensor[3] = Y_update(current_X_tensor[3],last_S_one_tensor[3],alpha,rol,snL_tensor[3],n)
    
            current_S_one_tensor = S_one_update(current_X_tensor,current_Y_tensor,delta,last_S_one_tensor)                   
            loss = convergence(current_X_tensor,current_Y_tensor)
            
            print("loss:",loss)
            
            if (loss < threshold1):
                break
            else: 
                last_Y_tensor = current_Y_tensor
                last_X_tensor = current_X_tensor
                last_S_one_tensor = current_S_one_tensor
                      
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
    plt.savefig("TCLS_CVpair.pdf")



            