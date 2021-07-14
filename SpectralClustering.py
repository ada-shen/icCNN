from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import torch

def spectral_clustering(similarity_matrix,n_cluster=8):
    W = similarity_matrix
    
    sz = W.shape[0]
    sp = SpectralClustering(n_clusters=n_cluster,affinity='precomputed',random_state=21)
    y_pred = sp.fit_predict(W)
    # for i in range(n_cluster):
    #     print(np.sum(y_pred==i))
    del W
    ground_true_matrix = np.zeros((sz,sz))
    loss_mask_num = []
    loss_mask_den = []
    for i in range(n_cluster):
        idx = np.where(y_pred==i)[0]
        cur_mask_num = np.zeros((sz,sz))
        cur_mask_den = np.zeros((sz,sz))
        for j in idx:
            ground_true_matrix[j][idx] = 1
            cur_mask_num[j][idx] = 1
            cur_mask_den[j][:] = 1
        loss_mask_num.append(np.expand_dims(cur_mask_num,0))
        loss_mask_den.append(np.expand_dims(cur_mask_den,0))
    loss_mask_num = np.concatenate(loss_mask_num,axis=0)
    loss_mask_den = np.concatenate(loss_mask_den,axis=0)
    return torch.from_numpy(ground_true_matrix).float().cuda(), torch.from_numpy(loss_mask_num).float().cuda(), torch.from_numpy(loss_mask_den).float().cuda() 