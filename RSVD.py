import numpy as np
import scipy as sp
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix

def my_base_rsvd(A, rank_A, s):
    Proj = np.random.random((A.shape[1], rank_A + s))
    Z = A @ Proj                                   
    Q, _ = np.linalg.qr(Z, mode = 'reduced')
    Y = Q.T @ A
    U_y, s, Vh = np.linalg.svd(Y, full_matrices = False)
    U_x = Q @ U_y
    return U_x, s, Vh
    
def basic_rSVD(A, rank_A, s, p):
    Proj = np.random.random((A.shape[1], rank_A + s))
    Z = A @ Proj                                   
    Q, _ = np.linalg.qr(Z, mode = 'reduced')
    for i in range(p):
        G, _ = np.linalg.qr(A.T @ Q, mode = 'reduced')
        Q, _ = np.linalg.qr(A @ G, mode = 'reduced')
    B = Q.T @ A
    U_y, s, Vh = np.linalg.svd(B, full_matrices = False)
    U_x = Q @ U_y
    return U_x[:,:rank_A], s[:rank_A], Vh[:rank_A,:]
    
def eigSVD(M):
    if M.shape[1] > M.shape[0]:
        print("incorrect matrix size. m should be >= n for Matrix - m*n")
    B = M.T @ M
    D, V = np.linalg.eigh(B)
    D = D[::-1]
    V = V[:,::-1]
    D = np.abs(D)
    S_ = np.sqrt(D)
    U = M @ V @ np.diag(1/S_)
    return U, S_, V.T
    
def eig_basic_rSVD(A, rank_A, s, p):
    if rank_A + s < A.shape[1]:
        s = A.shape[1] - rank_A
    Proj = np.random.random((A.shape[1], rank_A + s))
    Z = A @ Proj                                   
    Q, _ = np.linalg.qr(Z, mode = 'reduced')
    for i in range(p):
        G, _ = np.linalg.qr(A.T @ Q, mode = 'reduced')
        Q, _ = np.linalg.qr(A @ G, mode = 'reduced')
    B = Q.T @ A
    U_y, s, Vh = eigSVD(B)
    U_x = Q @ U_y
    return U_x[:,:rank_A], s[:rank_A], Vh[:rank_A,:]
    
def rSVD_PI(A, rank_A, s, p):
    Proj = np.random.random((A.shape[1], rank_A + s))
    Q = A @ Proj
    for i in range(0, p+1):
        if i < p:
            Q, _ =  sp.linalg.lu(Q, permute_l = True)
        else:
            Q, _, _ = eigSVD(Q)
            break
        Q = A @ (A.T @ Q)
    B = Q.T @ A
    V, S, U = eigSVD(B.T)
    ind = np.arange(s+1, rank_A + s)
    U = Q @ U
    return U[:,ind], S[ind,:][:,ind], V[:, ind]
    
def rSVD_BKI(A, rank_A, s, p):
    Proj = np.random.random((A.shape[1], rank_A + s))
    H_all = []
    H, _ =sp.linalg.lu(A @ Proj, permute_l = True)
    H_all.append(H)
    for i in range(1, p+1):
        if i < p:
            H, _ =  sp.linalg.lu(A @ (A.T @ H), permute_l = True)
        H_all.append(H)    
    H_all = np.hstack(H_all)
    Q, _ = np.linalg.qr(H_all, mode = 'reduced')
    B = Q.T @ A
    V, S, U = eigSVD(B.T)
    start = (rank_A + s) * (p + 1) - rank_A + 1
    stop = (rank_A + s) * (p + 1)
    U = Q @ U
    return U[:, start:stop], np.diagonal(S[start:stop, start:stop]), V[:, start:stop]
