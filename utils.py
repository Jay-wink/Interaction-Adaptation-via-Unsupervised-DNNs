import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
import cyipopt as ipopt
from scipy.linalg import svd, qr
from scipy.sparse import csc_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import case57
from pypower.api import opf, makeYbus
from pypower import idx_bus, idx_gen, ppoption

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()

###################################################################
# INTERACTION MESH
###################################################################
class InteractionMesh:
    def __init__(self, Q, p, A, G, h, X, bones_idx, radii, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._h = torch.tensor(h)
        self._X = torch.tensor(X)
        self._Y = None
        self._xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = G.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac

        # InteractionMesh attributes
        self._bones_idx = bones_idx
        self._radii = radii

        # it is not that easy to decompose A to get an invertible matrix like SimpleProblem, so we have two choices to deal with the both situations
        # 1. check if A can be decomposed to an invertible matrix and another matrix, if so do it as usual
        # 2. if not, use pseudo-inverse and projection instead 
        self._partial_vars, self._other_vars, self._A_partial, self._A_other_inv = self.split_invertible_block(self.A)

        # projection if no invertible block 
        self._AAt_pinv = torch.linalg.pinv(self._A @ self._A.T)    # (m,m), m=neq
        self._proj_mat = self._A.T @ self._AAt_pinv                # (n,m), n=ydim

        ### For Pytorch
        self._device = None

    

    def __str__(self):
        return 'InteractionMesh-{}-{}-{}-{}'.format(
            # self.fp, self.bones, self.num_samples
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G
    
    # update G for penetration detected in real-time
    @G.setter
    def G(self, value):
        self._G = value

    @property
    def h(self):
        return self._h
    
    # update h for penetration detected in real-time
    @h.setter
    def h(self, value):
        self._h = value

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars
    
    @property
    def A_partial(self):
        return self._A_partial
    
    @property
    def A_other_inv(self):
        return self._A_other_inv
    
    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5*(Y@self.Q@Y.T) + Y@self.p).sum(dim=1)

    def eq_resid(self, X, Y):
        return X - Y@self.A.T

    def ineq_resid(self, X, Y):
        return Y@self.G.T - self.h

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2*(Y@self.A.T - X)@self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2*ineq_dist@self.G

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = 2 * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    def project_eq(self, X, Y):
        # X: (B, m), Y: (B, n)
        # R = A Y - X  ->  (B, m)
        R = Y @ self._A.T - X
        # Y_proj = Y - R * (A^T (A A^T)^+)^T
        return Y - R @ self._proj_mat.T

        
    def split_invertible_block(self, A):
        """
        Given a matrix A (m x n), try to permute its columns so that
        A = [A_partial  A_other], where A_other is an invertible square matrix (m x m).

        Parameters
        ----------
        A : np.ndarray or torch.Tensor
            Input matrix of shape (m, n).

        Returns
        -------
        A_partial, A_other : same type as input (np.ndarray or torch.Tensor)
            - A_other is an m x m invertible submatrix formed by selecting m columns of A.
            - A_partial contains the remaining columns.
            - If such a block cannot be found (rank(A) < m or n < m), both are returned as None.
        """

        # ---- 1. Convert to numpy array (keep info to convert back later) ----
        device = A.device
        dtype = A.dtype
        A_np = A.detach().cpu().numpy()
        m, n = A_np.shape

        # If we have fewer columns than rows, we can never get an m x m submatrix
        if n < m:
            return None, None, None, None

        # ---- 2. QR with column pivoting to find linearly independent columns ----
        # A * P = Q * R,  where P encodes a permutation of the columns.
        # 'piv' gives the column indices in order of "importance" (independence).
        Q, R, piv = qr(A_np, mode='economic', pivoting=True)

        # ---- 3. Compute numerical rank from R's diagonal ----
        diag_R = np.abs(np.diag(R))
        eps = np.finfo(A_np.dtype).eps if np.issubdtype(A_np.dtype, np.floating) else 1e-12
        tol = max(m, n) * diag_R.max() * eps
        rank = int((diag_R > tol).sum())

        # If rank is less than m, we cannot get an m x m invertible block
        if rank < m:
            return None, None, None, None

        # ---- 4. Take first m pivot columns as "other" (invertible block) ----
        other_idx = piv[:m]                       # indices for A_other (invertible block)
        all_idx = np.arange(n)
        partial_idx = np.setdiff1d(all_idx, other_idx)  # remaining columns

        A_other_np = A_np[:, other_idx]          # shape (m, m), should be full-rank
        A_partial_np = A_np[:, partial_idx]      # shape (m, n - m)

        # ---- 5. Convert back to original type (torch or numpy) ----
        A_other = torch.tensor(A_other_np, dtype=dtype, device=device)
        A_partial = torch.tensor(A_partial_np, dtype=dtype, device=device)
        
        return partial_idx, other_idx, A_partial, torch.inverse(A_other)

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        if self.A_partial == None and self.A_other_inv == None:
            print("No invertible block")
            return self.project_eq(X, Z) 
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y


    def opt_solve(self, X, solver_type='osqp', tol=1e-4):
        if solver_type == 'qpth':
            print('running qpth')
            start_time = time.time()
            res = QPFunction(eps=tol, verbose=False)(self.Q, self.p, self.G, self.h, self.A, X)
            end_time = time.time()

            sols = np.array(res.detach().cpu().numpy())
            total_time = end_time - start_time
            parallel_time = total_time
        
        elif solver_type == 'osqp':
            print('running osqp')
            Q, p, A, G, h = \
                self.Q_np, self.p_np, self.A_np, self.G_np, self.h_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(h.shape[0]) * np.inf])
                my_u = np.hstack([Xi, h])
                solver.setup(P=csc_matrix(Q), q=p, A=csc_matrix(my_A), l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time/len(X_np)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask =  ~np.isnan(Y).all(axis=1)  
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y