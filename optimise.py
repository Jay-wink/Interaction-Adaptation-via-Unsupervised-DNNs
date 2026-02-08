import numpy as np
from scipy.spatial import Delaunay   
from scipy.sparse import vstack as sp_vstack
from numpy.linalg import solve
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, vstack, hstack, identity, kron, bmat, block_diag, issparse
from scipy.sparse.linalg import spsolve
import bvh2IM
import matplotlib.pyplot as plt

class Interaction:
    def __init__(self, num_frames, all_frames, bones, bones_idx, tet_edges):
        self.num_frames = num_frames
        self.all_frames = all_frames
        self.bones = bones
        self.bones_idx = bones_idx
        self.tet_edges = tet_edges

#=============================#
#       Helper Functions      #
#=============================#
def flatten(V):      # V shape (n,3) -> (3n,)
    return V.reshape(-1)

def unflatten(v1d):  # (3n,) -> (n,3)
    return v1d.reshape(-1, 3)

# build the large matrix of hard constraints
def build_diagonal_matrix(blocks, dtype=None, format='csr'):
    """
    blocks: list of matrices
    return: sparse block-diagonal matrix
    """
    sparse_blocks = [
        # if input matrices are not sparse, convert them to be sparse
        csr_matrix(b) if not issparse(b) else b
        for b in blocks
    ]

    return block_diag(sparse_blocks, format=format, dtype=dtype)

def build_vertical_matrix(blocks):
    """
    blocks: list of matrices
    return: vertical stacking result [[M1], [M2], [M3] ...]
    """
    return np.block([[b] for b in blocks])    

def build_Nx1_matrix(mats, dtype=None):
    arrs = [np.atleast_2d(np.asarray(a)) for a in mats]
    return np.hstack([a.T for a in arrs]).T


""" Laplacian Energy """
def _build_random_walk_laplacian(pts_raw_i, tet_edges_i, eps=1e-8):
    """
    tet_edges_i : point pairs that connected with an edge defined by Delaunay Tetrahedralization in frame i
    pts_raw_i   : all points in frame i
    eps         : a small number to avoid distance = 0
    """
    # n: number of points in frame i
    n = pts_raw_i.shape[0]

    adj = defaultdict(set)
    for a, b in tet_edges_i:
        adj[a].add(b)
        adj[b].add(a)

    L = lil_matrix((n, n), dtype=float)

    for i in range(n):
        neighbors = list(adj[i])

        # the point with no neighbours
        if len(neighbors) == 0:
            L[i, i] = 1.0
            continue

        # reciprocals of distance between point pairs
        inv_dists = []
        for j in neighbors:
            d = np.linalg.norm(pts_raw_i[i] - pts_raw_i[j])
            inv_d = 1.0 / max(d, eps)   # avoid d=0
            inv_dists.append(inv_d)

        inv_dists = np.array(inv_dists, dtype=float)
        inv_sum = inv_dists.sum()

        # normalization
        weights = inv_dists / inv_sum

        # Random-walk Laplacian: L = I - W_rw
        L[i, i] = 1.0
        for j, w_ij in zip(neighbors, weights):
            L[i, j] = -w_ij

    return L.tocsr()

def build_laplacian_weight_matrix(pts_raw_i, tet_edges_i):
    L  = _build_random_walk_laplacian(pts_raw_i, tet_edges_i)   # N × N
    I3 = identity(3, format='csr')                       # 3 × 3 Identity
    M  = kron(L, I3, format='csr')
    return M


def get_orignial_laplacian_coordinates_matrix(laplacian_weight_matrix, all_points_at_iteration_n):
    return laplacian_weight_matrix @ all_points_at_iteration_n.reshape(-1, 1)

def compute_laplacian_energy(V_updated, laplacian_weight_matrix, V):
    L  = laplacian_weight_matrix
    diff = L @ (V_updated - V)
    return 0.5 * diff @ diff 

""" Soft Constraint """

### Positional Constraint ###
def get_points_dict(all_frames, i, fixed_point_keys):
    """
    Get names of fixed points and return a dictionary whose keys are the names of fixed points
    and values are their positions
    """
    frame_dict = all_frames[i]
    return {
        key: frame_dict[key].copy()
        for key in frame_dict.keys()    # same order as them in the frame
        if key in fixed_point_keys      
    }

def convert_target_position_matrix(target_pos):
    return np.vstack(target_pos.reshape(-1, 1))

def get_positional_weight_matrix(point_name, frame_n):
    num_target_pos = len(point_name)
    num_points   = len(frame_n) 

    W = np.zeros((num_target_pos * 3, num_points * 3))

    name_to_index = {name: i for i, name in enumerate(frame_n.keys())}

    for row_id, name in enumerate(point_name):
        if name not in name_to_index:
            raise ValueError(f"Point name '{name}' not found in all_frame[0]")
        col_idx = name_to_index[name]

        W[row_id * 3 + 0, col_idx * 3 + 0] = 1
        W[row_id * 3 + 1, col_idx * 3 + 1] = 1
        W[row_id * 3 + 2, col_idx * 3 + 2] = 1

    return W

    
def compute_positional_energy(V_updated, positional_weight_matrix, target_position_matrix):
    K = positional_weight_matrix
    P = target_position_matrix
    part1 = 0.5 * V_updated.transpose() @ K.transpose() @ K @ V_updated
    part2 = P.transpose() @ K @ V_updated
    part3 = 0.5 * P.transpose() @ P
    positional_energy_matrix = part1 - part2 + part3
    return positional_energy_matrix.item()

### Penetration Constraint ###
def _dis_between_two_segments(a0, a1, b0, b1, eps=1e-12):
    u = a1 - a0
    v = b1 - b0
    w0 = a0 - b0
    a = u.dot(u) + eps
    b = u.dot(v)
    c = v.dot(v) + eps
    d = u.dot(w0)
    e = v.dot(w0)
    denom = a*c - b*b + eps

    t = (b*e - c*d) / denom
    uparam = (a*e - b*d) / denom
    t = np.clip(t, 0.0, 1.0)
    uparam = np.clip(uparam, 0.0, 1.0)

    P = a0 + t * u
    Q = b0 + uparam * v
    diff = P - Q
    dist = np.linalg.norm(diff)
    return t, uparam, P, Q, dist

def _pair_is_adjacent(pair_i, pair_j):
    return (pair_i[0] == pair_j[0] or pair_i[0] == pair_j[1] or
            pair_i[1] == pair_j[0] or pair_i[1] == pair_j[1])

def make_bone_radii(bones_idx, default_r=2.0):
    R = np.full(len(bones_idx), default_r, dtype=float)
    return R

def build_penetration_matrix(bones_idx, radii, unflatten_V, skip_adjacent = True):
    num_bones = len(bones_idx)
    N = unflatten_V.shape[0]
    rows, cols, data = [], [], []
    f_list = []

    for i in range(num_bones):
        pA, cA = bones_idx[i]
        A0, A1 = unflatten_V[pA], unflatten_V[cA]
        rA = radii[i]

        for j in range(i+1,num_bones):
            if skip_adjacent and _pair_is_adjacent(bones_idx[i], bones_idx[j]):
                continue

            pB, cB = bones_idx[j]
            B0, B1 = unflatten_V[pB], unflatten_V[cB]
            rB = radii[j]
            t, u, P, Q, dist = _dis_between_two_segments(A0, A1, B0, B1)
            r_sum = rA + rB
            if dist >= r_sum:        # no penetration
                continue
            
            if dist < 1e-8:
                ref = (A1 - A0)
                if np.linalg.norm(ref) < 1e-8: ref = (B1 - B0)
                if np.linalg.norm(ref) < 1e-8: ref = np.array([1.0,0.0,0.0])
                n = ref / (np.linalg.norm(ref) + 1e-12)
            else:
                n = (P - Q) / dist

            depth = r_sum - dist

            target_vec = (P - Q) + depth * n
            row_base = 3 * (len(f_list) // 3)

            for axis in range(3):
                r = row_base + axis

                # + (1 - t) * A0[axis]
                rows.append(r); cols.append(3*pA + axis); data.append(1.0 - t)
                # + t * A1[axis]
                rows.append(r); cols.append(3*cA + axis); data.append(t)
                # - (1 - u) * B0[axis]
                rows.append(r); cols.append(3*pB + axis); data.append(-(1.0 - u))
                # - u * B1[axis]
                rows.append(r); cols.append(3*cB + axis); data.append(-u)

            f_list.extend(target_vec.tolist())

            
    if not f_list:
        # no collision
        return None, None

    J = coo_matrix((data, (rows, cols)), shape=(len(f_list), 3*N)).tocsr()
    f = np.asarray(f_list, dtype=float)
    return J, f


""" Hard Constraint (Equality or Inequality)"""

### Bone-length Constraint ###
def build_bone_length_matrix(bones_idx, unflatten_V):
    V = np.asarray(unflatten_V)
    N = V.shape[0]
    B = len(bones_idx)
    M = np.zeros((B, 3*N), dtype=float)

    eps = 1e-12
    for i, (p, c) in enumerate(bones_idx):
        d = V[c] - V[p]          
        L = np.linalg.norm(d)    
        if L < eps:              
            continue
        u = d / L                
        M[i, 3*p:3*p+3] = -u     
        M[i, 3*c:3*c+3] = u     
    return M

def build_bone_length_vector(bones_idx, unflatten_V):
    V = np.asarray(unflatten_V)
    eps = 1e-12
    lengths = []  
    for i, (p, c) in enumerate(bones_idx):
        d = V[c] - V[p]          
        L = np.linalg.norm(d)    
        if L < eps:              
            print("error in Bone-Length")
        lengths.append([L])

    return np.asarray(lengths, dtype=float)  # (B,1)

def initialization(bvh_file0, bvh_file1):
    bvh0 = bvh2IM.BVH(bvh_file0)
    bvh1 = bvh2IM.BVH(bvh_file1)
    bvh0.parse()
    bvh1.parse()

    # check if number of frames of two characters' motion is equal
    num_frames = bvh0.frame_num if bvh0.frame_num == bvh1.frame_num else print("error")

    all_frames = []                      # merged result
    tet_edges = []
    for i in range(len(bvh0.all_frames)):
        bvh0.all_frames[i] = {key + "_A": value for key, value in bvh0.all_frames[i].items()}
        bvh1.all_frames[i] = {key + "_B": value for key, value in bvh1.all_frames[i].items()}
        
    # No Need to add suffix?
    for f in range(num_frames):
        merged = {}

        # --- character A: add suffix 'A' ---
        merged.update({k : v
                    for k, v in bvh0.all_frames[f].items()})

        
        # --- character B: add suffix 'B' ---
        merged.update({k: v
                    for k, v in bvh1.all_frames[f].items()})
        
        all_frames.append(merged)

        # Delaunay Tetrahedralizatoin to build neighbour relationship for every frame
        pts_raw_f = np.vstack(list(all_frames[f].values()))
        tet_f = Delaunay(pts_raw_f)
        edge_set_f = set()
        for a,b,c,d in tet_f.simplices:
            edge_set_f.update([tuple(sorted(e)) for e
                            in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]])
        tet_edges_f = sorted(edge_set_f) 
        tet_edges.append(tet_edges_f)
    
    # build bone index array
    bvh0.parent_child_pairs = [(k + "_A", v + "_A") for k, v in bvh0.parent_child_pairs]
    bvh1.parent_child_pairs = [(k + "_B", v + "_B") for k, v in bvh1.parent_child_pairs]
    all_pairs = bvh0.parent_child_pairs + bvh1.parent_child_pairs
    bones = [pair for pair in all_pairs if "VP" not in pair[0] and "VP" not in pair[1]]
    names = list(all_frames[50].keys())
    index_of = {name: i for i, name in enumerate(names)}
    bones_idx = []
    for p, c in bones:
        bones_idx.append((index_of[p], index_of[c]))

    # store all into an Interaction object
    return Interaction(num_frames, all_frames, bones, bones_idx, tet_edges)

