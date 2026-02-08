import numpy as np
import pickle
import torch

import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
from utils import InteractionMesh
import optimise
from scipy.linalg import qr
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

id = 178
original_size = 1.25
size_start = 0.75
size_end = 1.25
num_fix_pts = 3
frame_n = 100
num_examples = 1000

bvh_file0 = os.path.join('interaction','0{}_{}_retarget_C0.bvh'.format(id, original_size))
bvh_file1 = os.path.join('interaction','0{}_{}_retarget_C1.bvh'.format(id, original_size))

im = optimise.initialization(bvh_file0, bvh_file1)
num_frames = im.num_frames
all_frames = im.all_frames
bones_idx = im.bones_idx
bones = im.bones
tet_edges = im.tet_edges
V_current = np.vstack(list(all_frames[frame_n].values()))
pts0_raw = np.vstack(list(all_frames[0].values()))
pts_raw_i = np.vstack(list(all_frames[frame_n].values()))
ydim = V_current.shape[0] * V_current.shape[1]

fixed_point_keys_raw = ["LeftFoot_A", "VP_LeftFoot_End Site of LeftFoot_A", "End Site of LeftFoot_A",
                    "RightFoot_A", "VP_RightFoot_End Site of RightFoot_A", "End Site of RightFoot_A",
                    "LeftFoot_B",  "VP_LeftFoot_End Site of LeftFoot_B", "End Site of LeftFoot_B",
                    "RightFoot_B", "VP_RightFoot_End Site of RightFoot_B", "End Site of RightFoot_B"]
fixed_points = optimise.get_points_dict(all_frames, 100, fixed_point_keys_raw)

fixed_points_val  = np.vstack(list(fixed_points.values()))
fixed_points_keys = list(fixed_points.keys())

# Lapalacian term
M = optimise.build_laplacian_weight_matrix(pts_raw_i, tet_edges[frame_n])
rank = np.linalg.matrix_rank(M.toarray())

b = M @ optimise.flatten(V_current)  

# Collision term
radii  = optimise.make_bone_radii(bones_idx)
J_coll, f_coll = optimise.build_penetration_matrix(bones_idx, radii, V_current, skip_adjacent=True)

# Equality constraint term
H1 = optimise.get_positional_weight_matrix(fixed_points_keys, all_frames[frame_n]) # (3p, 3m)

# build bone-length matrix
bones_idx_cons = bones_idx.copy()
bones_cons = bones.copy()
for p1, p2 in bones_cons:
    if p1 in fixed_points_keys and p2 in fixed_points_keys:
        idx = bones_cons.index((p1, p2))
        # remove this bone
        del bones_idx_cons[idx]
        del bones_cons[idx]
H2 = optimise.build_bone_length_matrix(bones_idx_cons, V_current) # (B, 3m)
pts_i = V_current[np.array(bones_idx_cons)[:, 0]]   # shape (b, 3)
pts_j = V_current[np.array(bones_idx_cons)[:, 1]]   # shape (b, 3)

bone_lengths = np.linalg.norm(pts_i - pts_j, axis=1)   # shape (b,)
bone_lengths = bone_lengths.reshape(-1, 1)
B2 = H2 @ optimise.flatten(V_current)   # (B,)

# build scaling matrix
bone_sf = np.linspace(0.7, 1.3, 1001)
scaling_matrix = np.random.choice(bone_sf, size=(len(bones_idx_cons), num_examples))
ones_shape_scaling_matrix = np.ones(shape=(len(bones_idx_cons), num_examples))
B2_scaled = B2.reshape(-1, 1) + (scaling_matrix - ones_shape_scaling_matrix) * bone_lengths # (B, num_exmaples)
fp = np.tile(optimise.flatten(fixed_points_val).reshape(-1, 1), (1, num_examples)) # (3p, num_examples)
hard_constant  = np.block([[fp], 
                           [B2_scaled]]) # (3p+B, num_exmaples)

# Inequality constraint term (Collision constraint?)

Q = M.T @ M
p = np.transpose(-b.T @ M)
A = np.block([[H1],
              [H2]]) # (B+3p, 3m)

X = np.transpose(hard_constant) # (3p+B, num_exmaples) -> (num_exmaples, B+3p)

if J_coll is not None and J_coll.shape[0] > 0:
    G = J_coll
    h = f_coll
else:
    G = np.zeros((0, ydim), dtype=np.float64)   # (0, ydim)
    h = np.zeros((0,), dtype=np.float64)     # (0,)

# TODO: to avoid memory waste, any other better way?
if hasattr(Q, "toarray"):
    Q = Q.toarray()
if hasattr(p, "toarray"):
    p = p.toarray()
if hasattr(A, "toarray"):
    A = A.toarray()
if hasattr(G, "toarray"):
    G = G.toarray()

# ---------------Testing Here--------------- #

problem = InteractionMesh(Q, p, A, G, h, X, bones_idx, radii)
problem.calc_Y()
print("Feasible samples:", problem.num)
with open("./random_bonelength_im_dataset_fix{}_szstart{}_szend{}_ex{}".format(num_fix_pts, size_start, size_end, num_examples), 'wb') as f:
    pickle.dump(problem, f)