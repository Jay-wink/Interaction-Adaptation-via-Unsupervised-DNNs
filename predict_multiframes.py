import torch
import pickle
import os
import numpy as np
import sys
from method import NNSolver
import optimise
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# find model path
base_dir = 'results/InteractionMesh-198-0-64-1000/method'  
subfolder = '85eef291e525d6d099f54e46bb8ea0134790c3bf'
subsubfolder = '1764612326-1891706'
save_dir = os.path.join(base_dir, subfolder, subsubfolder)

# load args and datasets
with open(os.path.join(save_dir, 'args.dict'), 'rb') as f:
    args = pickle.load(f)
with open(os.path.join('datasets', 'im', 'random_bonelength_im_dataset_fix3_sz0.75_1.25frame98_102_ex1000'), 'rb') as f:
    data = pickle.load(f)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for attr in dir(data):
    var = getattr(data, attr)
    if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
        try:
            setattr(data, attr, var.to(DEVICE))
        except AttributeError:
            pass
data._device = DEVICE

# load model
solver_net = NNSolver(data, args)
state = torch.load(os.path.join(save_dir, 'solver_net.dict'), map_location=DEVICE)
solver_net.load_state_dict(state)
solver_net.to(DEVICE)
solver_net.eval()

# prepare new data
id = 178
original_size = 1.25
sf_A = 1.0
sf_B = 0.75
bvh_file0 = os.path.join('datasets', 'simple', 'im_datasets','0{}_{}_retarget_C0.bvh'.format(id, original_size))
bvh_file1 = os.path.join('datasets', 'simple', 'im_datasets','0{}_{}_retarget_C1.bvh'.format(id, original_size))
frame_start = 98
frame_end = 102
num_frames = frame_end - frame_start + 1

im = optimise.initialization(bvh_file0, bvh_file1)
num_all_frames = im.num_frames
all_frames = im.all_frames
bones_idx = im.bones_idx
bones = im.bones
tet_edges = im.tet_edges
radii  = optimise.make_bone_radii(bones_idx)


# assume that fixed points are the same across different frames
fixed_point_keys_raw = ["LeftFoot_A", "VP_LeftFoot_End Site of LeftFoot_A", "End Site of LeftFoot_A",
                "RightFoot_A", "VP_RightFoot_End Site of RightFoot_A", "End Site of RightFoot_A",
                "LeftFoot_B",  "VP_LeftFoot_End Site of LeftFoot_B", "End Site of LeftFoot_B",
                "RightFoot_B", "VP_RightFoot_End Site of RightFoot_B", "End Site of RightFoot_B"]
fixed_points_keys = list(optimise.get_points_dict(all_frames, 0, fixed_point_keys_raw).keys())

# remove the fixed bones (defined by fixed points) from bones_idx and bones
bones_idx_cons = bones_idx.copy()
bones_cons = bones.copy()
for p1, p2 in bones_cons:
    if p1 in fixed_points_keys and p2 in fixed_points_keys:
        idx = bones_cons.index((p1, p2))
        # remove this bone
        del bones_idx_cons[idx]
        del bones_cons[idx]
# build scaling matrix
scaling_matrix = np.zeros(shape=(len(bones_idx_cons), 1))
bone_end_A = int(len(bones_idx_cons) / 2)
scaling_matrix[:bone_end_A, :] = sf_A
scaling_matrix[bone_end_A:, :] = sf_B
ones_shape_scaling_matrix = np.ones(shape=(len(bones_idx_cons), 1))

hard_constant_list = []
for i in range(frame_start, frame_end+1):
    pts_raw_i = np.vstack(list(all_frames[i].values()))

    fixed_points = optimise.get_points_dict(all_frames, i, fixed_point_keys_raw)
    fixed_points_val  = np.vstack(list(fixed_points.values()))

     # The endpoints that make up bones
    pts_a = pts_raw_i[np.array(bones_idx_cons)[:, 0]]   # shape (B, 3)
    pts_b = pts_raw_i[np.array(bones_idx_cons)[:, 1]]   # shape (B, 3)

    # calculate bone lengths matrix [b1, b2, ..., bn]^T
    bone_lengths = np.linalg.norm(pts_a - pts_b, axis=1)   # shape (B,)
    bone_lengths = bone_lengths.reshape(-1, 1)
    
    H_bone = optimise.build_bone_length_matrix(bones_idx_cons, pts_raw_i)
    
    fp = optimise.flatten(fixed_points_val).reshape(-1, 1)
    bones_scaled = (H_bone @ pts_raw_i.flatten()).reshape(-1, 1) + (scaling_matrix - ones_shape_scaling_matrix) * bone_lengths
    
    hard_constant = np.block([[fp],
                              [bones_scaled]])

    # append matrices to corresponding list
    hard_constant_list.append(hard_constant)
  
# integrate small matrices into the big matrix
big_hard_constant = optimise.build_vertical_matrix(hard_constant_list)

X_new = np.transpose(big_hard_constant)
X_new = torch.tensor(X_new, dtype=torch.float64, device=DEVICE)

A = data.A.to(DEVICE)          # (neq, ydim)
X = X_new.to(DEVICE)           # (B, neq)


# Get new Y
with torch.no_grad():
    Y_pred = solver_net(X_new)
    y_np = Y_pred[0].detach().cpu().numpy()
    new_vertices_frames = y_np.reshape(int(y_np.shape[0] / 3), 3)
    eq_res = data.eq_resid(X_new, Y_pred)        
    eq_mean = torch.mean(torch.abs(eq_res), dim=1)
    eq_max  = torch.max(torch.abs(eq_res), dim=1)[0]
    print('eq_mean (per-sample):', eq_mean.detach().cpu().numpy())
    print('eq_max  (per-sample):', eq_max.detach().cpu().numpy())  

# export new vertices and bones_idx to Blender
export_parentdir = "data_blender"
export_subfolder = "multiframes_sample"
export_vertices_folder = os.path.join(export_parentdir, export_subfolder, "new_vertices")
export_bonesIdx_folder = os.path.join(export_parentdir, export_subfolder, "bones_idx")
export_tetEdges_folder = os.path.join(export_parentdir, export_subfolder, "tet_edges")
os.makedirs(export_vertices_folder, exist_ok=True)
os.makedirs(export_bonesIdx_folder, exist_ok=True)
os.makedirs(export_tetEdges_folder, exist_ok=True)
num_pts_one_frame = int(new_vertices_frames.shape[0] / num_frames)
count = 0
for i in range(frame_start, frame_end + 1):
    new_vertices_name = "vertices" + str(id) + "_" + "frame" + str(i) + "_" + "scaleA_" + str(sf_A) + "scaleB" + str(sf_B) + ".npy"
    bones_idx_name = "bonesIdx_frame" + str(i) 
    tet_edges_name = "tetEdges_frame" + str(i)
    np.save(os.path.join(export_vertices_folder, new_vertices_name), new_vertices_frames[count*num_pts_one_frame:(count+1)*num_pts_one_frame, :])
    print(new_vertices_frames[int(count*num_pts_one_frame):int((count+1)*num_pts_one_frame), :].shape)
    np.save(os.path.join(export_bonesIdx_folder, bones_idx_name), bones_idx)
    np.save(os.path.join(export_tetEdges_folder, tet_edges_name), tet_edges[i])
    count += 1

print("Saved vertices, bone indices and tet edges to:", os.getcwd() +"\\" + os.path.join(export_parentdir, export_subfolder))
