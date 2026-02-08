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
subfolder = 'b7186039498232bffefd62954d62777bd1970881'
subsubfolder = '1770569102-5687087'
save_dir = os.path.join(base_dir, subfolder, subsubfolder)

# load args and datasets
with open(os.path.join(save_dir, 'args.dict'), 'rb') as f:
    args = pickle.load(f)
with open(os.path.join('datasets', 'random_bonelength_im_dataset_fix3_szstart0.75_szend1.25_ex1000'), 'rb') as f:
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
bvh_file0 = os.path.join('datasets', 'interaction', '0{}_{}_retarget_C0.bvh'.format(id, original_size))
bvh_file1 = os.path.join('datasets', 'interaction', '0{}_{}_retarget_C1.bvh'.format(id, original_size))

im = optimise.initialization(bvh_file0, bvh_file1)
num_frames = im.num_frames
all_frames = im.all_frames
bones_idx = im.bones_idx
bones = im.bones
tet_edges = im.tet_edges
V_current = np.vstack(list(all_frames[100].values()))

fixed_point_keys_raw = ["LeftFoot_A", "VP_LeftFoot_End Site of LeftFoot_A", "End Site of LeftFoot_A",
                    "RightFoot_A", "VP_RightFoot_End Site of RightFoot_A", "End Site of RightFoot_A",
                    "LeftFoot_B",  "VP_LeftFoot_End Site of LeftFoot_B", "End Site of LeftFoot_B",
                    "RightFoot_B", "VP_RightFoot_End Site of RightFoot_B", "End Site of RightFoot_B"]
fixed_points = optimise.get_points_dict(all_frames, 100, fixed_point_keys_raw)

fixed_points_val  = np.vstack(list(fixed_points.values()))
fixed_points_keys = list(fixed_points.keys())


bones_idx_cons = bones_idx.copy()
bones_cons = bones.copy()
for p1, p2 in bones_cons:
    if p1 in fixed_points_keys and p2 in fixed_points_keys:
        idx = bones_cons.index((p1, p2))
        # remove this bone
        del bones_idx_cons[idx]
        del bones_cons[idx]
fp = optimise.flatten(fixed_points_val).reshape(-1, 1)
H2 = optimise.build_bone_length_matrix(bones_idx_cons, V_current) # (B, 3m)

B2 = H2 @ optimise.flatten(V_current)   # (B,)

# build scaling matrix
pts_i_ori = V_current[np.array(bones_idx_cons)[:, 0]]   # shape (b, 3)
pts_j_ori = V_current[np.array(bones_idx_cons)[:, 1]]   # shape (b, 3)
bone_lengths_ori = np.linalg.norm(pts_i_ori - pts_j_ori, axis=1).reshape(-1, 1)   # shape (b,)
scaling_matrix = np.zeros(shape=(len(bones_idx_cons), 1))
bone_end_A = int(len(bones_idx_cons) / 2)
scaling_matrix[:bone_end_A, :] = sf_A
scaling_matrix[bone_end_A:, :] = sf_B
ones_shape_scaling_matrix = np.ones(shape=(len(bones_idx_cons), 1))
B2_scaled = B2.reshape(-1, 1) + (scaling_matrix - ones_shape_scaling_matrix) * bone_lengths_ori # (B, num_exmaples)

# create new X
X_new = np.transpose(np.block([[fp], 
                               [B2_scaled]]))
X_new = torch.tensor(X_new, dtype=torch.float64, device=DEVICE)

A = data.A.to(DEVICE)          # (neq, ydim)
X = X_new.to(DEVICE)           # (B, neq)

# Get new Y
with torch.no_grad():
    Y_pred = solver_net(X_new)
    y_np = Y_pred[0].detach().cpu().numpy()
    new_vertices = y_np.reshape(66,3)
    eq_res = data.eq_resid(X_new, Y_pred)        
    eq_mean = torch.mean(torch.abs(eq_res), dim=1)
    eq_max  = torch.max(torch.abs(eq_res), dim=1)[0]
    print('eq_mean (per-sample):', eq_mean.detach().cpu().numpy())
    print('eq_max  (per-sample):', eq_max.detach().cpu().numpy())  

# ============ Export for subsequent visualisation via Blender ============
# export new vertices and bones_idx to Blender
export_parentdir = "data_blender"
export_subfolder = "deformed_interaction" # change here to create different subfolders
export_dir = os.path.join(export_parentdir, export_subfolder)
os.makedirs(export_dir, exist_ok=True)
new_vertices_name = "vertices_" + str(id) + "_" + str(sf_A) + "_" + str(sf_B) + ".npy"
np.save(os.path.join(export_dir, new_vertices_name), new_vertices)
np.save(os.path.join(export_dir, "bones_idx.npy"), bones_idx)
np.save(os.path.join(export_dir, "tet_edges.npy"), tet_edges[100])
print("Saved vertices.npy and bones_idx.npy to:", os.getcwd() +"\\" + export_dir)


# ============ Numerical Testing ============
pts_i_new = new_vertices[np.array(bones_idx_cons)[:, 0]]   # shape (b, 3)
pts_j_new = new_vertices[np.array(bones_idx_cons)[:, 1]]   # shape (b, 3)
bone_lengths_new = np.linalg.norm(pts_i_new - pts_j_new, axis=1).reshape(-1, 1)   # shape (b,)

# LeftFoot_A change
print(np.sum(np.abs(V_current[22:25,:] - new_vertices[22:25,:])))
print('-------------------------')
# RightFoot_A change
print(np.sum(np.abs(V_current[30:33,:] - new_vertices[30:33,:])))
print('-------------------------')

# LeftFoot_B change
print(np.sum(np.abs(V_current[55:58,:] - new_vertices[55:58,:])))
print('-------------------------')

# RightFoot_B change
print(np.sum(np.abs(V_current[63:66,:] - new_vertices[63:66,:])))
print('-------------------------')

# ratio of deformed bone lengths to original one
print(np.divide(bone_lengths_new, bone_lengths_ori))



# ============= Simple Visualisation (for better perception use Blender)=============
def to_plot(v):  
    return np.array([v[0], -v[2], v[1]])

def visualize_pose(V66x3, bones_idx, title='Predicted Pose', split_2_chars=None, tet_edges=None):
    """
    V66x3: np.ndarray (66,3) — predicted vertices
    bones_idx: List[Tuple[int,int]] — bones
    split_2_chars: split all the vertices into 2 parts that belongs to 2 characters
    tet_edges: Tetrahedron edges
    """
    Vp = np.vstack([to_plot(p) for p in V66x3])
    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    ax.view_init(15, -70)

    if split_2_chars is None:
        ax.scatter(Vp[:,0], Vp[:,1], Vp[:,2], s=20, c='tab:red', label='Points')
    else:
        nA = split_2_chars
        ax.scatter(Vp[:nA,0], Vp[:nA,1], Vp[:nA,2], s=20, c='tab:red',  label='Char A')
        ax.scatter(Vp[nA:,0], Vp[nA:,1], Vp[nA:,2], s=20, c='tab:blue', label='Char B')

    for (i, j) in bones_idx:
        p, q = Vp[i], Vp[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c='k', lw=1)


    if tet_edges is not None:
        for (i, j) in tet_edges:
            p, q = Vp[i], Vp[j]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c='0.8', lw=0.5, alpha=0.6)

    x, y, z = Vp[:,0], Vp[:,1], Vp[:,2]
    margin = 5.0
    ax.set_xlim(x.min()-margin, x.max()+margin)
    ax.set_ylim(y.min()-margin, y.max()+margin)
    ax.set_zlim(z.min()-margin, z.max()+margin)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.show()
#visualize_pose(new_vertices, bones_idx, title='Predicted Pose', split_2_chars=33, tet_edges=None)