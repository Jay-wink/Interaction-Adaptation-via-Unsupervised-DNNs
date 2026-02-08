import numpy as np
from scipy.spatial import Delaunay        
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix  

# The Structure of BVH
# Root Channel (1. A position vector 2.Euler Angles, Intrinsic Rotation) Child (Joint/End Site)
class Offset:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

    def scaling(self, scaling_factor):
        self.x *= scaling_factor
        self.y *= scaling_factor
        self.z *= scaling_factor


class RotationMatrix:
    def __init__(self, z, y, x):
        self.z = float(z)
        self.y = float(y)
        self.x = float(x)
    
    # convert three euler angles (extrinsic) to the whole rotation matrix
    def euler_angles_to_rotation_matrix(self):
        # degree to radian
        rad_z = np.deg2rad(self.z)  # Z rotation
        rad_y = np.deg2rad(self.y)  # Y rotation
        rad_x = np.deg2rad(self.x)  # X rotation

        # rotation matrix around z
        Rz = np.array([
            [np.cos(rad_z), -np.sin(rad_z), 0],
            [np.sin(rad_z),  np.cos(rad_z), 0],
            [            0,              0, 1]
        ])

        # rotation matrix around y
        Ry = np.array([
            [ np.cos(rad_y), 0, np.sin(rad_y)],
            [             0, 1,             0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])

        # rotation matrix around x
        Rx = np.array([
            [1,         0,          0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x),  np.cos(rad_x)]
        ])

        # Total rotation matrix
        R = Rz @ Ry @ Rx
        return R

class Joint:
    def __init__(self, name="", length=0, is_root=False, children=None, offset=None, R=None):
        self.name = name
        self.length = length
        self.is_root = is_root
        # distribute every joint its own children list
        self.children = [] if children is None else children
        self.offset = Offset(0, 0, 0) if offset is None else offset
        self.R = RotationMatrix(0, 0, 0).euler_angles_to_rotation_matrix() if R is None else R


class BVH:
    def __init__(self, filepath, scaling_factor=1.0, positional_constraint=1):
        self.hierarchy_start = 0
        self.hierarchy_exist = False
        self.motion_start = 0
        self.motion_exist = False
        self.parent_child_pairs = []
        self.all_frames = []
        self.channel_num = 0
        self.frame_num = 0
        self.radius = 2
        self.root = None
        self.scaling_factor = scaling_factor
        self.positional_constraint=positional_constraint

        with open(filepath, 'r') as f:
            self.lines = f.readlines()

        for i, line in enumerate(self.lines):
            if self.hierarchy_exist and self.motion_exist:
                break
            if 'HIERARCHY' in line:
                self.hierarchy_start = i
                self.hierarchy_exist = True

            if 'MOTION' in line:
                self.motion_start = i
                self.frame_num_line = self.lines[self.motion_start + 1].strip().split(":")
                self.frame_num = int(self.frame_num_line[1].strip())
                self.fps_line = self.lines[self.motion_start + 2].strip().split(":")
                self.fps = int(round((1.0 / float(self.fps_line[1].strip())), 1))
                self.motion_exist = True

        if not self.hierarchy_exist or not self.motion_exist:
            print("Wrong Format Error: No HIERARCHY or MOTION part exists.")

        self.all_frames = [{} for _ in range(self.frame_num)]

        self.root_line = self.lines[self.hierarchy_start + 1].strip().split()
        if self.root_line[0] != "ROOT":
            print("Wrong Format Error: HIERARCHY does not begin with ROOT.")
        
    def add_offset(self, offset1: Offset, offset2: Offset) -> Offset:
        return Offset(round(offset1.x + offset2.x, 6), round(offset1.y + offset2.y, 6), round(offset1.z + offset2.z, 6))


    def check_offset(self, offset_line):
        if offset_line[0] != "OFFSET":
            print("Error: Wrong Format, OFFSET")
        return

    def check_channels(self, channels_line):
        if channels_line[0] != "CHANNELS" or int(channels_line[1]) != len(channels_line) - 2:
            print("Error: Wrong Format, CHANNELS")
        return

    def parse_hierarchy(self, i, parent):
        node = Joint()
        

        joint_line = self.lines[i].strip().split()
        offset_line = self.lines[i+2].strip().split()
        self.check_offset(offset_line) 

        offset = Offset(offset_line[1], offset_line[2], offset_line[3])
        node.offset = offset
        node.offset.scaling(self.scaling_factor)
        node.name = joint_line[1]

        if joint_line[0] == "ROOT":
            node.is_root = True

        else:
            virtual_point = Joint()
            virtual_point.name = "VP_" + parent.name + "_" + node.name
            v_offset = self.get_virtual_offset(node.offset.position, self.radius)
            virtual_point.offset = Offset(v_offset[0], v_offset[1], v_offset[2])

            if joint_line[0] == "End" and joint_line[1] == "Site":
                node.length = 4
                node.name = f"End Site of {parent.name}"
                virtual_point.name = "VP_" + parent.name + "_" + node.name
                self.parent_child_pairs.insert(0, (parent.name, virtual_point.name))
                parent.children.append(virtual_point)
                return node
                    
        child_node = self.parse_hierarchy(i+4, node)
        node.children.append(child_node) # child node
        self.parent_child_pairs.insert(0, (node.name, child_node.name))

        next_child_line_num = i + 4 + child_node.length
        next_child_line = self.lines[next_child_line_num].strip()
        while (next_child_line != "}"):
            child_node = self.parse_hierarchy(next_child_line_num, node)
            node.children.append(child_node)
            self.parent_child_pairs.insert(0, (node.name, child_node.name))
            next_child_line_num += child_node.length
            next_child_line = self.lines[next_child_line_num].strip()

        if joint_line[0] != "ROOT":
            self.parent_child_pairs.insert(0, (parent.name, virtual_point.name))
            parent.children.append(virtual_point)

        sum = 0
        for child_node in node.children:
            sum += child_node.length

        node.length = 5 + sum

        return node

    def parse_motion(self, frame, joint, parent_pos, parent_R, cnt):
        channels_line = self.lines[frame + self.motion_start + 3].strip().split()

        R = RotationMatrix(channels_line[3*cnt], channels_line[3*cnt + 1], channels_line[3*cnt + 2]).euler_angles_to_rotation_matrix()

        if joint.is_root:
            parent_pos = self.scaling_factor*Offset(channels_line[0], channels_line[1], channels_line[2]).position 
            joint.R = R
            parent_R = RotationMatrix(0, 0, 0).euler_angles_to_rotation_matrix()
        else:
            joint.R = parent_R @ R

        joint_position = parent_pos + parent_R @ joint.offset.position

        self.all_frames[frame][joint.name] = joint_position

        for child in joint.children:
            if child.name.startswith("End Site") or child.name.startswith("VP"):
                end_pos = joint_position + joint.R @ child.offset.position       
                self.all_frames[frame][child.name] = end_pos
                continue

            cnt += 1
            cnt = self.parse_motion(frame, child, joint_position, joint.R, cnt)
        return cnt

    def get_virtual_offset(self, joint_pos, radius):
        axis = (joint_pos) / np.linalg.norm(joint_pos)
        ref = np.array([0, 0, 1])
        if abs(np.dot(axis, ref)) > 0.98:
            ref = np.array([1, 0, 0])
        
        dir_perp = np.cross(axis, ref)
        dir_perp = dir_perp / np.linalg.norm(dir_perp)

        mid_point = joint_pos * 0.5
        vp_offset = mid_point + dir_perp * radius
        
        return vp_offset

    def parse(self):
        self.root = self.parse_hierarchy(self.hierarchy_start + 1, None)

        # create FRAME_NUM number of all_points arrays
        for frame in range(self.frame_num):
            self.parse_motion(frame, self.root, None, None, cnt=1)