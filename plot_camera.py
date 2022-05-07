import numpy as np
import pandas as pd
import open3d as o3d

def get_camera(focal_len_scaled=0.5, aspect_ratio=0.3):
    points = np.array([
        [0, 0, 0],
        [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled],
        [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled],
        [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled],
        [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled]
    ])
    return points

def plot_camera_object(Rotation_Mat, Translation, color=[1.0,0.,0.]):
    #print(np.matmul(Rotation_Mat, points.T).T+Translation.T)
    points = get_camera()
    #points = np.matmul(Rotation_Mat, points.T).T + Translation.T
    #print(points.T.shape, Translation.shape, (points.T - Translation).shape)
    points = np.matmul(Rotation_Mat.T, (points.T - Translation)).T
    #print(points.shape)
    
    #points = np.matmul(Rotation_Mat.T,(points-Translation).T).T
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    #points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    #lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [color for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set, points