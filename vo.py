import open3d as o3d
import numpy as np
import scipy
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from local_feature import get_correspondences
from plot_camera import plot_camera_object

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
                    #print(R)
                    #print(t)
                    line_set, _ = plot_camera_object(R,t)
                    vis.add_geometry(line_set)
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def reproject_3D(self, k, R, t, points1_2D, points2_2D):
        extrinsic1 = np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        proj_Mat1 = np.matmul(k, extrinsic1)
        extrinsic2 = np.concatenate((R,t), axis=1)
        proj_Mat2 = np.matmul(k, extrinsic2)
        
        points4D = cv.triangulatePoints(
            projMatr1=proj_Mat1,
            projMatr2=proj_Mat2,
            projPoints1=points1_2D.T,
            projPoints2=points2_2D.T    
        )
        #print(points4D.shape)
        # homogeneous to non-homogeneous
        points3D = points4D / points4D[3,:]
        return points3D.T

    def kp2array(self, keypoints):
        kp_array = np.array([kp.pt for kp in keypoints])
        return kp_array

    def get_rescale_ratio(self, pre_points3D, points3D):
        assert points3D.shape[1] == 4 and pre_points3D.shape[1] == 4
        ### points3D (homogeneous) Nx4
        min_idx = min(pre_points3D.shape[0], points3D.shape[0])
        pre_points3D = pre_points3D[:min_idx]
        points3D = points3D[:min_idx]
        
        #sel = np.random.randint(low=0,high=pre_points3D.shape[0], size=2)
        ### ratio
        #print(pre_t, np.linalg.norm(pre_t,ord=2))
        #print(t, np.linalg.norm(t,ord=2))
        # print(pre_points3D.shape)
        # print(points3D.shape)
        pd3D_pre = scipy.spatial.distance.pdist(pre_points3D)
        pd3D = scipy.spatial.distance.pdist(points3D)
        #print(pd3D_pre.mean())
        #print(pd3D.mean())
        ratio = np.median(pd3D) / np.median(pd3D_pre)
        #print(pd3D/pd3D_pre)
        #ratio = np.median(pd3D/pd3D_pre)
        
        #ratio = np.median(pd3D) / np.median(pd3D_pre)
        # pre_sel_pts = pre_points3D[sel,:3]
        # sel_pts = points3D[sel,:3]
        # lower = np.linalg.norm(pre_sel_pts[0]-pre_sel_pts[1], ord=2)
        # upper = np.linalg.norm(sel_pts[0]-sel_pts[1], ord=2)
        # ratio = upper / lower
        return ratio
                
    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        ### Intrinsic K 3x3
        #img_pre = cv.imread(self.frame_paths[0])
        pre_norm = 1.0
        for idx, _ in enumerate(self.frame_paths[1:], start=1):
            pre_frame = cv.imread(self.frame_paths[idx-1])
            cur_frame = cv.imread(self.frame_paths[idx])
            pos_frame = cv.imread(self.frame_paths[idx+1])

            #TODO: compute camera pose here
            points01_pre, points01_cur, _, _, good_matches01 = get_correspondences(pre_frame, cur_frame, method='orb')
            es_Mat01, mask01 = cv.findEssentialMat(points01_cur, points01_pre, cameraMatrix=self.K, method=cv.RANSAC)
            _, R01, t01, _ = cv.recoverPose(es_Mat01, points01_cur, points01_pre, cameraMatrix=self.K)

            points12_cur, points12_pos, _, _, good_matches12 = get_correspondences(cur_frame, pos_frame, method='orb')
            es_Mat12, mask12 = cv.findEssentialMat(points12_cur, points12_pos, cameraMatrix=self.K, method=cv.RANSAC)
            _, R12, t12, _ = cv.recoverPose(es_Mat12, points12_cur, points12_pos, cameraMatrix=self.K)
            
            #match_pairs = np.array([[m.queryIdx, m.trainIdx] for m in good_matches])

            ### rescale t
            #ratio = 1.01 if np.random.rand() > 0.5 else 0.99
            if True:
                points3D01 = self.reproject_3D(
                    self.K, R01, t01,
                    points01_cur,
                    points01_pre
                )

                points3D12 = self.reproject_3D(
                    self.K, R12, t12,
                    points12_cur,
                    points12_pos
                )
                ratio = self.get_rescale_ratio(points3D01, points3D12)    
            print(ratio)
            ### accumulate scale
            dt = ratio* pre_norm* t12
            R = np.matmul(R12,R)
            t = np.matmul(R12, t) + dt
            
            queue.put((R, t))

            img = cur_frame
            pre_norm = np.linalg.norm(dt, ord=2)
            cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
