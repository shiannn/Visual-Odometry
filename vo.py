import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
from local_feature import get_correspondences

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
                    pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def reproject_3D(self, R, t, points1_2D, points2_2D):
        proj_Mat1 = np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
        proj_Mat2 = np.concatenate((R,t), axis=1)
        
        points3D = cv.triangulatePoints(
            projMatr1=proj_Mat1,
            projMatr2=proj_Mat2,
            projPoints1=points1_2D.T,
            projPoints2=points2_2D.T    
        )
        return points3D

    def kp2array(self, keypoints):
        kp_array = np.array([kp.pt for kp in keypoints])
        return kp_array

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        ### Intrinsic K 3x3
        img_pre = cv.imread(self.frame_paths[0])
        pre_kp1 = None
        pre_kp2 = None
        pre_match_pairs = None
        pre_R = None
        pre_t = None
        for idx, frame_path in enumerate(self.frame_paths[1:], start=1):
            img = cv.imread(frame_path)
            #TODO: compute camera pose here
            points1, points2, kp1, kp2, good_matches = get_correspondences(img_pre, img, method='orb')
            kp1 = self.kp2array(kp1)
            kp2 = self.kp2array(kp2)

            es_Mat, mask = cv.findEssentialMat(points1, points2, cameraMatrix=self.K)
            print(es_Mat)
            _, R, t, _ = cv.recoverPose(es_Mat, points1, points2, cameraMatrix=self.K)
            
            match_pairs = np.array([[m.queryIdx, m.trainIdx] for m in good_matches])
            print(match_pairs.shape)
            if pre_kp1 is not None:
                print(pre_kp1.shape)
            if pre_kp2 is not None:
                print(pre_kp2.shape)
            if pre_match_pairs is not None:
                print(pre_match_pairs.shape)
                print(pre_match_pairs.T)
                print(match_pairs.T)
                _, comm1, comm2 = np.intersect1d(pre_match_pairs[:,1], match_pairs[:,0], return_indices=True)
                print(comm1)
                print(comm2)
                assert (comm1 < pre_match_pairs.shape[0]).all()
                assert (comm2 < match_pairs.shape[0]).all()
                print(pre_match_pairs[comm1], pre_match_pairs[comm1,0], pre_kp1.shape)
                print(pre_kp1[pre_match_pairs[comm1,0]])
                print(pre_kp2[pre_match_pairs[comm1,1]])
                print(kp1[match_pairs[comm2,0]])
                print(kp2[match_pairs[comm2,1]])
                #print(match_pairs[comm2])
            ### rescale t
            points3D = self.reproject_3D(R, t, points1, points2)
            ### points3D 3xN

            if idx == 3:
                exit(0)
            queue.put((R, t))
            
            pre_kp1 = kp1
            pre_kp2 = kp2
            pre_match_pairs = match_pairs
            pre_R = R
            pre_t = t
            cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break

            img_pre = img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
