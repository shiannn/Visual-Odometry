import cv2 as cv
import numpy as np

def get_correspondences(img1, img2, method='orb'):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if method == 'orb':
        ft_extractor = cv.ORB_create()
    elif method == 'sift':
        ft_extractor = cv.SIFT_create()
    else:
        print('invalid method')
        exit(0)
    kp1, des1 = ft_extractor.detectAndCompute(img1,None)
    kp2, des2 = ft_extractor.detectAndCompute(img2,None)

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    good_matches = sorted(matches, key=lambda x: x.distance)

    kp1 = kp2array(kp1)
    kp2 = kp2array(kp2)
    
    return kp1, des1, kp2, des2, good_matches

def kp2array(keypoints):
    kp_array = np.array([kp.pt for kp in keypoints])
    return kp_array

def get_orb_correspondences(img1, img2):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    return kp1, kp2, des1, des2

def get_sift_correspondences(img1, img2):
    
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, kp2, des1, des2