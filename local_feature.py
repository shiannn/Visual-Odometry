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
    if method == 'orb':
        ft_extractor = cv.ORB_create()
    elif method == 'sift':
        ft_extractor = cv.SIFT_create()
    else:
        print('invalid method')
        exit(0)
    kp1, des1 = ft_extractor.detectAndCompute(img1,None)
    kp2, des2 = ft_extractor.detectAndCompute(img2,None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    return points1, points2, kp1, kp2, good_matches

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