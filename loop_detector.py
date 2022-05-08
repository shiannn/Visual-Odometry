import cv2
import numpy as np
import scipy.cluster.vq as vq
from sklearn.metrics.pairwise import cosine_similarity

class LoopDetector(object):
    def __init__(self, dictionary_size, threshold, window=10):
        self.dictionary_size = dictionary_size
        self.bow = cv2.BOWKMeansTrainer(self.dictionary_size)

        self.history = None
        self.threshold = threshold
        #self.init_des = []

        self.window = window
    
    def init_dictionary(self, init_des):
        #self.init_des = np.vstack(init_des)
        self.bow.add(init_des.astype(np.float32))
        self.dictionary = self.bow.cluster()

    def extract_feature(self, in_desriptors):
        code, dist = vq.vq(in_desriptors, self.dictionary)
        hist, bins = np.histogram(code, bins=np.arange(0,self.dictionary.shape[0]+1))
        #print(hist)
        return hist
    
    def detect(self, feature):
        if self.history is None or self.history.shape[0] <= self.window:
            return None, np.array([False])
        else:
            pdist = cosine_similarity(np.expand_dims(feature, axis=0), self.history[:-self.window])
            is_loop = (pdist.max(axis=1)>self.threshold)
            return pdist.max(axis=1), is_loop

    def add2history(self, feature):
        if self.history is None:
            self.history = np.expand_dims(feature, axis=0)
        else:
            self.history = np.concatenate((self.history, np.expand_dims(feature, axis=0)))
    
if __name__ == '__main__':
    des01_pre = np.load('des01_pre.npy')
    des01_cur = np.load('des01_cur.npy')

    ld = LoopDetector(dictionary_size = 400)
    ld.init_dictionary(init_des = des01_pre)
    feat01_pre = ld.extract_feature(des01_pre)
    ld.add2history(feat01_pre)
    feat01_cur = ld.extract_feature(des01_cur)
    ld.add2history(feat01_cur)
    print(ld.history)
    pdist, is_loop = ld.detect(feat01_cur)
    print(pdist, is_loop)

    # des01_pre = des01_pre.astype(np.float32)
    # ld.bow.add(des01_pre)
    # dictionary = ld.bow.cluster()
    # print(dictionary.shape)
    # code, dist = vq.vq(des01_cur, dictionary)
    # print(code.shape)
    # print(dist.shape)
    # hist, bins = np.histogram(code, bins=np.arange(0,dictionary.shape[0]+1))
    # print(hist)
    # print(hist.shape)