import numpy as np
import glob
import utils
import time

class Dataset():
    def __init__(self, args):
        self.dataset_name = args.dataset_name  # Thumos14reduced
        self.num_class = args.num_class 
        self.feature_size = args.feature_size
        self.path_to_features = '%s-%s-JOINTFeatures.npy' %(args.dataset_name, args.feature_type)
        self.path_to_annotations = self.dataset_name + '-Annotations/'
        self.features = np.load(self.path_to_features, encoding='bytes')
        self.segments = np.load(self.path_to_annotations + 'segments.npy') # [[67.5,75.9],[85.9,90.6],[139.3,148.2]]  412
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy')  # ['HighJump'],['HammerThrow'],... 每个视频包含的动作类别   # Specific to Thumos14
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy')  # 20个动作类别
        self.subset = np.load(self.path_to_annotations + 'subset.npy')  #validation/test  每个视频所属的subset
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []   # 训练视频idx
        self.testidx = []    # 测试视频idx
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs,self.classlist) for labs in self.labels]  # one hot形式的视频动作类别标注

        self.train_test_idx()  # 将不同subset的视频添加到对应列表
        self.classwise_feature_mapping()  # [[65,130,131,132,133,134,135,136,137,138,137],[],[]]  获取属于某个类的视频idx


    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == 'validation':   # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i); break;
            self.classwiseidx.append(idx)


    def load_data(self, n_similar=3, is_training=True):
        if is_training==True:
            features = []
            labels = []
            idx = []  # [10,16,111,115,117,112] 3个类别*每个类别2个视频

            # Load similar pairs
            rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)  # 随机选择3个类别id
            for rid in rand_classid:
                rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)  # 从某个类别中再随机选择一对视频
                idx.append(self.classwiseidx[rid][rand_sampleid[0]])
                idx.append(self.classwiseidx[rid][rand_sampleid[1]])

            # Load rest pairs
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size-2*n_similar)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])
          
            return np.array([utils.process_feat(self.features[i], self.t_max) for i in idx]), np.array([self.labels_multihot[i] for i in idx])

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array(feat), np.array(labs), done

