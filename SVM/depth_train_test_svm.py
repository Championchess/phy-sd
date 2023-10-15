import ipdb
import torch
import pycocotools.mask as coco_mask
from PIL import Image
import pickle
import os
import numpy as np
import torch.nn as nn
import json
import cv2
import scipy.io
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC


jobid = 0
# best combination of parameters via grid search on the val set
model = 'sd'
t = 157
up_ft_index = 1
c_value = 1


support_img_h = 427
support_img_w = 561

region_dir = '' # fill in the path to store the region annotations of the original dataset


train_feat_base_dir = '' # fill in the path to the directory of storing stable diffusion features
test_feat_base_dir = '' # fill in the path to the directory of storing stable diffusion features


train_img_name_list = json.load(open('train_img_name_list.json'))
test_img_name_list = json.load(open('test_img_name_list.json')) 


save_result_folder = '' # fill in the path to save the test results
save_result_dir = os.path.join(save_result_folder, model)
if not os.path.exists(save_result_folder):
    os.mkdir(save_result_folder)
if not os.path.exists(save_result_dir):
    os.mkdir(save_result_dir)

train_fn_2_positive_list = json.load(open('train_regions_pairs.json'))
test_fn_2_positive_list = json.load(open('test_regions_pairs.json'))



# for the train set
folder_name = 't_' + str(t) + '_up-ft-index_' + str(up_ft_index)
train_feat_dir = os.path.join(train_feat_base_dir, folder_name)


svm_vectors = []
svm_labels = []
for fn in train_img_name_list:
    print(fn)
        
    feat = torch.load(os.path.join(train_feat_dir, fn[:-4] + '.pt'))
    src_ft = feat.unsqueeze(0)
    src_ft = nn.Upsample(size=(support_img_h, support_img_w), mode='bilinear')(src_ft).squeeze(0)

    img_id = int(fn[:-4])
    region = scipy.io.loadmat(os.path.join(region_dir, 'regions_from_labels_' + str(img_id).rjust(6, '0') + '.mat'))['imgRegions']

    # map label to feature
    label_2_feat = {}
    max_label = np.unique(region).max()
    for label in np.unique(region):
        if label == 0:
            continue
        cur_mask = (region == label)
        cur_mask = torch.tensor(cur_mask).unsqueeze(0)
        cur_feat = (src_ft * cur_mask).sum(-1).sum(-1) / cur_mask.sum()
        label_2_feat[float(label)] = cur_feat

    positive_list = train_fn_2_positive_list[fn]['positive_list']
    negative_list = train_fn_2_positive_list[fn]['negative_list']
    
    # positive and negative vectors
    for item in positive_list:
        svm_vectors.append((label_2_feat[item[0]] - label_2_feat[item[1]]).tolist())
        svm_labels.append(1)
    for item in negative_list:
        svm_vectors.append((label_2_feat[item[0]] - label_2_feat[item[1]]).tolist())
        svm_labels.append(0)


# train the svm
clf = make_pipeline(Normalizer(), SVC(C=c_value, kernel='linear'))
clf.fit(svm_vectors, svm_labels)


# for the test set
test_feat_dir = os.path.join(test_feat_base_dir, folder_name)
test_vectors = []
test_labels = []
prob_vectors = []
for fn in test_img_name_list:
    print(fn)
        
    feat = torch.load(os.path.join(test_feat_dir, fn[:-4] + '.pt'))
    src_ft = feat.unsqueeze(0)
    src_ft = nn.Upsample(size=(support_img_h, support_img_w), mode='bilinear')(src_ft).squeeze(0)

    img_id = int(fn[:-4])
    region = scipy.io.loadmat(os.path.join(region_dir, 'regions_from_labels_' + str(img_id).rjust(6, '0') + '.mat'))['imgRegions']

    # map label to feature
    label_2_feat = {}
    max_label = np.unique(region).max()
    for label in np.unique(region):
        if label == 0:
            continue
        cur_mask = (region == label)
        cur_mask = torch.tensor(cur_mask).unsqueeze(0)
        cur_feat = (src_ft * cur_mask).sum(-1).sum(-1) / cur_mask.sum()
        label_2_feat[float(label)] = cur_feat

    positive_list = test_fn_2_positive_list[fn]['positive_list']
    negative_list = test_fn_2_positive_list[fn]['negative_list']

    # positive and negative vectors
    for item in positive_list:
        test_vectors.append((label_2_feat[item[0]] - label_2_feat[item[1]]).tolist())
        test_labels.append(1)
    for item in negative_list:
        test_vectors.append((label_2_feat[item[0]] - label_2_feat[item[1]]).tolist())
        test_labels.append(0)  


# get the final test results
test_preds = clf.predict(test_vectors)

test_correct_cnt = (test_preds == test_labels).sum()
total_cnt = len(test_labels)
test_acc = test_correct_cnt / total_cnt

print('test_acc: ', test_acc)



c_2_test_results = {
    'correct_cnt':float(test_correct_cnt), 'total_cnt':float(total_cnt), 
    }

test_values = clf.decision_function(test_vectors).tolist()
c_2_test_results['test_values'] = test_values
c_2_test_results['test_gts'] = test_labels
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(np.array(test_labels), np.array(test_values))
c_2_test_results['auc_score'] = auc_score 


with open(os.path.join(save_result_dir, str(jobid) + '.json'), 'w') as dump_f:
    json.dump(c_2_test_results, dump_f)
