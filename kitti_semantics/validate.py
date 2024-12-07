import numpy as np
import laspy as lp
from tool import DataProcessing as DP
from helper_ply import write_ply, read_ply
from tqdm import tqdm
import matplotlib.pyplot as plt

from labels import id2label
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix
files = ['instance1_2.ply','instance2_2.ply','instance3_2.ply']
for file in files:
    path = '/home/mspl/kitti_semantics/test/Log_2022-10-19_14-29-04_notfinetuned/test_preds2/' + file

    xyz, rgb, labels = DP.read_ply_data(path, with_rgb=True, with_label=True)

    _, _, labels_gt = DP.read_ply_data('/home/mspl/kitti_semantics/Dataset/kitti/train/' + file, with_rgb=True, with_label=True)
    print(file)

    correct = np.sum(labels == labels_gt)

    conf_matrix = confusion_matrix(labels_gt, labels, labels=np.arange(0, 15, 1), normalize='true',)
    gt_classes = np.sum(conf_matrix, axis=1)
    positive_classes = np.sum(conf_matrix, axis=0)
    true_positive_classes = np.diagonal(conf_matrix)

    iou_list = []
    for n in range(0, 15, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(15)

    print(iou_list)
    print('miou = ', mean_iou)
    # print(conf_matrix)
    print('acc = ', correct/len(labels_gt))

    cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=['road',
                                    'sidewalk',
                                    'building',
                                    'wall',
                                    'fence',
                                    'pole',
                                    'traffic light',
                                    'traffic sign',
                                    'vegetation',
                                    'terrain',
                                    'person',
                                    'car',
                                    'truck',
                                    'motorcycle',
                                    'bicycle'
                                    ])
    # plt.figure(figsize=(30, 30))

    cmd.plot(cmap = 'Blues', values_format = '.2f', xticks_rotation = 'vertical')
    cmd.ax_.set(xlabel='Predicted', ylabel='True')
    plt.show()