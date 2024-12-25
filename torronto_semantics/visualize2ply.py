import numpy as np
from tool import DataProcessing as DP
from helper_ply import write_ply, read_ply  # xuek
from tqdm import tqdm
from labels import trainId2label
from os import listdir
from os.path import isfile, join, split

# 颜色标签
# label_colors = [[81,  0, 81], 
#                 [128, 64,128],  # tree -> Green
#                 [70, 70, 70],  # building -> orange
#                 [153,153,153],  # Walls ->  darkblue
#                 [107,142, 35],  # Bridge -> black
#                 [70,130,180],  # parking -> blue
#                 [220, 20, 60],  # rail -> Magenta
#                 [0,  0,142]#,  # traffic Roads ->  grey
#                 # [89, 47, 95],  # Street Furniture  ->  DimGray
#                 # [255, 0, 0],  # cars -> red
#                 # [255, 255, 0],  # Footpath  ->  deeppink
#                 # [0, 255, 255],  # bikes -> cyan
#                 # [0, 191, 255]  # water ->  skyblue
#                 ]
# label_color = np.asarray(label_colors, dtype=np.uint8)
# 　label_color = label_color / 255.0

# 地址
# 1\读取的ply地址
labelpath = 'test/Log_2022-11-04_05-39-18/test_preds'
inputpath = 'Dataset/Toronto_3D/original_block_ply'
labelpaths = listdir(labelpath)
inputpaths = listdir(inputpath)
labelpaths.sort()
inputpaths.sort()
inputlabelpath = ['test/Log_2022-11-08_08-05-19/test_preds/L002.label']#[join(labelpath, f) for f in labelpaths if isfile(join(labelpath, f))]
inputplypath = ['Dataset/Toronto_3D/original_block_ply/L002.ply']#[join(inputpath, f) for f in inputpaths if isfile(join(inputpath, f))]
outputpath = 'test/Log_2022-11-08_08-05-19/test_preds2'
outputplypath = [join(outputpath, 'L002.ply')]#[join(outputpath, f) for f in inputpaths if isfile(join(inputpath, f))]


# print(labelpaths)
# print(inputpaths)
# print(outputplypath)

# inputplypath = ['/home/mspl/kitti_semantics/Dataset/kitti/test/0000000002_0000000282.ply']

# inputlabelpath = ['/home/mspl/kitti_semantics/test/Log_2022-06-24_04-01-47/test_preds/0000000002_0000000245.label']
                #   'test/Log_2022-05-30_15-59-32/test_preds/output_7_trans_3.label',/home/mspl/SensatUrban/test/Log_2022-05-04_15-08-37/test_preds
                #   'test/Log_2022-05-30_15-59-32/test_preds/output_9_trans_3.label',
                #   'test/Log_2022-05-30_15-59-32/test_preds/output_10_trans_3.label',
                #   'test/Log_2022-05-30_15-59-32/test_preds/output_11_trans_3.label',
                #   'test/Log_2022-05-30_15-59-32/test_preds/output_12_trans_3.label',
                #   'test/Log_2022-05-30_15-59-32/test_preds/output_13_trans_3.label']
                #   'test/Log_2022-05-02_15-07-34/test_preds/output_1.label',
                #   'test/Log_2022-05-02_15-07-34/test_preds/output_2.label',
                #   'test/Log_2022-05-02_15-07-34/test_preds/output_3.label']
                #   'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_2.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_8.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_15.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_16.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_22.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_27.label']/home/mspl/SensatUrban/test/Log_2022-05-31_05-57-28
# 2\写出的ply地址
# outputplypath = ['/home/mspl/kitti_semantics/test/Log_2022-06-24_04-01-47/test_preds/0000000002_0000000282.ply']
                #  'test/Log_2022-05-30_15-59-32/test_preds/output_7_trans_3.ply',
                #  'test/Log_2022-05-30_15-59-32/test_preds/output_9_trans_3.ply',
                #  'test/Log_2022-05-30_15-59-32/test_preds/output_10_trans_3.ply',
                #  'test/Log_2022-05-30_15-59-32/test_preds/output_11_trans_3.ply',
                #  'test/Log_2022-05-30_15-59-32/test_preds/output_12_trans_3.ply',
                #  'test/Log_2022-05-30_15-59-32/test_preds/output_13_trans_3.ply']
                # test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_2.ply',/home/mspl/SensatUrban/test/Log_2022-05-30_14-21-44
                #  'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_8.ply',/home/mspl/SensatUrban/test/Log_2022-05-30_15-59-32
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_15.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_16.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_22.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_27.ply']

# 数据处理
# """
for i in tqdm(range(len(inputplypath))):
    # print(i)
    xyz, rgb = DP.read_ply_data(inputplypath[i], with_rgb=True, with_label=False)  # xyz.astype(np.float32), rgb.astype(np.uint8)
    print(len(xyz))
    label = np.fromfile(inputlabelpath[i], dtype=np.uint8)
    print(len(label))
    for j in tqdm(range(len(label))):
        
        # rgb[j] = label_colors[label[j]]
        rgb[j] = np.array(trainId2label[label[j]].color)
    # print(rgb)
    write_ply(outputplypath[i], [xyz, rgb, label], ['x', 'y', 'z', 'red', 'green', 'blue', 'scalar_Label'])  # xuek [xyz, rgb, label]
    # write_ply(outputplypath[i], [xyz, label], ['x', 'y', 'z', 'class'])
# """

# for i in tqdm(range(len(inputplypath))):
#     # print(i)
#     xyz, rgb, label = DP.read_ply_data(inputplypath[i], with_rgb=True, with_label=True)  # xyz.astype(np.float32), rgb.astype(np.uint8)
#     print(len(xyz))
#     # label = np.fromfile(inputlabelpath[i], dtype=np.uint8)
#     print(len(label))
#     for j in tqdm(range(len(label))):
        
#         # rgb[j] = label_colors[label[j]]
#         if label[j] == 15:
#             label[j] = 0
#         rgb[j] = np.array(trainId2label[label[j]].color)
#     # print(rgb)
#     write_ply(outputplypath[i], [xyz, rgb, label], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])  # xuek [xyz, rgb, label]
#     # write_ply(outputplypath[i], [xyz, label], ['x', 'y', 'z', 'class'])

# import zlib
# from array import array

# compressed_binary_file = open('./val_label/test50M_down20.bin.pcd (1).zlib', 'rb').read()
# # print(compressed_binary_file)
# binary_content = zlib.decompress(compressed_binary_file)
# labels = array('B', binary_content);
# label = np.asarray(labels)

# inputplypath = './validationTT1.ply'
# # inputlabelpath = ''
# outputplypath = './groundtruth.ply'

# xyz, rgb = DP.read_ply_data(inputplypath, with_rgb=True, with_label=False)  # xyz.astype(np.float32), rgb.astype(np.uint8)
# print(len(xyz))
# # label = np.fromfile(inputlabelpath, dtype=np.uint8)
# print(len(label))
# for j in tqdm(range(len(label))):
    
#     # rgb[j] = label_colors[label[j]]
#     rgb[j] = np.array(trainId2label[label[j]].color)
# # print(rgb)
# write_ply(outputplypath, [xyz, rgb, label], ['x', 'y', 'z', 'red', 'green', 'blue', 'semantic'])  # xuek [xyz, rgb, label]
# # write_ply(outputplypath[i], [xyz, label], ['x', 'y', 'z', 'class'])