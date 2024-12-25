import numpy as np
from tool import DataProcessing as DP
from helper_ply import write_ply, read_ply  # xuek
from tqdm import tqdm


# 颜色标签
label_colors = [[85, 107, 47], [0, 255, 0],  # tree -> Green
                [255, 165, 0],  # building -> orange
                [41, 49, 101],  # Walls ->  darkblue
                [0, 0, 0],  # Bridge -> black
                [0, 0, 255],  # parking -> blue
                [255, 0, 255],  # rail -> Magenta
                [200, 200, 200],  # traffic Roads ->  grey
                [89, 47, 95],  # Street Furniture  ->  DimGray
                [255, 0, 0],  # cars -> red
                [255, 255, 0],  # Footpath  ->  deeppink
                [0, 255, 255],  # bikes -> cyan
                [0, 191, 255]  # water ->  skyblue
                ]
label_color = np.asarray(label_colors, dtype=np.uint8)
# 　label_color = label_color / 255.0

# 地址
# 1\读取的ply地址

inputplypath = ['output3/output_2.ply',
                'output3/output_3.ply',
                'output3/output_6.ply',
                'output3/output_7.ply',
                'output3/output_9.ply',
                'output3/output_10.ply',
                'output3/output_11.ply',
                'output3/output_12.ply',
                'output3/output_13.ply']
                # 'Dataset/SensatUrban/original_block_ply/birmingham_block_2.ply',
                # 'Dataset/SensatUrban/original_block_ply/birmingham_block_8.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_15.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_16.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_22.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_27.ply']
# 2\读取的标签地址 /home/mspl/SensatUrban/test/Log_2022-04-12_06-01-31
inputlabelpath = ['test/Log_2022-05-02_15-07-34/test_preds/output_1.label',
                  'test/Log_2022-05-02_15-07-34/test_preds/output_2.label',
                  'test/Log_2022-05-02_15-07-34/test_preds/output_3.label']
                #   'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_2.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_8.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_15.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_16.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_22.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_27.label']
# 2\写出的ply地址
outputplypath = ['Dataset/SensatUrban/2/output_2_trans_3.ply',
                 'Dataset/SensatUrban/2/output_3_trans_3.ply',
                 'Dataset/SensatUrban/2/output_6_trans_3.ply',
                 'Dataset/SensatUrban/2/output_7_trans_3.ply',
                 'Dataset/SensatUrban/2/output_9_trans_3.ply',
                 'Dataset/SensatUrban/2/output_10_trans_3.ply',
                 'Dataset/SensatUrban/2/output_11_trans_3.ply',
                 'Dataset/SensatUrban/2/output_12_trans_3.ply',
                 'Dataset/SensatUrban/2/output_13_trans_3.ply']

for i in tqdm(range(len(inputplypath))):
    xyz, rgb = DP.read_ply_data(inputplypath[i], with_rgb=True, with_label=False)
    label = np.random.randint(13, size=len(xyz), dtype=np.uint8)
    tx = -271745
    ty = -2767261
    tz = 0
    newxyz = np.ones(xyz.shape)
    for j in tqdm(range(len(xyz))):
        newxyz[j, :] = xyz[j,:] + np.array([tx, ty, tz]) #np.dot(xyz[j,:],T_matrix) #
    write_ply(outputplypath[i], [newxyz, rgb, label], ['x', 'y', 'z', 'red', 'green', 'blue', 'value']) 