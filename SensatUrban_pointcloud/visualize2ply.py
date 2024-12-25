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
inputplypath = ['output/output_1.ply',
                'output/output_2.ply',
                'output/output_3.ply']
                # 'Dataset/SensatUrban/original_block_ply/output_7_trans_3.ply',
                # 'Dataset/SensatUrban/original_block_ply/output_9_trans_3.ply',
                # 'Dataset/SensatUrban/original_block_ply/output_10_trans_3.ply',
                # 'Dataset/SensatUrban/original_block_ply/output_11_trans_3.ply',
                # 'Dataset/SensatUrban/original_block_ply/output_12_trans_3.ply',
                # 'Dataset/SensatUrban/original_block_ply/output_13_trans_3.ply']
                # 'Dataset/SensatUrban/original_block_ply/birmingham_block_2.ply',
                # 'Dataset/SensatUrban/original_block_ply/birmingham_block_8.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_15.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_16.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_22.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_27.ply']
# 2\读取的标签地址 /home/mspl/SensatUrban/test/Log_2022-04-12_06-01-31
inputlabelpath = ['test/Log_2022-05-04_15-08-37/test_preds/output_1_nor.label',
                  'test/Log_2022-05-04_15-08-37/test_preds/output_2_nor.label',
                  'test/Log_2022-05-04_15-08-37/test_preds/output_3_nor.label']
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
outputplypath = ['test/Log_2022-05-04_15-08-37/test_preds/output_1_nor1.ply',
                 'test/Log_2022-05-04_15-08-37/test_preds/output_2_nor1.ply',
                 'test/Log_2022-05-04_15-08-37/test_preds/output_3_nor1.ply']
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
for i in tqdm(range(len(inputplypath))):
    # print(i)
    xyz, rgb = DP.read_ply_data(inputplypath[i], with_rgb=True,
                                with_label=False)  # xyz.astype(np.float32), rgb.astype(np.uint8)
    print(len(xyz))
    label = np.fromfile(inputlabelpath[i], dtype=np.uint8)
    print(len(label))
    for j in tqdm(range(len(label))):
        
        rgb[j] = label_colors[label[j]]
    # print(rgb)
    write_ply(outputplypath[i], [xyz, rgb, label], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])  # xuek [xyz, rgb, label]
    # write_ply(outputplypath[i], [xyz, label], ['x', 'y', 'z', 'class'])