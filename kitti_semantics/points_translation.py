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
inputplypath = ['200x200ply/output_2.ply',
                '200x200ply/output_3.ply',
                '200x200ply/output_6.ply',
                '200x200ply/output_7.ply',
                '200x200ply/output_9.ply',
                '200x200ply/output_10.ply',
                '200x200ply/output_11.ply',
                '200x200ply/output_12.ply',
                '200x200ply/output_13.ply']
                # 'Dataset/SensatUrban/original_block_ply/birmingham_block_2.ply',
                # 'Dataset/SensatUrban/original_block_ply/birmingham_block_8.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_15.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_16.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_22.ply',
                # 'Dataset/SensatUrban/original_block_ply/cambridge_block_27.ply']
# 2\读取的标签地址 /home/mspl/SensatUrban/test/Log_2022-04-12_06-01-31
# inputlabelpath = ['test/Log_2022-05-02_15-07-34/test_preds/output_1.label',
#                   'test/Log_2022-05-02_15-07-34/test_preds/output_2.label',
#                   'test/Log_2022-05-02_15-07-34/test_preds/output_3.label']
                #   'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_2.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_8.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_15.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_16.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_22.label',
                #   'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_27.label']
# 2\写出的ply地址
outputplypath = ['Dataset/kitti/2/output_2_trans2.ply',
                 'Dataset/kitti/2/output_3_trans2.ply',
                 'Dataset/kitti/2/output_6_trans2.ply',
                 'Dataset/kitti/2/output_7_trans2.ply',
                 'Dataset/kitti/2/output_9_trans2.ply',
                 'Dataset/kitti/2/output_10_trans2.ply',
                 'Dataset/kitti/2/output_11_trans2.ply',
                 'Dataset/kitti/2/output_12_trans2.ply',
                 'Dataset/kitti/2/output_13_trans2.ply']
                # test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_2.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/birmingham_block_8.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_15.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_16.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_22.ply',
                #  'test/Log_2022-04-12_06-01-31/test_preds/cambridge_block_27.ply']

# 数据处理
for i in tqdm(range(len(inputplypath))):
    # print(i)
    # '/home/mspl/SensatUrban/Dataset/SensatUrban/1/cambridge_block_17.ply'
    xyz, rgb = DP.read_ply_data(inputplypath[i], with_rgb=True, with_label=False)  # xyz.astype(np.float32), rgb.astype(np.uint8)
    # label = np.random.randint(13, size=len(xyz), dtype=np.uint8)
    # print(xyz)
    # a = np.ones((len(xyz),1), dtype=np.uint)
    # print(a)
    # xyz = np.append(xyz, a, axis=1)
    # print(xyz)
    # tx = -271201
    # ty = -2766607
    # tz = -46
    factor=100
    xyz = xyz[::factor]
    rgb = rgb[::factor]
    tx = -271023
    ty = -2767801
    tz = 0
    # T_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[tx,ty,tz,1]], dtype=np.uint)
    # print(T_matrix)
    # newxyz = np.dot(xyz[0,:],T_matrix)
    newxyz = np.ones(xyz.shape)
    print(newxyz.shape)
    # print(newxyz)
    # newxyz = np.expand_dims(newxyz, axis=0)
    # print(newxyz)
    for j in tqdm(range(len(xyz))):
        newxyz[j, :] = xyz[j,:] + np.array([tx, ty, tz]) #np.dot(xyz[j,:],T_matrix) #
        # print(newxyz[j,:] - xyz[j,:])
        
    # print(xyz - newxyz)
    # label = np.fromfile(inputlabelpath[i], dtype=np.uint8)
    # print(len(label))
    # for j in tqdm(range(len(label))):
        
    #     rgb[j] = label_colors[label[j]]
    # print(rgb)
    write_ply(outputplypath[i], [newxyz, rgb], ['x', 'y', 'z', 'red', 'green', 'blue'])  # xuek [xyz, rgb, label]
    # write_ply(outputplypath[i], [xyz, label], ['x', 'y', 'z', 'class'])