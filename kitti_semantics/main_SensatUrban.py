from os.path import join, exists, dirname, abspath
from RandLANet import Network
from tester_SensatUrban import ModelTester
from helper_ply import read_ply
from tool import ConfigSensatUrban as cfg
from tool import DataProcessing as DP
from tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os, shutil


class SensatUrban:
    def __init__(self):
        self.name = 'kitti'
        root_path = './Dataset'  # path to the dataset
        self.path = join(root_path, self.name)
        print(self.path,1111111111111111111111111)
        self.label_to_names = { 
                                0: 'road',
                                1: 'sidewalk',
                                2: 'building',
                                3: 'wall',
                                4: 'fence',
                                5: 'pole',
                                6: 'traffic light',
                                7: 'traffic sign',
                                8: 'vegetation',
                                9: 'terrain',
                                10: 'person',
                                11: 'car',
                                12: 'truck',
                                13: 'motorcycle',
                                14: 'bicycle',
                                15: 'skip'                                
                                }
            # 19: 'void', 
            #                    0: 'road', 
            #                    1: 'sidewalk', 
            #                    2: 'building', 
            #                    3: 'wall', 
            #                    4: 'fence', 
            #                    5: 'pole', 
            #                    6: 'traffic light', 
            #                    7: 'traffic sign', 
            #                    8: 'vegetation',  
            #                    9: 'terrain',  
            #                    10: 'sky',  
            #                    11: 'person',  
            #                    12: 'rider',  
            #                    13: 'car',  
            #                    14: 'truck',  
            #                    15: 'bus',  
            #                    16: 'train', 
            #                    17: 'motorcycle',  
            #                    18: 'bicycle'}
        # {0: 'void', 1: 'flat', 2: 'construction', 3: 'object',
        #                        4: 'nature', 5: 'sky', 6: 'human', 7: 'vehicle'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = [15]#np.array([15])

        self.all_files = np.sort(glob.glob(join(self.path, 'original_block_ply_2', '*.ply')))
        # self.val_file_name = []
        self.val_file_name = ['0000000372_0000000610', '0000000599_0000000846', '0000000834_0000001286', 
                              '0000001270_0000001549', '0000001537_0000001755', '0000001740_0000001991', 
                              '0000002695_0000002925', '0000003221_0000003475', '0000003463_0000003724', 
                              '0000003711_0000003928', '0000004391_0000004625', '0000004613_0000004846', 
                              '0000004835_0000005136', '0000005125_0000005328', '0000005317_0000005517', 
                              '0000005506_0000005858', '0000005847_0000006086', '0000006069_0000006398', 
                              '0000015189_0000015407', '0000000002_0000000282', '0000002897_0000003118', 
                              '0000003107_0000003367', '0000003356_0000003586', '0000003570_0000003975', 
                              '0000004370_0000004726', '0000004708_0000004929', '0000004771_0000005011', 
                              '0000004998_0000005335', '0000005324_0000005591', '0000005579_0000005788', 
                              '0000005777_0000006097', '0000006086_0000006307', '0000000002_0000000403', 
                              '0000000387_0000000772', '0000000754_0000001010', '0000001000_0000001219', 
                              '0000002280_0000002615', '0000002511_0000002810', '0000009213_0000009393', 
                              '0000009383_0000009570', '0000000002_0000000125', '0000000119_0000000213', 
                              '0000000208_0000000298', '0000000293_0000000383', '0000000378_0000000466', 
                              '0000000461_0000000547', '0000000778_0000001026', '0000001005_0000001244', 
                              '0000001234_0000001393', '0000004475_0000004916', '0000005156_0000005440', 
                              '0000005422_0000005732', '0000006272_0000006526', '0000012398_0000012693', 
                              '0000013370_0000013582', '0000013575_0000013709', '0000013701_0000013838', 
                              '0000001872_0000002033', '0000002024_0000002177', '0000002168_0000002765', 
                              '0000002756_0000002920', 'instance0_2'] #fold 4
        self.test_file_name =  ['instance1_2', 'instance2_2', 'instance3_2'] #fold 4 ['validationTT2']
                            #     ['output_2_trans2',
                            #     'output_3_trans2',
                            #     'output_6_trans2',
                            #     'output_7_trans2',
                            #     'output_9_trans2',
                            #    'output_10_trans2',
                            #    'output_11_trans2',
                            #    'output_12_trans2',
                            #    'output_13_trans2']#,'output_2_trans','output_3_trans']#, 'output_2.ply', 'output_3.ply']
                                # ['0000000372_0000000610', '0000000599_0000000846', '0000000834_0000001286']
                               
                            #   ['0000000002_0000000245', '0000000235_0000000608', '0000000581_0000000823', 
                            #    '0000000812_0000001058', '0000001046_0000001295', '0000001277_0000001491', 
                            #    '0000002404_0000002590', '0000002580_0000002789', '0000002769_0000003002', 
                            #    '0000004623_0000004876', '0000004854_0000005104', '0000005093_0000005329', 
                            #    '0000005316_0000005605', '0000005588_0000005932', '0000005911_0000006258', 
                            #    '0000006247_0000006553', '0000006517_0000006804', '0000006792_0000006997', 
                            #    '0000006988_0000007177', '0000007161_0000007890', '0000007875_0000008100', 
                            #    '0000008090_0000008242', '0000008236_0000008426', '0000008417_0000008542', 
                            #    '0000008536_0000008643', '0000008637_0000008745', '0000000002_0000000341', 
                            #    '0000000330_0000000543', '0000000530_0000000727', '0000000717_0000000985', 
                            #    '0000000975_0000001200', '0000001191_0000001409', '0000001399_0000001587', 
                            #    '0000001577_0000001910', '0000001878_0000002099', '0000002090_0000002279', 
                            #    '0000002269_0000002496', '0000002487_0000002835', '0000002827_0000003047', 
                            #    '0000003033_0000003229', '0000003215_0000003513', '0000003503_0000003724']

                            #    'cambridge_block_16', 'cambridge_block_27',
                            #    'birmingham_block_2', 'birmingham_block_8',
                            #    'cambridge_block_15', 'cambridge_block_22']
                            #    'output_9_trans_3', 
                            #    'output_10_trans_3', 
                            #    'output_11_trans_3', 
                            #    'output_12_trans_3', 
                            #    'output_13_trans_3']#,'output_2_trans','output_3_trans']#, 'output_2.ply', 'output_3.ply']
                            #
        self.use_val = True  # whether use validation set or not

        # initialize
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'grid_{:.3f}'.format(sub_grid_size))

        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if cloud_name in self.test_file_name:
                cloud_split = 'test'
            elif cloud_name in self.val_file_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['semantic']

            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]
            # print(self.input_trees,self.input_colors,self.input_labels,self.input_names)
            # print(self.input_names)
            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # val projection and labels
            if cloud_name in self.val_file_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

            # test projection and labels
            if cloud_name in self.test_file_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        else:
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    shutil.rmtree('__pycache__') if exists('__pycache__') else None
    if Mode == 'train':
        shutil.rmtree('results') if exists('results') else None
        shutil.rmtree('train_log') if exists('train_log') else None
        for f in os.listdir(dirname(abspath(__file__))):
            if f.startswith('log_'):
                os.remove(f)

    dataset = SensatUrban()
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snapshot = -1
        logs = np.sort([os.path.join('./results_3', f) for f in os.listdir('./results_3') if f.startswith('Log')])
        chosen_folder = './results/Log_2022-07-21_kitti'#logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
        shutil.rmtree('train_log') if exists('train_log') else None
    
    elif Mode == 'finetune':
        # cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snapshot = -1
        logs = np.sort([os.path.join('./results', f) for f in os.listdir('./results') if f.startswith('Log')])
        chosen_folder = logs[-1]
        snap_path = join(chosen_folder, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        print(chosen_snap)
        # tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        # tester.test(model, dataset)
        # shutil.rmtree('train_log') if exists('train_log') else None
        model.finetune(dataset, restore_snap=chosen_snap)

    else:

        with tf.Session() as sess:
            model = Network(dataset, cfg)
            # chosen_snapshot = -1
            # logs = np.sort([os.path.join('./results', f) for f in os.listdir('./results') if f.startswith('Log')])
            # chosen_folder = logs[-1]
            # snap_path = join(chosen_folder, 'snapshots')
            # snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            # chosen_step = np.sort(snap_steps)[-1]
            # chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
            # tester = ModelTester(model, dataset, restore_snap=chosen_snap)
            # tester.test(model, dataset)
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                data_list = sess.run(dataset.flat_inputs)
                xyz = data_list[0]
                sub_xyz = data_list[1]
                label = data_list[21]
                Plot.draw_pc_sem_ins(xyz[0, :, :], label[0, :])
