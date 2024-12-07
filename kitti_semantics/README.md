
# ThinkTron Point cloud segmentation

This is the repository of utilizing pretrained model of the **SensatUrban** dataset for running inference on ThinkTron Dataset. For technical details, please refer to the detailed report.

### (1) Dataset

#### 1.1 Overview

SenSatUrban dataset is an urban-scale photogrammetric point cloud dataset with nearly three billion richly annotated points, 
which is five times the number of labeled points than the existing largest point cloud dataset. 
Our dataset consists of large areas from two UK cities, covering about 6 km^2 of the city landscape. 
In the dataset, each 3D point is labeled as one of 13 semantic classes, such as *ground*, *vegetation*, 
*car*, *etc.*. 

<p align="center"> <img src="imgs/Fig1.png" width="100%"> </p>

ThinkTron dataset is an urban-scale point cloud collected by LiDar sensor with nearly 275 milion points. The dataset consist of views from Taoyuan-Taiwan streetview. The dataset is not labeled.

<p align="center"> <img src="imgs/thinktron_data.png" width="100%"> </p>
<p align="center"> <img src="imgs/thinktron_data2.png" width="100%"> </p>


#### 1.2 Data Collection

The 3D point clouds are generated from high-quality aerial images captured by a professional-grade UAV mapping system. In order to fully and evenly cover the survey area, all flight paths are pre-planned in a grid fashion and automated by the flight control system (e-Motion).

<p align="center"> <img src="imgs/Fig2.png" width="70%"> </p>

#### 1.3 Semantic Annotations

<p align="center"> <img src="imgs/Fig3.png" width="100%"> </p>

- Ground: including impervious surfaces, grass, terrain
- Vegetation: including trees, shrubs, hedges, bushes
- Building: including commercial / residential buildings
- Wall: including fence, highway barriers, walls
- Bridge: road bridges
- Parking: parking lots
- Rail: railroad tracks
- Traffic Road: including main streets, highways
- Street Furniture: including benches, poles, lights
- Car: including cars, trucks, HGVs
- Footpath: including walkway, alley
- Bike: bikes / bicyclists
- Water: rivers / water canals


#### 1.4 Statistics
<p align="center"> <img src="imgs/Fig5.png" width="100%"> </p>

### (4) Training and Evaluation
Here we provide the training and evaluation script of [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) for your reference.
- Download the dataset 

Download the files named "data_release.zip" [here](https://forms.gle/m4HJiqZxnq8rmjc8A). Uncompress the folder and move it to `/Dataset/SensatUrban`.

- Setup the environment
```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
```

- Preparing the dataset
```
python input_preparation.py --dataset_path $YOURPATH
cd $YOURPATH; 
cd ../; mkdir original_block_ply; mv data_release/train/* original_block_ply; mv data_release/test/* original_block_ply;
mv data_release/grid* ./
```
The data should organized in the following format:
```
/Dataset/SensatUrban/
          └── original_block_ply/
                  ├── birmingham_block_0.ply
                  ├── birmingham_block_1.ply 
		  ...
	    	  └── cambridge_block_34.ply 
          └── grid_0.200/
	     	  ├── birmingham_block_0_KDTree.pkl
                  ├── birmingham_block_0.ply
		  ├── birmingham_block_0_proj.pkl 
		  ...
	    	  └── cambridge_block_34.ply 
```

- Start training: (Please first modified the root_path)
```
python main_SensatUrban.py --mode train --gpu 0 
```
- Evaluation:
```
python main_SensatUrban.py --mode test --gpu 0 
```



