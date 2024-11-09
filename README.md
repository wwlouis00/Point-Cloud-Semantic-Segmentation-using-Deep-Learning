# Point-Cloud-Semantic-Segmentation-using-Deep-Learning

## Motivation and Scenario
- In image processing, semantic segmentation, in which pixels are linked with semantic labels, is a key research challenge.
- Point clouds, or fundamental 3D data, have become more accessible because of recently developed stereo vision algorithms and the deployment of many types of 3D sensors. 
- Since the provided data is unlabeled, our task is to use the point cloud as the input, assign each point with a semantic label and illustrate the segmentation result by using the 3rd party software.

## Kitti-360
- Across 320k pictures and 100k laser scans were gathered from numerous Karlsruhe, Germany, suburbs over a driving distance of 73.7km. 
- In order to create dense semantic & instance annotations for both 3D point clouds and 2D images, the authors annotate both static and dynamic 3D scene objects with rough bounding primitives.

## Toronto-3D
- Toronto-3D is a large-scale urban outdoor point cloud dataset acquired by an MLS system in Toronto, Canada for semantic segmentation. 
- This dataset covers approximately 1 km of road and consists of about 78.3 million points. Here is an overview of the dataset and the tiles. The approximate location of the dataset is at (43.726, -79.417).pedestrian crossings.

## Summary
- Overall Transfering process
- RandLA-Net
- SenSatUrban Dataset: Not the same format as ThinkTron Pointcloud data. 
- Kitti-360: Close to the format of ThinkTron data but does not include some required classes. 
- Toronto-3D: Close to the format of ThinkTron data, including road mark class, but cannot be applied to ThinkTron data.

