# Bundle of helper functions for Deep Learning pipelines

A set of simple yet useful Python functions to expedite the execution of common routines from your Deep Learning training/testing pipelines. I found myself using some version of these functions repeatedly over time, reason why I decided to make them generic and to share them publicly.

Last update: July 13th, 2021

Requirements: PyTorch 1.3.1, Opencv-python 4.2.0.   
I believe that different versions (in particular of OpenCV) might create problems requiring small modifications in some of the functions used.

## Summary of functions 

### print_gpu_stats
Description: reports your GPU ID and memory usage.   
Usage: use it in the beginning of each epoch of a training routine for improved memory monitoring. Employs PyTorch-based functionalities.  
Sample:   
<img align="center" src="https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/print_gpu_stats.png?raw=true">  

### iou_segm
Description: calculates the pixel-level intersection-over-union (IoU) between a prediction image and a ground truth mask.  
Usage: instance and semantic segmentation frameworks perform pixel-level predictions for two or more classes. Use this function to compare such predictions against a ground-truth mask. Note that each class' prediction and their respective masks have to be parsed individually.  
Sample [image](https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/iou_segm.png?raw=true)  

### mse
Description: calculates the mean squared error (mse) between two images.  
Usage: Useful (in particular in combination with SSIM) when evaluating the results of reconstruction/denoising/enhancing methods against a ground truth reference.  

### show_img
Description: Catch-all printing function for images of different formats and containers (e.g., ndarray, Tensor) using OpenCV.  
Usage: given the images' different dataformats, ranges and containers we often work with when training vision systems, I created a function that could be generically used to try and plot the content of most images.  
Sample:  
<img align="center" src="https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/show_img.png?raw=true">  

### filter_detectron2_detections
Description: filters detectron2's bounding-box-based detections.  
Usage: if you work with object detection frameworks offered by detectron2, this function can help you filter their detection results based on desired classes and detection thresholds, as well as valid y-axis range (so that you can ignore detection in certain parts of the image).  

### concatenate_bbs
Description: concatenates outputs composed by multiple bounding boxes.  
Usage: use when desiring to combine multiple overlapping bounding boxes (often seen in the output of object detector even after NMS). Note that the concatenation result is given by the score and class of the highest-scoring detection; thus the use of this function is recommend when only one class is presented, or when it is reasonable to merge detection of distinct classes.   
Sample [image](https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/concatenate_bbs.png?raw=True)   

### plot_all_bbs
Description: plots all the detection bounding boxes found in a given image.  
Usage: useful when debugging the multiple detections of an object detector. Detections of different classes can be distinguished with different colours.    
Sample [image](https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/plot_all_bb.png?raw=True)   

### read_all_images
Description: reads all image files of a given format from a folder.  
Usage: as an important step of dataloading operations, this function can easily read all images from a directory given a specific format.  
Sample:  
<img align="center" src="https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/read_all_images.png?raw=true">  

### check_create_dir
Description: check if a directory exists and creates it if it doesn't.  
Usage: use it to create output directories for different experiments, among others. 

### create_metadata
Description: creates a text file with generic metadata from DL models.  
Usage: given the large number of hyperparameters involved in DL-based training/testing, I created this function to organize the most commonly-used pieces of metadata under the same text file.    
Sample [image](https://github.com/tunai/DL-helper-functions/blob/master/samples/create_metadata.png?raw=True)   

### iou_bbs
Description: Returns the IoU between two bounding boxes (third party by [Adrian Rosebrock, 2016](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)).  
Usage: IoU is a key metric when post-processing and evaluating object detection and instance segmentation systems. This function calculates the IoU between a pair of bounding boxes in this context.  
Sample:  
<img align="center" src="https://raw.githubusercontent.com/tunai/DL-helper-functions/master/samples/iou_bbs.png?raw=true">  


