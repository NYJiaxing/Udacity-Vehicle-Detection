# Udacity-Vehicle-Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---
### Extract features in respect of color and HOG features

#### 1. Explain how you extracted HOG features from the training images.

The first step is to create a function called: 'get_hog_features' to get the HOG features from a picture with differetn parameters ('orientations', 'pixels_per_cell', and 'cells_per_block') pre-defined. The follow return pictures with parameters set as orientation = 9, pix_per_cell = 8 and cell_per_block = 2:

![alt text](/output_images/hog_features.png)

#### 2. Explain how you extract color features from the training images.

The second step is to create a function called 'extract_color_features'. It combine the features of image's color and hisgoram features of image as well. The 'extract_color_features' function with pre-defined parameters 'color space' i.e. 'RGB', 'LUV', 'YCrCb'(used in follow pictures) etc., and spatial_size, hist_bins and hist_range. The output of this function is a one dimension array combine the color and hist feature together to send to the classifier.

![alt text](/output_images/car1_color.png)
![alt text](/output_images/car2_color.png)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG and color features (and color features if you used them).

In the final choice of the features, I choose three features.
1. HOG Features. I am using all the 3 channels for extracting the HOG features.
2. Binned Color Features. spatial_size = (32, 32)
3. Color Histogram Features. color_space = 'YCrCb'
4. Orient = 9  
5. Pix_per_cell = 8  
6. Cell_per_block = 2  
7. Hog_channel = 'ALL'  
9. Spatial_feat = True  
10. Hist_feat = True  
11. Hog_feat = True

I create a function called 'extract_features' to extract hog and color features in one function. Then extract car_features and non_car_features with this function and stack them up as X_train sample. Then create y_train samples by set all cat_features as '1' and non_car_features as '0'. The third step is using a standard scaler to transfrom the X_train to scaled_X. Finally, put scaled X_train and y_train into the Linear SVC classifier and 20% of the data as test sample.

The total accuracy is 99.07%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To achive a sliding window search, I defined three functions to help, 'slidw_window' to define an interesting area to do the window search and return a window list, 'draw_boxes', a function to draw bounding boxes and 'search_windows', a function, with help of 'extract_features', to extract features in the windows and send it to the classifier to check if it was a car or not. Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text](/output_images/find_cars.png)

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a test video output 'test_video_output.mp4'


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text](/output_images/heatmap.png)
![alt text](/output_images/heat_map.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The majority problems I face is in some of the frames, expeacilly the frames contains half of the cars, the classifier cannot identify it as a car. No bounding box draw on it, this is dangerous if we were in a self-driving car because if we wanted to change lane in this time if the car cannot be identified, a crush will be caused. To fix this issue, I think we can take more data of the vehicle in different shape, size and directions to train the classifier to 'learn' this is the the vehicle and draw a bounding box on it. Also, I can try different methods, like YOLO and fast R-CNN to identify the vihicle faster and more accurate.

