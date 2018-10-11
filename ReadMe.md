# **Traffic Sign Recognition** 

## Writeup
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Visualization"
[image2]: ./examples/image2.png "Grayscaling"
[image3]: ./examples/image3.png "Random Noise"
[image4]: ./examples/tcn1.png "Traffic Sign 1"
[image5]: ./examples/tcn2.png "Traffic Sign 2"
[image6]: ./examples/tcn3.png "Traffic Sign 3"
[image7]: ./examples/tcn4.png "Traffic Sign 4"
[image8]: ./examples/tcn5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1] 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce the influence of the color variance 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to remove the data scale variance because that machine learning models perform better in the same scale

I decided to generate additional data because :

+ data imbalance
+ make the model more robust 

To add more data to the the data set, I used the following techniques because :

+ add random noise to make model generalize better
+ rotation to make model rotation insensitive
+ Translation to make model translation insensitive

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

+ data balance
+ random noise image
+ rotation image
+ translation image

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x80 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x120 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x120				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x180 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x200 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x200				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x200 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x200				|
| Fully connected		| outputs 100       									|
| Fully connected		| outputs 100       									|
| Fully connected		| outputs 43       									|
| Softmax				|.        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:

+ type of optimizer : AdamOptimizer
+ the batch size : 43*3, in each batch, I choose 3 random images  from each label
+ number of epochs : 20
+ learning rate : 0.001


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were: 

* training set accuracy of 0.996264
* validation set accuracy of 0.985714
* test set accuracy of 0.9740300869432028
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| turn-right     		| turn-right   									| 
| Speed limit (50km/h) | Speed limit (50km/h)
| Stop Sign					| Stop Sign											|
| Road work	      		| Road work					 				|
| No entry			| No entry      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Turn right ahead (probability of 1.0), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn right ahead   									| 
| 1.2295730300593277e-07   				| Go straight or right 										|
| 1.0701820762903935e-08					| Ahead only											|
|  2.613339822321592e-10	      			| No passing for vehicles over 3.5 metric tons					 				|
| 2.228611600818553e-10				    |  No passing      							|


For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9785341024398804         			| Speed limit (50km/h)   									| 
| 0.016302425414323807   				|   Speed limit (30km/h) 										|
| 0.004919261671602726					|   Speed limit (80km/h)											|
|  0.00023208011407405138	      			| Speed limit (70km/h)					 				|
| 9.29400175664341e-06				    |  label Speed limit (60km/h)      							|


For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999737739562988         			| Road work   									| 
| 2.6151828933507204e-05   				|   Bicycles crossing 										|
| 3.2052355436462676e-08					|   Beware of ice/snow											|
|  2.8662196172035692e-08	      			| Bumpy road					 				|
| 1.1503982655369782e-08				    |  Children crossing      							|


For the forth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop   									| 
| 1.9003904149655394e-15   				|   Speed limit (30km/h) 										|
|  8.95112760801941e-16					|   Speed limit (50km/h)											|
|  2.6975794437408644e-16	      			| Go straight or right					 				|
| 3.717828775824097e-17				    |  Yield      							|



For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry   									| 
| 7.976126426001429e-10   				|   No passing 										|
|   2.76681011435187e-10					|   Yield											|
|  1.3926997055602186e-10	      			| Keep left					 				|
| 5.73403789727589e-11				    |  End of all speed and passing limits      							|









