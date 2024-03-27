# Machine Learning Overview

*Author: Valentina Staneva*

This is an overview of machine learning with examples in hydrology and cryosphere science. It is based on a presentation given at the [GeoSmart Hackweek 2024](https://geosmart.hackweek.io/) at the University of Washington. 

The objective of this overview is that readers learn how to:

* Identify and formulate major types of ML problems from scientific questions
* List major differences between types of methods, and steps to proceed to evaluate their performance
* Recognize contexts where deep learning can be useful
* Identify popular computer vision/time series tasks and popular deep learning frameworks for them
* Organize labeled datasets in format for those frameworks
* Outline elements of ML pipelines


### Machine Learning Algorithms

In the abundance of Machine Learning Algorithms, new learners often struggle to identify which one is best for their tasks. Let's look at this chart from [Andreas Müller](https://amueller.github.io/), the author of the Python [`scikit-learn`](https://scikit-learn.org/stable/). It attempts to provide a decision path for suggesting methods from the package, depending on the type of problems and the available data. Despite relying on such charts can be dangerous (it may be hard to determine the right algorithm in advance as one often needs to test and compare performance between different methods), this chart is very useful to identify a few major types of problems and approaches. The following problems are discussed:

**Supervised Learning:**

* *classification*: predicting a category 
* *regression*: predicting a quantity

**Unsupervised Learning:**

* *clustering*: identifying categories without labels
* *dimensionality reduction*: identifying "templates" representing the data

Often, the first step is to decide which one of these types of problems is suitable for the overarching research problem. In practice, the boundaries are fluid, and the same research problem can be addressed by either of these ML problems or a combination of them. For example,

* one can first run a dimensionality reduction algorithm to reduce the number of variables and then run a regression on those newly formed variables.
* One could address the problem of detecting snow in satellite imagery as a binary classification problem: snow or no snow; or one could use a continuous snow index and use regression to predict it.


2023 

Many methods from the chart are still applicable to address problems for which data is in tabular format. Since the deep learning revolution in 2014, many scenarios where there is a correlation between the input variables (such as in images, time series, text) have been successfully addressed with deep neural networks (CNNs for images, videos, lstm and transformers for time series and text).


### Data in Earth Sciences

Many traditional machine learning algorithms assume the input data is in tabular format. However, data in earth science can take all sizes, shapes, and scales. 

### Feature Engineering
Traditional approaches often require designing a set of features from our complex datasets, which can be fed to a model such as random forest or logistic regression. Deep neural networks allow us to feed data in its raw format (for example 2D images) and the network acts as a feature extractor. It has been shown that the first layers of a convolutional neural network act as filters which are traditionally used in image processing (edge detectors, Gabor filters) [reference] [figure]. Deeper layers extract more global information. 

**Example 1: Audio Signal**

* sound descriptors (small set of variables)
* power spectrum (1D)
* spectrogram PCA
* spectrogram (2D)


**Example 2: Image**

* color histograms
* histograms of oriented gradients
* PCA
* raw image




### Computer Vision Tasks:

There are several common tasks in computer vision that can be addressed well with deep learning. Each task requires different mathematical problem formulation and benefits from a different architecture, a different training dataset format, and a different evaluation strategy. It is often helpful to realize if one's scientific research question can be fitted within one of these common tasks, as then the researcher can benefit from the existing tools in the community to address this kind of tasks. The terminology can often be confusing as similar language may mean something else outside of the computer vision literature, and vary across domains. We will focus on the following tasks described in this diagram. 

* **Image Classification:** predicting the category of an entire image

	* image -> label	
	* label the whole image
* **Semantic Segmentation:** predicting the category of every pixel
	* image -> mask
	* label for each pixel	
* **Object Detection:** localization and identification of objects in an image
	* image -> set of bounding boxes corresponding to objects in the image and corresponding labels
* **Instance Segmentation:** detection and delineation of objects in an image
	* image -> set of objects with corresponding pixels, and labels indicating which class of objects the object belongs to, can be also described by two masks: one with the labels of the object classes, and the other one indicating the individual object id.


One can see that the appropriate task can vary based on:

* the shape of the objects: can they be inscribed by a bounding box or not 
* the precision at which we want to do detection: do we care about extracting boundaries, or the overall location of the object is good enough and we do not need precision beyond the size of the image or a bounding box
* the number of objects within an image
* the need to distinguish between individual objects of the same class


Go through different examples for different categories, or the same example formulated differently for the different categories?


### Image Classification:

### Image Segmentation:

One of the common approaches to Image Segmentation is using the U-Net architecture (or some variation of it). U-Net was introduced in the biomedical imaging community, for which many problems require identifying regions in images with high precision but not so much training data. Since then U-Net architectures have become prevalent across different applications: from detecting rivers in satellite images, to etc......

A simple U-net can be constructed in `keras` and can be trained on your laptop (when the training set ).

* encoder 
* decoder

An important point to realize is that the output of a semantic segmentation task should have the same dimensions as the output image. Usually, a convolutional neural network for classification can be thought of as a transformation of a 2D image into lower dimensional features (encoder) which eventually are transformed into a predicted class. In the context of semantic segmentation, one needs to upscale the features to the original resolution. That can be obtained by an "upscaling" type of operation (decoder). 

One of the key elements of setting up a semantic segmentation workflow is to prepare a training dataset suitable for such a network. The key is to understand how the loss is calculated. 

* calculate categorical cross-entropy loss at each pixel
* sum over each pixel, sum over all images in batches












### Object Detection:

### Instance Segmentation: 



















	







 




