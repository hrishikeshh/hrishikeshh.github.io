---
layout: page
title: DeepGlobe Land Cover Classification 
description: Classifying land coverage using remote sensing satelite data.
img: assets/img/land.jpg
importance: 1
category: AI
---


## 1. Background

This project presumes some prior high level understanding of machine learning, deep neural networks, and working in Python, Keras, and TensorFlow 2.0.

**Satellite Imagery**: Satellite and remote sensing have seen great advances in recent years with one of the key catalyst being the price compression in launch costs from private space launch companies. This has allowed a wider cast of entities and organizations to deploy more satellites. The quality of imagery available to the general public, such as those from Sentinel-2, is also increasing; although still a ways from privately paid satellite services. Sentinel-2 has a typical pixel resolution of 10m band which means that each pixel represents 10m x 10m area.

**CNNs**: Convolutional Neural Networks are the state of the art neural network architecture for image recognition and computer vision tasks. There are many different variations of CNNs but they typically consist of convolution, pooling, and dense connected layers. CNNs perform much better than regular feed-forward neural networks in images due to its ability to reduce the resolution (dimensions) of an image onto feature maps while still encoding the key features of that image. This greatly reduces the complexity and computational need of the network. A typical CNN architecture is shown below.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deepglobe/cnn.webp" title="Typical structure of CNN" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A typical Convolutional Neural Network. Source: HML
</div>


**FCNs**: Typical CNNs perform great for classifying entire images but have difficulty classifying objects and segments within images. This is due to the fact that the convolution and pooling layers essentially encode the key features of the image down into ever smaller feature maps then feed that into fully connected layers, before a final Sigmoid or Softmax activation layer for final classification. The encoded features or final classifications cannot be easily mapped back to the original image.

This is where *Long et al. (2015)’s Fully Convolutional Network (FCN)* come in. FCNs are just like CNNs except they have a series of convolutional and upsampling layers to mirror the convolutional and pooling layers in place of the dense connected layers. This allows the network to encode (downsample) and then decode (upsample) back into the original image’s resolution and spatial dimensions which allow for pixel-wise classification. If we edit the typical CNN layer from earlier, we might reach a configuration of something that looks like below.



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deepglobe/cnn-int.webp" title="CNN" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    What a Fully Convolutional Network might look like. Note that this is not a true FCN but helps in gaining some intuition into the how and why. Source: HML
</div>


**Common Types of Computer Vision Problems:** For this project, I will be working with a type of computer vision problem called **semantic segmentation**. There are many other types of problem sets like object detection, instance segmentation, and others which warrant their own discussions at a later date.

Semantic Segmentation is determining which pixels in an image belongs to which class as shown below. For these problems, pixel-wise classification and localization of objects and edges within an image are important and therefore are good uses of FCNs.





<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deepglobe/segmentation.webp" title="segmentation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Semantic Segmentation Example. Source: Jeremy Jordan
</div>


## 2. Dataset

> The [DeepGlobe Land Cover Classification Challenge](https://competitions.codalab.org/competitions/18468) and hence, the dataset are governed by [DeepGlobe Rules](http://deepglobe.org/docs/DeepGlobe_Rules_3_2.pdf), The [DigitalGlobe’s Internal Use License Agreement](http://deepglobe.org/docs/CVPR_InternalUseLicenseAgreement_07-11-18.pdf), and [Annotation License Agreement](http://deepglobe.org/docs/Annotation%20License%20Agreement.pdf).

> Data Source : [DeepGlobe Land Cover Classification](https://competitions.codalab.org/competitions/18468)


**Training Data**: The dataset comprised of **803 satellite imagery** in RGB of size **2448x2448** with one set comprised of satellite images and the other set comprised of labeled masks. Each satellite image is at the 50cm pixel resolution band collected by DigitalGlobe’s satellite. 50cm pixel resolution is quite high by typical publicly accessible satellite imagery resolution standards which are usually around the 5m to 10m pixel resolution band.

**Test Data:** The dataset also contained **171 validation** and **172 test images**, both without labeled masks. I ended up combining these 2 groups into a testing dataset and randomly sampled a subset of the training data for validation.

Here are some sample training images:


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deepglobe/sample-img.webp" title="Sample data" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Sample of 8 training satellite images and their corresponding Ground Truth Mask labels.
</div>


































<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>