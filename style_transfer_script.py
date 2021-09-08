# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:47:47 2021

@author: diggee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from imageio import mimsave
from skimage.exposure import match_histograms

#%% restricting tensorflow GPU usage

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)   # prevents tf from allocating entire GPU RAM upfront
  except RuntimeError as e:
    print(e)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #to force tf to only use CPU and not GPU 

#%% display images

def display_images(images, titles):
    plt.figure(figsize=(20, 12), dpi = 300)
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        plt.title(title)
    
#%% VGG model

def vgg_model(layer_names):
    vgg = VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = Model(inputs = vgg.input, outputs = outputs)
    return model
    
#%% computing gram matrices

def gram_matrix(input_tensor):
  # calculate the gram matrix of the input tensor
  gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) 

  # get the height and width of the input tensor
  input_shape = tf.shape(input_tensor) 
  height = input_shape[1] 
  width = input_shape[2] 

  # get the number of locations (height times width), and cast it as a tf.float32
  num_locations = tf.cast(height * width, tf.float32)

  # scale the gram matrix by dividing by the number of locations
  scaled_gram = gram / num_locations    
  return scaled_gram

#%% style image features

def get_style_image_features(image):  
    image = tf.cast(image, dtype=tf.float32)
    preprocessed_style_image = preprocess_input(image) 
    outputs = vgg(preprocessed_style_image) 
    style_outputs = outputs[:num_style_layers] 
    gram_style_features = [gram_matrix(style_layer) for style_layer in style_outputs] 
    return gram_style_features

#%% content image features

def get_content_image_features(image):
    image = tf.cast(image, dtype=tf.float32)
    preprocessed_content_image = preprocess_input(image)
    outputs = vgg(preprocessed_content_image) 
    content_outputs = outputs[num_style_layers:]
    return content_outputs

#%% style, content and total loss computation

def get_style_loss(style_image, target_image):
    style_loss = tf.reduce_mean(tf.square(style_image - target_image))
    return style_loss

def get_content_loss(content_image, target_image):
    content_loss = 0.5*tf.reduce_sum(tf.square(content_image - target_image))
    return content_loss

def get_style_content_loss(style_targets, style_outputs, content_targets, content_outputs, style_weight, content_weight):    
    # sum of the style losses
    style_loss = tf.add_n([get_style_loss(style_output, style_target)
                             for style_output, style_target in zip(style_outputs, style_targets)])
  
    # Sum up the content losses
    content_loss = tf.add_n([get_content_loss(content_output, content_target)
                             for content_output, content_target in zip(content_outputs, content_targets)])

    # scale the style loss by multiplying by the style weight and dividing by the number of style layers
    style_loss = style_loss * style_weight / num_style_layers 

    # scale the content loss by multiplying by the content weight and dividing by the number of content layers
    content_loss = content_loss * content_weight / num_content_layers 
    
    # sum up the style and content losses
    total_loss = style_loss + content_loss 
    return total_loss

#%% style transfer

def fit_style_transfer(style_image, content_image, style_weight, content_weight, var_weight, optimizer, epochs, steps_per_epoch):
    images = []
    losses = []
    step = 0
    style_targets = get_style_image_features(style_image)
    content_targets = get_content_image_features(content_image)

  # initialize the generated image for updates
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image) 
  
  # incrementally update the content image with the style features
    for n in range(epochs):
        images.append(generated_image)  # append generated image from the previous epoch
        for m in range(steps_per_epoch):
            step += 1

            with tf.GradientTape() as tape:
                style_features = get_style_image_features(generated_image) 
                content_features = get_content_image_features(generated_image) 
                loss = get_style_content_loss(style_targets, style_features, content_targets, 
                                              content_features, style_weight, content_weight)
              # add the total variation loss
                loss += var_weight*tf.image.total_variation(generated_image)

            gradients = tape.gradient(loss, generated_image)  
            optimizer.apply_gradients([(gradients, generated_image)]) 
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 255.0))
            losses.append(loss)

            print(".", end='')
    
        print("Train step: {}".format(step), 'Learning rate = ', optimizer._decayed_lr('float32').numpy(), 'loss = ', loss.numpy()[0])
        if losses[-1].numpy()[0] > (0.99*losses[-steps_per_epoch].numpy()[0]):
            images.append(generated_image)
            generated_image = tf.cast(generated_image, dtype=tf.uint8)
            print('terminating early at the end of epoch', n+1)
            return generated_image, images, losses  
  
    images.append(generated_image)  # append final generated image
    # convert to uint8 (expected dtype for images with pixels in the range [0,255])
    generated_image = tf.cast(generated_image, dtype=tf.uint8)

    return generated_image, images, losses

#%% main

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] 
content_layers = ['block5_conv2'] 
output_layers = style_layers + content_layers 
vgg = vgg_model(output_layers)
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_weight =  1e-4
content_weight = 1e-16 
var_weight = 1e-4

resolution = (1080, 1920)
style_images = os.listdir('style_images/')
content_images = os.listdir('content_images/')

for style in style_images:
    for content in content_images:
        style_image = img_to_array(load_img('style_images/' + style, 
                                              target_size = resolution, interpolation = 'bilinear'), dtype = 'uint8')
        content_image = img_to_array(load_img('content_images/' + content, 
                                              target_size = resolution, interpolation = 'bilinear'), dtype = 'uint8')
        
        retain_color_scheme = True  # change to False if color scheme of style image is to be used
        if retain_color_scheme:
            processed_style_image = match_histograms(style_image, content_image, multichannel = True)
            display_images([content_image, style_image, processed_style_image], ['content image', 'style image', 'style image with content image colors'])
            style_image = tf.convert_to_tensor([processed_style_image], dtype = 'uint8')
        else:    
            display_images([content_image, style_image], ['content image', 'style image'])
            style_image = tf.convert_to_tensor([style_image], dtype = 'uint8')  
        content_image = tf.convert_to_tensor([content_image], dtype = 'uint8') 
        adam = tf.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 5.0, decay_steps = 100, decay_rate = 0.9))
        print(f'starting transfer for content file = {content}, style file = {style}')
        stylized_image, all_generated_images, losses = fit_style_transfer(style_image = style_image, content_image = content_image, 
                                                style_weight = style_weight, content_weight = content_weight,
                                                var_weight = var_weight, optimizer = adam, epochs = 20, steps_per_epoch = 100)
        
        # saves stylized image in folder stylized_images
        plt.imsave(f'stylized_images/content_{content}_style_{style}.jpg', stylized_image[0].numpy())
        # displays the content, style, and stylized image
        display_images([content_image[0], style_image[0], stylized_image[0]], ['content image', 'style image', 'stylized image'])
