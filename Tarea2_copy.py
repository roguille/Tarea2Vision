#!/usr/bin/env python

import numpy as np
import cv2
import pymeanshift as pms
import json
from sys import exit
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import watershed, disk
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import data, segmentation, color
from skimage.future import graph
import argparse

cap = cv2.VideoCapture(0)

printMenu = False
mySegmenter = pms.Segmenter()
mkeyPress = False
wKeyPress = False

#params = {}
#params['ms'] = []
#params['ms'].append({
#    'spatial_radius': mySegmenter.spatial_radius,
#    'range_radius': mySegmenter.range_radius,
#    'min_density': mySegmenter.min_density
#})

#with open('params.txt', 'w') as outfile:
#    json.dump(params, outfile)

with open('params.txt', 'r') as json_file:
    params = json.load(json_file)

for ms in params['ms']:
    mySegmenter.spatial_radius = ms['spatial_radius']
    mySegmenter.range_radius = ms['range_radius']
    mySegmenter.min_density = ms['min_density']

while(True):

    if not printMenu:
        print("This program performs image segmentation of the images taken with the webcam.")
        print("Press ms in image window to execute image segmentation using mean shift algorithm.")
        print("Press ws in image window to execute image segmentation using watersheds algorithm.")
        print("Press ns in image window to execute image segmentation using Normalized Cut Segments algorithm.")
        print("Press q in image window for exit.")
        printMenu = True

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Tarea2: Segmentacion de imagen',frame)

    key = cv2.waitKey(2)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('m'):
        mKeyPress = True
        wKeyPress = False
        rKeyPress = False
    elif key & 0xFF == ord('r'):
        mKeyPress = False
        wKeyPress = False
        rKeyPress = True
    elif key & 0xFF == ord('w'):
        wKeyPress = True
        mKeyPress = False
        rKeyPress = False
    elif key & 0xFF == ord('s') and mKeyPress:
        print("Using Mean Shift Algorithm!")
        (segmentedImage, labelsImage, numberRegions) = mySegmenter(frame)
        cv2.imshow('Tarea2: Imagen Segmentada', segmentedImage)
        print("Number of regions: " + str(numberRegions))
        print("Mask:")
        print(str(len(labelsImage)) + "x" + str(len(labelsImage[0])))
        print(labelsImage)
        mKeyPress = False
    elif key & 0xFF == ord('s') and wKeyPress:
        print("Using Watersheds Algorithm!")
        img = frame
        img_gray = rgb2gray(img)

        image = img_as_ubyte(img_gray)


        #Calculate the local gradients of the image
        #and only select the points that have a
        #gradient value of less than 20
        markers = rank.gradient(image, disk(5)) < 20
        markers = ndi.label(markers)[0]

        gradient = rank.gradient(image, disk(2))

        #Watershed Algorithm
        labels = watershed(gradient, markers)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box'})
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[0].set_title("Original")

        ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral, interpolation='nearest')
        ax[1].set_title("Local Gradient")

        ax[2].imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
        ax[2].set_title("Markers")

        ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest', alpha=.7)
        ax[3].set_title("Segmented")


        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()
        #cv2.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        #cv2.imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest', alpha=.7)
        wKeyPress = False

        # normalized_cut_segments method
    elif key & 0xFF == ord('s') and rKeyPress:
        print("Using Normalized cut segments Algorithm!")
        img = frame

        img_segments = segmentation.slic(img, compactness=30, n_segments=200)
        out1 = color.label2rgb(img_segments, img, kind='avg')

        segment_graph = graph.rag_mean_color(img, img_segments, mode='similarity')
        img_cuts = graph.cut_normalized(img_segments, segment_graph)
        normalized_cut_segments = color.label2rgb(img_cuts, img, kind='avg')

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 8))

        ax[0].imshow(img)
        ax[1].imshow(normalized_cut_segments)

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.show()
        rKeyPress = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
