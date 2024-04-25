# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:56:08 2020

@author: Jalba
"""
import sys
import os
import numpy as np
import cv2 as cv

def compressPCA(pcaset, mean, maxComp, testset):
    
    mean, eigenvectors, eigenvalues = cv.PCACompute2(pcaset, mean, maxComponents = maxComp) # compute PCA decomposition
    compressed = cv.PCAProject(testset,mean,eigenvectors)  # project testset on PCA subspace
    reconstructed = cv.PCABackProject(compressed,mean,eigenvectors) # recover testset from PCA subspace
    mean_error = cv.norm(testset, reconstructed, cv.NORM_L2) # compute mean reconstruction error
    print('Projection dimension = ', maxComp , ' mean reconstruction error = ', mean_error / testset.shape[0])
    for i in range(maxComp):
         print('eigenvalue: ', eigenvalues[i])
         print('eigenvector: ', eigenvectors[i])

    return compressed



#%%  ///////////////////////////////////////////////////////////////
#    /// First part: reading images and printing their real size ///
#    ///////////////////////////////////////////////////////////////


print('Sample code for project2')

image_file = input('Introduce a microscopy image: ')
if not os.path.isfile(image_file):
    print('The file ', image_file, 'does not exist')
    sys.exit(1)
    
img = cv.imread(image_file)
if img is None:
    print('The file ', image_file, ' doesn\'t contain an image')
    sys.exit(1)


cv.namedWindow("Original image", cv.WINDOW_NORMAL)
cv.imshow('Original image', img)   # Show our image inside it.
cv.waitKey(0)
cv.destroyAllWindows()


# Show image size in the Console

print('Image Size in pixels :  ', img.shape)
print('Image Size in mm: ', img.shape[0] / 2100.0, ' x ', img.shape[1] / 2100.0 ) 


#%%  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#    /// Second part: Trying clustering in the RGB space: this piece of code shows the more common colors in the image ///
#    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS

data = np.float32(img.flatten().reshape(img.shape[0]*img.shape[1],3))   # reshaping the image to accomodate it as a data matrix
                                                            
for K in range(2,8):                                              # Testing with increasing K
    print('K = ', K)
    compactness, labels, centers = cv.kmeans(data, K, None, criteria, 3, flags) # compactness is the measurement of how compact are the clusters around their centroids
    centers = np.uint8(centers)
    clustered_image = centers[labels.flatten()] #   This is the clustered image, where each pixel is assigned to the cluster of its center
    clustered_image = clustered_image.reshape(img.shape) # Reshaping the clustered image to the original image shape
    cv.namedWindow("Clustered image", cv.WINDOW_NORMAL)
    print('centers: ')
    for k in range(0,K):                                             # Show in console the BGR values of these centers
        print(centers[k]) 

    file_name = 'clustered_image_' + str(K) + '.png'  # comment this block if you don't want to save the clustered image
    print(file_name)
    cv.imwrite(file_name, clustered_image)
    
    print('Compactness = ', compactness)        # Show in console the measurement of how compact are the clusters around their centroids
    print('Compactness*K = ', compactness*K)    # Using a simple measurement to find when to stop increasing clusters (Colors)
    cv.imshow('Clustered image', clustered_image)
    cv.waitKey(0)
cv.destroyAllWindows()    


#%%  ////////////////////////////////////////////////////////////////////////////////////
#    /// Third part: Trying binarization and thresholding in different color channels ///
#    ////////////////////////////////////////////////////////////////////////////////////


    #BGR channels:
    
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

cv.imshow("Blue Channel", B);
cv.imshow("Green Channel", G);
cv.imshow("Red Channel", R);

# it seems that adding blue and green channel, nuclei are highlighted
srcBG = B+G
srcBG[srcBG < np.minimum(B,G)] = 255    # This trick to truncate, instead of overflowing the uitn8 type
cv.imshow("B+G channel", srcBG)
th, dst = cv.threshold(srcBG, 0, 255, cv.THRESH_OTSU)        # Otsu thresholding over B+G channel
cv.imshow("Binarized image (Otsu)", dst)
cv.waitKey(0)
cv.destroyAllWindows() 

drawing = np.zeros(img.shape, np.uint8)
contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    #Find Contours of the segmented image
cv.drawContours(drawing, contours, -1, (0,0,255), 2)
cv.imshow("Contours", drawing)
cv.waitKey(0)
cv.destroyAllWindows() 

dst = cv.adaptiveThreshold(srcBG,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,0)    # Adaptive tresholding over B+G channel
cv.imshow("Binarized image (adaptive)", dst);
cv.waitKey(0)
cv.destroyAllWindows() 


#%%  /////////////////////////////////////////////////////////////////////////////////////////////////
#    /// Fourth part: Finding an "optimum" 3D to 1D projection (grayscale image) for thresholding  ///
#    /// We will use PCA (Principal Component Analysis) also called KLT (Karhunen-Loewe Transform) ///
#    /// when applied for projecting in a lower dimensional space                                  ///
#    /////////////////////////////////////////////////////////////////////////////////////////////////


compressed_data = compressPCA(data, np.uint8([]), 3, data)
minVal, maxVal, _ , _ = cv.minMaxLoc(compressed_data) # Finding the min and max values in the compressed data
compressed_image = compressed_data.reshape(img.shape)  # Resizing the destination matrix as the original image
compressed_image = 255.0*(compressed_image - minVal )/(maxVal - minVal) # Normalizing the image to 0-255 range
compressed_image = np.uint8(compressed_image) # Converting to uint8 type
cv.imshow("PCA-rotated image", compressed_image)
cv.waitKey(0)
cv.destroyAllWindows() 

# Show the 3 principal chanels sorted by largest eigenvalue

cv.imshow("PCA-1st projection", compressed_image[:,:,0])
cv.imshow("PCA-2nd projection", compressed_image[:,:,1])
cv.imshow("PCA-3rd projection", compressed_image[:,:,2])


minVal, maxVal,_ ,_ = cv.minMaxLoc(compressed_image[:,:,0])
thOTSU, dst = cv.threshold(compressed_image[:,:,0], minVal, maxVal, cv.THRESH_OTSU)
cv.imshow("Binarized PCA image (Otsu)", dst)
cv.waitKey(0)
cv.destroyAllWindows() 
compressed_data = compressPCA(data, np.uint8([]), 2, data); # just to check the reconstruction error when discarding the 3rd dimension
compressed_data = compressPCA(data, np.uint8([]), 1, data); # just to check the reconstruction error when discarding the 2nd and 3rd dimensions



