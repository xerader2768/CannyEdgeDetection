from matplotlib.image import imread
import matplotlib.pyplot as plt
import scipy.ndimage as scn
import numpy as np
from scipy import ndimage
import scipy
from PIL import Image

# This module has all the modules used in the Driver Code


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D 
    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 70 # The parameters
                r = 70
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] <= q) and (img[i,j] <= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

# #---------------------------------------------------------------------------------------------------------------------
# PART I - Transforming an image from color to grayscale
# #---------------------------------------------------------------------------------------------------------------------

def TurnGray(image_file):
    # Here we import the image file as an array of shape (nx, ny, nz)
    input_image = image_file # this is the array representation of the input image
    [nx, ny, nz] = np.shape(input_image)  # nx: height, ny: width, nz: colors (RGB)

    # Extracting each one of the RGB components
    r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]

    # The following operation will take weights and parameters to convert the color image to grayscale
    gamma = 1  # a parameter
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
    grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma

  
    # This command will display the grayscale image alongside the original image
    return grayscale_image

# #---------------------------------------------------------------------------------------------------------------------
# PART II - Applying the Sobel operator
# #---------------------------------------------------------------------------------------------------------------------

"""
The kernels Gx and Gy can be thought of as a differential operation in the "input_image" array in the directions x and y 
respectively. These kernels are represented by the following matrices:
      _               _                   _                _
     |                 |                 |                  |
     | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
     | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
     |_               _|                 |_                _|
"""
def ApplySobel(grayscale_image):
    # Here we define the matrices associated with the Sobel filter
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
    sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)
    theta = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

    # Now we "sweep" the image in both x and y directions and compute the output
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
            theta[i+1, j+1] = np.arctan2(gy, gx)
    # Display the original image and the Sobel filtered image
    return sobel_filtered_image, theta

    # Show both images
    # plt.show()

    # Save the filtered image in destination path
    # plt.imsave('sobel_filtered_image.png', sobel_filtered_image, cmap=plt.get_cmap('gray'))

def ApplyGaussian(imageArr, sig): # Returns grayscaled and gaussianed image

    box = (340, 220 , 420, 300)
    imgrayarray = TurnGray(imageArr)

    GF = scn.gaussian_filter(imgrayarray, sigma = sig) # apply the filter
  

    image2 = Image.fromarray(GF)
    box = (340, 220 , 420, 300)
    image2 = image2.crop(box)
    GF = np.asarray(image2)
    return GF 

def threshold(img, lowThresholdRatio=0.5, highThresholdRatio=1): # Code for removing unnessacary pixels
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25) # pixels between 25 and 255 are kept as weak. Above 255 are strong
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255): # checks if neighboring pixels are strong as well. If they are keep them 
                                    # If not, we remove them 
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img