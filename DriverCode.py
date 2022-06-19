import cv2
from PIL import Image
import ImpModules
import os
import matplotlib.pyplot as plt

# CHANGE TO CORRECT DIRECTORY

dir1 = "C:\\Users\\UserName\\'FileName'.mp4" # Input the directory 

from scipy import ndimage

vidcap = cv2. VideoCapture(dir1)
box = (340, 220 , 420, 300) # cropping the image
count = 0 
success,image = vidcap.read()

while success: # run in a loop through the entire MP4 file
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,ogArr = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

    ogIm = Image.fromarray(ogArr) # The original Image
    ogcropped = ogIm.crop(box) # Crop the image 

    GaussArr = ImpModules.ApplyGaussian(ogArr, 1.5) # Apply a Gaussian to the array of the original Image
    GaussIm = Image.fromarray(GaussArr) # Turn that into a Gaussian

    sobArr, theta = ImpModules.ApplySobel(GaussArr) # Sobel filters as well as the angles (direction of the edge)

    nonmax = ImpModules.non_max_suppression(sobArr,theta) # A non max supression algorithm

    res, weak, strong = ImpModules.threshold(nonmax, 0.01, 0.1) # Apply a threshold to filter out not required pixels

    hy = ImpModules.hysteresis(res, weak) # Apply hysteresis to make the edges smooth and remove unnesseccary pixels

    resim = Image.fromarray(res)
    hyim = Image.fromarray(hy)

    # Plotting everything

    plt.subplot(3,3,1), plt.imshow(ogIm), plt.title("Original image")
    plt.subplot(3,3,2), plt.imshow(ogcropped), plt.title("Original Cropped")
    plt.subplot(3,3,3),plt.imshow(GaussIm), plt.title("Gaussian")
    plt.subplot(3,3,4),plt.imshow(sobArr), plt.title("Sobel")
    plt.subplot(3,3,5), plt.imshow(nonmax), plt.title("nonmax")
    plt.subplot(3,3,6),plt.imshow(resim), plt.title("Threshold")
    plt.subplot(3,3,7),plt.imshow(hyim), plt.title("hysterisis")

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/') # put results in folder in the directory
    sample_file_name = str(count)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)


    success = False
    plt.show()
    #plt.savefig(results_dir + sample_file_name)
    


