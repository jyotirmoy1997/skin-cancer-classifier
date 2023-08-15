import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2hsv
from skimage.filters import gaussian
import segmentation



def blackhat(img):
   rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
   hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)
   return hat

def tophat(img):
   rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
   hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
   return hat

def findThreshold(img):
   fig, ax = plt.subplots(1, 1)
   y,x,_ = ax.hist(img.ravel(), bins=256, range=[0, 255])
   x_index = x[np.argmax(y)]
   y_mean = y.mean()
   index = int(x_index)
   for i in range (index, 256):
      if(y_mean > y[np.where(x==x[i])]):
         index = i
         break
   ax.axvline(x[index], color='k', linestyle='dashed', linewidth=1)
   plt.show()
   return index

# masking
def connected_component_label(img):
   connectivity = 4
   numLabels,labels,stats,_ = cv2.connectedComponentsWithStats(img,connectivity,cv2.CV_32S)
   area = np.mean(stats[1:, cv2.CC_STAT_AREA])
   mask = np.zeros(img.shape, dtype="uint8")
   for i in range(1,numLabels):
      if stats[i, cv2.CC_STAT_AREA]>area :
         mask[labels == i] = 255
   maskarea = np.sum(mask) / 255
   return maskarea,mask

def removeNestedContours(binary):
   # Find the contours in the binary image
   contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   # Get the indices of the contours that do not have a parent (i.e. they are not nested)
   unique_indices = np.where(hierarchy[0][:,3] == -1)[0]

   # Get the unique contours from the indices
   unique_contours = [contours[i] for i in unique_indices]

   # creating white mask
   mask = np.zeros_like(binary)
   # set all pixels within contours to 255
   cv2.drawContours(mask, unique_contours, -1, 255, -1)
   return mask


def testHoughLines(img):
   # Convert the image to grayscale
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # Apply Canny edge detection
   edges = cv2.Canny(gray, 50, 150, apertureSize=3)
   # Apply Hough Line Transform to detect hairlines
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=25, maxLineGap=5)
   if lines is not None:
      return len(lines)
   else:
      return 0


def process(original, fldrpath, filename): #path = filename

   original = cv2.imread(original)
   # cv2.imwrite(f'{fldrpath}\{filename}.png',original)

   # converting to LAB color space
   lab= cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
   l_channel, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   cl = clahe.apply(l_channel)
   limg = cv2.merge((cl,a,b))

   # converting back to original
   enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

   enhanced_r = enhanced_img[:,:,2]
   enhanced_b = enhanced_img[:,:,0]
   # cv2.imwrite(f'{fldrpath}\{filename} Enhanced R Channel.png',enhanced_r)


   black = blackhat(enhanced_r)
   # cv2.imwrite(f'{fldrpath}\{filename} Blackhat B.png',black)
   ret,threshB = cv2.threshold(black,findThreshold(black),255,cv2.THRESH_BINARY)
   # cv2.imwrite(f'{fldrpath}\{filename} Binarization B.png',threshB)



   # filtering
   hairareaB,filteredB = connected_component_label(threshB)
   # cv2.imwrite(f'{fldrpath}\{filename} After Filtering B.png',filteredB)

   # inpaint the original image depending on the mask
   resultB = cv2.inpaint(original,filteredB,1,cv2.INPAINT_TELEA)
   # cv2.imwrite(f'{fldrpath}\{filename} InPaint B.png',resultB)


   #TOPHAT
   # top = tophat(original[:,:,0])
   top = tophat(enhanced_b)
   # cv2.imwrite(f'{fldrpath}\{filename} Tophat T.png',top)
   ret,threshT = cv2.threshold(top,findThreshold(top),255,cv2.THRESH_BINARY)
   # cv2.imwrite(f'{fldrpath}\{filename} Binarization T.png',threshT)

   # filtering
   hairareaT,filteredT = connected_component_label(threshT)
   # cv2.imwrite(f'{fldrpath}\{filename} After Filtering T.png',filteredT)

   # inpaint the original image depending on the mask
   resultT = cv2.inpaint(original,filteredT,1,cv2.INPAINT_TELEA )
   # cv2.imwrite(f'{fldrpath}\{filename} InPaint T.png',resultT)

   # segmentation
   binary_image1 = segmentation.segmentation(resultB)
   binary_image2 = segmentation.segmentation(resultT)

   binary_image1 = removeNestedContours(binary_image1)
   cv2.imwrite(f'{fldrpath}\{filename} Mask B.png',binary_image1)
   binary_image2 = removeNestedContours(binary_image2)
   cv2.imwrite(f'{fldrpath}\{filename} Mask T.png',binary_image2)

   white_pixels1 = cv2.countNonZero(binary_image1)
   white_pixels2 = cv2.countNonZero(binary_image2)

   if(white_pixels1 < white_pixels2):
      minmask = binary_image1
   else:
      minmask = binary_image2

   # minmask = binary_image1

   if testHoughLines(original) > 50 :
      if testHoughLines(resultB) < testHoughLines(resultT):
            masked_img = cv2.bitwise_and(resultB, resultB, mask=minmask)
      else:
            masked_img = cv2.bitwise_and(resultT, resultT, mask=minmask)
   else:
      masked_img = cv2.bitwise_and(original, original, mask=minmask)

   cv2.imwrite(f'{fldrpath}\{filename}_masked.png',masked_img)
   
