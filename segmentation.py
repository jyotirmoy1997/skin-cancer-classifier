import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import color
from skimage import exposure, img_as_float
from skimage.filters import threshold_otsu
import cv2
import thresholding
from sklearn.preprocessing import MinMaxScaler
import imquantize
from sklearn.cluster import KMeans


def segmentation(img):
    n = 0.5; #distance scaling parameter (n) such that 0.1 ≤ n ≤ 1.0
    NIC = 3; # number of image components
    levels = 8; # image quantization level
    NCD = 512; #levels^3; # maximum number of desired colors
    LCL = 0; # lab color light
    LCA = 1; # lab color A
    LCB = 2; # lab color B
    PXC = 3; # pixel x-coordinate
    PYC = 4; # pixel y-coordinate
    PDC = 5; # pixel distance to image center
    CPC = 6; # cluster pixel count
    CPL = 7; # cluster pixel label
    CCC = 7; # cluster color contrast
    CSS = 8; # cluster saliency score

    K = 0

    img_arr = np.array(img)

    #---- Image Quantization ----
    M, N, _ = img_arr.shape

    Index = imquantize.imquantize(img_arr, levels)

    # Convert image to LAB color space
    lab_image = color.rgb2lab(img)

    Input = exposure.rescale_intensity(lab_image, out_range=(0, 1))

    Palette = np.zeros((NCD,9))

    for x in range(M):
        for y in range(N):
            Palette[Index[x, y], CPC] += 1
            Palette[Index[x, y], LCL] += Input[x, y, LCL]
            Palette[Index[x, y], LCA] += Input[x, y, LCA]
            Palette[Index[x, y], LCB] += Input[x, y, LCB]
            Palette[Index[x, y], PXC] += x
            Palette[Index[x, y], PYC] += y

    Cluster = np.zeros((NCD,9))
    for z in range(NCD):
        if Palette[z, CPC] > 0:
            Palette[z, CPL] = K
            Palette[z, PXC] /= M
            Palette[z, PYC] /= N
            Palette[z, 0:PYC+1] /= Palette[z, CPC]
            Cluster[K, 0:CPC+1] = Palette[z, 0:CPC+1]
            K += 1

    Cluster = np.delete(Cluster, range(K, NCD), axis=0)
    Wr = Cluster[:, CPC] / (M * N)

    for x in range(K):
        Cluster[x, CCC] = 0
        for y in range(K):
            Cluster[x, CCC] += Wr[y] * np.linalg.norm(Cluster[x, :NIC] - Cluster[y, :NIC])

    for x in range(M):
        for y in range(N):
            cluster_index = int(Palette[Index[x, y], CPL])
            Cluster[cluster_index, PDC] += ((x / M - 0.5) ** 2) + ((y / N - 0.5) ** 2)

    for z in range(K):
        Cluster[z, PDC] = Cluster[z, PDC]/(n * n * Cluster[z, CPC])

    for x in range(K):
        Cluster[x, CSS] = 0
        for y in range(K):
            Ds = np.linalg.norm(Cluster[x, PXC:PYC] - Cluster[y, PXC:PYC])
            Phixy = (Cluster[x, CCC] + 0.05) / (Cluster[y, CCC] + 0.05)
            Cluster[x, CSS] += Wr[y] * Phixy * np.exp(-Ds)
        
        Cluster[x, CSS] = np.exp(-Cluster[x, PDC]) * (Wr[x] * Cluster[x, CCC] + Cluster[x, CSS])


    # Cluster[:, CSS] = minmax_scale(Cluster[:, CSS])
    Cluster[:, CSS] = MinMaxScaler().fit_transform(np.array(Cluster[:, CSS]).reshape(-1, 1)).ravel()

    for x in range(M):
        for y in range(N):
            Input[x, y, LCL] = Cluster[int(Palette[Index[x, y], CPL]), CSS]

    # Apply Fuzzy Divergence thresholding to Input
    # threshold = threshold_otsu(Input[:, :, LCL])
    threshold = abs(thresholding.fuzzy_divergence_thresholding(Input[:, :, LCL]))
    Output = Input[:, :, LCL] >= threshold

    Output = Output.astype(np.uint8) * 255


   # assuming that the variable "Output" contains the binary image
    Output = cv2.morphologyEx(Output, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Perform connected component labeling and analysis
    totalLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(Output, 4, cv2.CV_32S)

    # Find the center of the image
    h, w = Output.shape
    cx = w // 2
    cy = h // 2

    # Compute the distance of each pixel from the center
    dist = np.sqrt((np.arange(w) - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2)

    # Find the largest component area
    max_area = 0
    for i in range(1, totalLabels):
        area = stats[i][cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            x, y = centroids[i]
    
    min_size = 0.2 * max_area
    max_dist = dist[int(y), int(x)]

    for i in range(1, totalLabels):
        area = stats[i][cv2.CC_STAT_AREA]
        x, y = centroids[i]
        # Compute the distance of the centroid from the center
        d = dist[int(y), int(x)]
        # If the area is smaller than min_size and the distance is larger than max_dist, remove the component
        if area < min_size and d > max_dist:
            Output[labels == i] = 0

    # # Convert binary image into an array of (x,y) coordinates
    # coords = np.column_stack(np.nonzero(Output))

    # # Perform k-means clustering on the coordinates
    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(coords)

    # # Get the cluster labels for each coordinate
    # labels = kmeans.labels_

    # # Find the cluster with the maximum number of components
    # largest_cluster = max(set(labels), key=list(labels).count)

    # # Create an output image with the same shape as the input image
    # out_im = np.zeros_like(Output)

    # # Assign each pixel in the output image to the largest cluster based on its label
    # for i, coord in enumerate(coords):
    #     if labels[i] == largest_cluster:
    #         out_im[coord[0], coord[1]] = 255
    
    # Output = out_im

    return Output


# #  Load image
# img = Image.open("D:\\archive\HAM10000_images_part_1\ISIC_0024481.jpg")
# output = segmentation(img)
# output = Image.fromarray(output.astype('uint8'))
# output.show()