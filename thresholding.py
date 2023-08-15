from scipy.stats import gamma, entropy
import numpy as np
from skimage.filters import threshold_otsu

def fuzzy_divergence_thresholding(gray):
    with np.errstate(invalid='ignore'):
        # Compute the Otsu threshold as the initial guess
        otsu_thresh = threshold_otsu(gray)

        # Define the range of candidate threshold values
        thresh_range = np.arange(otsu_thresh - 0.1, otsu_thresh + 0.1, 0.01)

        # Compute the fuzzy histogram
        fuzzy_hist = np.zeros((256, 2))
        for i in range(256):
            # Use gamma distribution to compute the membership function
            # fuzzy_hist[i][0] = gamma.pdf(i, gray.flat[i], scale=20)
            fuzzy_hist[i][0] = np.exp(-((i - gray.flat[i]) / 20) ** 2)
            fuzzy_hist[i][1] = 1 - fuzzy_hist[i][0]

        # Compute the divergence between the fuzzy histograms
        divergence = np.zeros(len(thresh_range))
        for i, thresh in enumerate(thresh_range):
            foreground = fuzzy_hist[:int(thresh * 255), 0].sum() + fuzzy_hist[int(thresh * 255):, 1].sum()
            background = fuzzy_hist[:int(thresh * 255), 1].sum() + fuzzy_hist[int(thresh * 255):, 0].sum()
            divergence[i] = foreground * np.log(foreground / (foreground + background)) + background * np.log(background / (foreground + background))

        # Find the threshold value that minimizes the divergence
        # min_div_idx = (np.argmin(divergence)+np.argmax(-divergence))//2
        min_div_idx = np.argmin(divergence)
        threshold = thresh_range[min_div_idx]

    return threshold