import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

def quantize_channels(x, k):
    quantized_x = x.copy()
    for d in range(3):
        channel = x[:, :, d].copy()
        # k_means = MiniBatchKMeans(k, compute_labels=False, n_init='auto')
        k_means = KMeans(n_clusters=k, random_state=0, n_init='auto')
        k_means.fit(channel.reshape(-1, 1))
        labels = k_means.predict(channel.reshape(-1, 1))
        quantized_x[:,:,d] = labels.reshape(channel.shape)
    return quantized_x

def imquantize(x, levels):
    quantized_img  = quantize_channels(x, levels)
    M, N, _ = x.shape
    Index = np.zeros((M,N)).astype('uint8')
    for i in range(M):
        for j in range(N):
            # (0 0 0)base8 = (0*8^2 + 0*8^1 + 0*8^0)base10
            Index[i][j] = quantized_img[i][j][0] * 64 + quantized_img[i][j][1] * 8 + quantized_img[i][j][2]
            # Index[i][j] = quantized_img[i][j][0] * (levels*levels) + quantized_img[i][j][1] * levels + quantized_img[i][j][2]
    return Index
