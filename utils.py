import cv2
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from sklearn.cluster import MiniBatchKMeans
from numpy.lib.stride_tricks import as_strided

def quantize32levels_image(image):
    (h, w) = image.shape[:2]
    image = image.reshape((image.shape[0] * image.shape[1], 1))

    clt = MiniBatchKMeans(n_clusters = 32, max_iter=10)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w))
    return quant

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    
def compute_GLCM(img):
    img = cv2.resize(img,(256,256))
    plt.imshow(img, cmap="gray")
    #perform transformation from 256 to 32 colors 
    img = quantize32levels_image(img)
    plt.imshow(img, cmap="gray")
    #GLCM extraction degrees 0, 45, 90, 135
    radians = [0.0,0.785398,1.5708,2.35619]
    glcm0 = greycomatrix(img, angles=[radians[0]],distances=[1.])
    glcm45 = greycomatrix(img, angles=[radians[1]],distances=[1.])#,levels=32
    glcm90 = greycomatrix(img, angles=[radians[2]],distances=[1.])#,levels=32
    glcm135 = greycomatrix(img, angles=[radians[3]],distances=[1.])#,levels=32

    Sg0 = greycoprops(glcm0, 'energy')[0, 0]
    Sg0_ = -np.log(Sg0)
    Sg45 = greycoprops(glcm45, 'energy')[0, 0]
    Sg45_ = -np.log(Sg45)
    Sg90 = greycoprops(glcm90, 'energy')[0, 0]
    Sg90_ = -np.log(Sg90)
    Sg135 = greycoprops(glcm135, 'energy')[0, 0]
    Sg135_ = -np.log(Sg135)
    Sp0 = shannon_entropy(glcm0)
    Sp45 = shannon_entropy(glcm45)
    Sp90 = shannon_entropy(glcm90)
    Sp135 = shannon_entropy(glcm135)

    energy = (Sg0+Sg45+Sg90+Sg135)/4.0
    entropy = (Sp0+Sp45+Sp90+Sp135)/4.0
    
    # mean
    # S = (Sg0_+Sg45_+Sg90_+Sg135_)/4.0+(Sp0+Sp45+Sp90+Sp135)/4.0
    
    # Max pooling of matrix (Just for a better visualization)
    glcm = pool2d(glcm0[:,:,0,0], kernel_size=2, stride=2, padding=0, pool_mode='max')
    glcm = pool2d(glcm, kernel_size=2, stride=2, padding=0, pool_mode='max')
    return energy, entropy, glcm