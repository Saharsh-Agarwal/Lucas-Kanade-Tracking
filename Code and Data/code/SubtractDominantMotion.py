import numpy as np
import LucasKanadeAffine as lka
import InverseCompositionAffine as ica
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import affine_transform as at
import matplotlib.pyplot as plt

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    
    # put your implementation here
    mask = None
    mask = np.zeros(image1.shape, dtype=bool)
    
    #M = lka.LucasKanadeAffine(image1,image2,threshold,num_iters)
    
    M = ica.InverseCompositionAffine(image1,image2,threshold*200,num_iters)
    
    M = np.vstack((M, np.asarray([[0,0,1]])))
    M = np.linalg.inv(M)
    
    spline_image1 = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    spline_image2 = RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)

    x = np.arange(0, image2.shape[1])
    y = np.arange(0, image2.shape[0])
    xx, yy = np.meshgrid(x, y)
    X = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
    Y = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]

    invalid = (X < 0) | (X >= image1.shape[1]) | (Y < 0) & (Y >= image1.shape[0])
    
    I1 = spline_image1.ev(Y, X)
    I2 = spline_image2.ev(Y, X)
    I1[invalid] = 0
    I2[invalid] = 0
    #print(I1.shape)
    #plt.imshow(I1)
    #plt.show()

    # calculate the difference
    diff = np.absolute(I2 - I1)
    ind = (diff > tolerance) & (I2 != 0)
    mask[ind] = 1
    #print(mask.shape)
    
    '''
    aerial :
    mask = binary_erosion(mask, structure=np.eye(2), iterations=1)
    mask = binary_erosion(mask, structure=np.ones((1,3)), iterations=1)
    st = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
    mask = binary_dilation(mask, structure=st, iterations = 2)
    '''
    #mask = binary_erosion(mask, structure=np.eye(2), iterations=1)
    #mask = binary_erosion(mask, structure=np.ones((1,3)), iterations=1)
    st = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
    mask = binary_dilation(mask, structure=st, iterations = 1)
    #plt.imshow(mask)
    #plt.show()
    
    return mask 
"""
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

'''
def subt(image1, image2, threshold, num_iters, tolerance):
    
    mask = np.zeros(image1.shape, dtype=bool)
    M = lka.LucasKanadeAffine(image1,image2,threshold,num_iters)
    
    M = np.vstack((M, np.asarray([[0,0,1]])))
    M = np.linalg.inv(M)
    
    
    Iw = at(image1, M[:2,:2], offset=M[:2,2], output_shape=image2.shape)
    d = np.absolute(Iw-image2)

    mask[d > tolerance] = 1
    mask[Iw == 0] = 0

    plt.imshow(mask)
    plt.show()
    #print(mask)
    
    #mask = binary_erosion(mask, structure=np.eye(2), iterations=1)
    st = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
    mask = binary_dilation(mask, structure=st, iterations = 2)
    plt.imshow(mask)
    plt.show()
'''
'''
seq = np.load('../data/antseq.npy')
frame = seq[:,:,59]
nextf = seq[:,:,60]
M = SubtractDominantMotion(frame, nextf, threshold = 0.001, num_iters=100, tolerance=0.025)
#M_ = subt(frame, nextf, threshold = 0.01, num_iters=1000, tolerance=0.075)
'''