import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) 
    
    i = 0 
    xl,yl,xr,yr = 0,0,It.shape[0]-1,It.shape[1]-1
    del_p = threshold+1
    
    # current image
    X1 = np.arange(0, It1.shape[0], 1)
    Y1 = np.arange(0, It1.shape[1], 1)   
    It_real_spline = RectBivariateSpline(X1,Y1,It1)
    ### print("Real",len(X1),len(Y1)) - 240 by 320
    
    # Template - t
    X = np.arange(0, It.shape[0], 1)
    Y = np.arange(0, It.shape[1], 1)   
    It_temp_spline = RectBivariateSpline(X,Y,It)
    
    # Meshing of Template - threshold
    xtemp_patchlen = np.arange(xl, xr + 0.01)
    ytemp_patchlen = np.arange(yl, yr + 0.01)
    xtemp, ytemp = np.meshgrid(xtemp_patchlen, ytemp_patchlen)
    
    template = It_temp_spline.ev(ytemp, xtemp)
    
    #Derivatives
    dItx = It_temp_spline.ev(ytemp, xtemp, 0, 1).flatten()
    dIty = It_temp_spline.ev(ytemp, xtemp, 1, 0).flatten()

    dItx = np.expand_dims(dItx, axis = 1)
    dIty = np.expand_dims(dIty, axis = 1)
    
    xt = np.expand_dims(xtemp.flatten(), axis = 1)
    yt = np.expand_dims(ytemp.flatten(), axis = 1)
    
    A = np.hstack((dItx*xt, dItx*yt, dItx, dIty*xt, dIty*yt, dIty))
    H = A.T@A
    
    while (i<num_iters and del_p >= threshold):
        #print(del_p, threshold)
        xreal, yreal = np.meshgrid(xtemp_patchlen, ytemp_patchlen)
        ### For inside the frame 
        idx = (xreal > 0) & (xreal < It.shape[1]) & (yreal>0) & (yreal<It.shape[0])
        ### Warped 
        xreal = xreal[idx]
        yreal = yreal[idx]
        xt1 = M[0,0]*xreal + M[0,1]*yreal + M[0,2]
        yt1 = M[1,0]*xreal + M[1,1]*yreal + M[1,2]
        
        real_patch = It_real_spline.ev(yt1,xt1)

        ## try
        #xt1 = np.expand_dims(xt1.flatten(), axis = 1)
        #yt1 = np.expand_dims(yt1.flatten(), axis = 1)
        
        b = template[idx] - real_patch

        dp = np.linalg.pinv(H)@A[idx.flatten()].T@b.flatten()
        dm = np.reshape(dp, (2,3))
        dm = dm + np.asarray([[1,0,0],[0,1,0]])
        
        M = np.vstack((M,np.asarray([[0,0,1]])))
        dm = np.vstack((dm,np.asarray([[0,0,1]])))
        
        M =( M @ np.linalg.pinv(dm))[:2,:]
        del_p = np.linalg.norm(dp)
        i = i+1
        
        #print(del_p, "LK-ICA")
        
    return M

'''
seq = np.load('../data/aerialseq.npy')
frame = seq[:,:,0]
nextf = seq[:,:,1]
M = LucasKanadeAffine(frame, nextf, threshold = 0.01, num_iters=1000)
print(M)
'''

"""
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """