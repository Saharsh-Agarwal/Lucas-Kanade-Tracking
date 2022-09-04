import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) #add last [0.0, 0.0, 1.0]
    
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
    
    while (i<num_iters and del_p >= threshold):
        
        # Meshing of Template - threshold
        xtemp_patchlen = np.arange(xl, xr + 0.01)
        ytemp_patchlen = np.arange(yl, yr + 0.01)
        xtemp, ytemp = np.meshgrid(xtemp_patchlen, ytemp_patchlen)
        template = It_temp_spline.ev(ytemp, xtemp)
        
        ### Warped 
        xt1 = M[0,0]*xtemp + M[0,1]*ytemp + M[0,2]
        yt1 = M[1,0]*xtemp + M[1,1]*ytemp + M[1,2]
        
        ### For inside the frame 
        idx = (xt1 > 0) & (xt1 < It.shape[1]) & (yt1>0) & (yt1<It.shape[0])
        
        xt1 = xt1[idx]
        yt1 = yt1[idx]
        xtemp = xtemp[idx]
        ytemp = ytemp[idx]
        real_patch = It_real_spline.ev(yt1,xt1)

        #Derivatives
        dIt1x = It_real_spline.ev(yt1, xt1, 0, 1).flatten()
        dIt1y = It_real_spline.ev(yt1, xt1, 1, 0).flatten()

        dIt1x = np.expand_dims(dIt1x, axis = 1)
        dIt1y = np.expand_dims(dIt1y, axis = 1)
        
        ## try
        xt1 = np.expand_dims(xtemp.flatten(), axis = 1)
        yt1 = np.expand_dims(ytemp.flatten(), axis = 1)
        
        #xt1 = np.expand_dims(xt1.flatten(), axis = 1)
        #yt1 = np.expand_dims(yt1.flatten(), axis = 1)

        A = np.hstack((dIt1x*xt1, dIt1x*yt1, dIt1x, dIt1y*xt1, dIt1y*yt1, dIt1y))
        b = template[idx] - real_patch

        dp = np.linalg.lstsq(A,b.flatten(),rcond=None)[0]
        M = M + np.reshape(dp, (2,3))

        del_p = np.linalg.norm(dp)
        i = i+1
        #print(i, M)
        print(i, "LKA")
        
    return M

"""
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

'''
seq = np.load('../data/aerialseq.npy')
frame = seq[:,:,0]
nextf = seq[:,:,1]
M = LucasKanadeAffine(frame, nextf, threshold = 0.01, num_iters=1000)
print(M)
'''