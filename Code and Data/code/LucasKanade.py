import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
	
    # Put your implementation here
    p = p0
    xl,yl,xr,yr = rect # l is left and r is right
    i = 0
    
    # Adding 1 to include last values
    #RectBivariateSpline - X and Y length ke image patch pe spline/ curve 
    #                      fit karna taaki interpolate kar sake
    # .ev - evaluates the value at spline -> derivatives can also be checked
    
    # current image
    X1 = np.arange(0, It1.shape[0], 1)
    Y1 = np.arange(0, It1.shape[1], 1)   
    It_real_spline = RectBivariateSpline(X1,Y1,It1)
    ### print("Real",len(X1),len(Y1)) - 240 by 320
    
    # Template - t
    X = np.arange(0, It.shape[0], 1)
    Y = np.arange(0, It.shape[1], 1)   
    It_temp_spline = RectBivariateSpline(X,Y,It)
    ## print("Template",len(X),len(Y))
    
    del_p = threshold + 1 # random value just so that begining above threshold
    
    while (i < num_iters and del_p > threshold):
        # Moving through Real Image and Meshing
        xreal_patchlen = np.arange(xl + p[0], xr + p[0] + 0.01)
        yreal_patchlen = np.arange(yl + p[1], yr + p[1] + 0.01)
        xreal, yreal = np.meshgrid(xreal_patchlen, yreal_patchlen)
        patch_real = It_real_spline.ev(yreal, xreal)
        #print(i, "Real Patch", patch_real.shape)
        
        # Meshing of Template - t
        xtemp_patchlen = np.arange(xl, xr + 0.01)
        ytemp_patchlen = np.arange(yl, yr + 0.01)
        xtemp, ytemp = np.meshgrid(xtemp_patchlen, ytemp_patchlen)
        template = It_temp_spline.ev(ytemp, xtemp) #spline values for all (y,x)
        ## print(template.shape, "temp shape")
        
        #Derivatives
        dIt1x = It_real_spline.ev(yreal, xreal, 0, 1).flatten()
        dIt1y = It_real_spline.ev(yreal, xreal, 1, 0).flatten()
        
        A = np.hstack((np.expand_dims(dIt1x, axis = 1), np.expand_dims(dIt1y, axis = 1)))
        b = template - patch_real
        dp = np.linalg.lstsq(A,b.flatten(),rcond=None)[0]
        p[0] = p[0] + dp[0]
        p[1] = p[1] + dp[1]
        del_p = np.sqrt(dp[0]**2 + dp[1]**2)
        i = i + 1
    
    return p
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """