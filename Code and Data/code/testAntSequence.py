import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.001, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.05, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

frame = seq[:,:,0]
start = time.time()
for i in range(seq.shape[2]-1):
    print(i)
    next_frame = seq[:,:,i+1]
    mask = SubtractDominantMotion.SubtractDominantMotion(frame,next_frame,threshold,num_iters,tolerance)
    temp_img = np.zeros((next_frame.shape[0],next_frame.shape[1],3))
    for kk in range(3):
        temp_img[:,:,kk] = next_frame
    
    temp_img[:,:,2][mask==1] = 1  ### to make blue
         
    
    if i in [29,59,89,119]:
        fig, ax = plt.subplots()
        ax.imshow(temp_img)
        plt.show()
        
    frame = next_frame
print("Time Taken - ", time.time()-start)
