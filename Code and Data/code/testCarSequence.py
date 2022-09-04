import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy") #2Dimage - 3rd dimension is time (240,320) in 415 frames
rect = [59, 116, 145, 151] #bounding box

rects = []
rects.append(rect)

for i in range(seq.shape[2]-1):
    template = seq[:,:,i]
    rect = rects[i]
    p = LucasKanade(template, seq[:,:,i+1], rect, threshold, num_iters, p0=np.zeros(2))
    frame_rect = np.asarray([rect[0]+p[0], rect[1]+p[1],rect[2]+p[0], rect[3]+p[1]])
    rects.append(frame_rect)
    
    width = rect[2]-rect[0]
    height = rect[3]-rect[1]
    ### print("patch shape", width,height)
    
    #Visualization 
    if (i == 0 or i == 99 or i == 199 or i==299 or i == 399):
        fig, ax = plt.subplots()
        ax.imshow(seq[:,:,i], cmap= 'gray')
        rectan = patches.Rectangle((rect[0], rect[1]), width+1, height+1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectan)
        plt.show()
        
rects = np.asarray(rects)
print(rects.shape)
with open('carseqrects.npy', 'wb') as f:
    np.save(f,rects)
    