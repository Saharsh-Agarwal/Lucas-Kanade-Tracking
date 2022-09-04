import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=0, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

rects_not_corr = np.load("../code/girlseqrects.npy")
corr_rect = []
corr_rect.append(rect)
pstar = None
p = None

start_frame = seq[:,:,0]

for i in range(seq.shape[2]-1): 
    next_frame = seq[:,:,i+1]
    p = LucasKanade(start_frame, next_frame, rect, threshold, num_iters)
    
    frame_rect = np.asarray([rect[0]+p[0], rect[1]+p[1],rect[2]+p[0], rect[3]+p[1]])
    corr_rect.append(frame_rect)
    
    width = frame_rect[2]-frame_rect[0]
    height = frame_rect[3]-frame_rect[1]
    ### print("patch shape", width,height)
    
    temp_img = next_frame
    
    #Visualization 
    if i in [0,19,39,59,79]:
        print(i)
        fig, ax = plt.subplots()
        ax.imshow(temp_img, cmap= 'gray')
        rectan = patches.Rectangle((frame_rect[0], frame_rect[1]), width+1, height+1, linewidth=2, edgecolor='r', facecolor='none')
        a,b,c,d = rects_not_corr[i+1]
        rectancorr = patches.Rectangle((a, b), c-a+1, d-b+1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rectan)
        ax.add_patch(rectancorr)
        plt.show()
    
    if pstar is None or np.sum((pstar-p)**2)**0.5 < template_threshold:
        print("############################################################")
        start_frame = next_frame
        rect = frame_rect
        pstar = np.copy(p)
        
corr_rect = np.asarray(corr_rect)
with open('girlseqrects-wcrt.npy', 'wb') as f:
    np.save(f,corr_rect)