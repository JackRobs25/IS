import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import slic

# create adjacency matrix for SLIC superpixelation segments
def create_adj_matrix(segments_slic, kernel_size, n_sp, width, length):
    G = np.zeros((n_sp, n_sp)) # represents neighbouring relationship between superpixels
    for seg in np.unique(segments_slic):
        mask = segments_slic == seg
        
        xy = np.where(mask)
        max_x, min_x = np.max(xy[0]), np.min(xy[0])
        max_y, min_y = np.max(xy[1]), np.min(xy[1])
        min_x = max(0, min_x - kernel_size) 
        min_y = max(0, min_y - kernel_size)
        max_x = min(width, max_x + kernel_size)
        max_y = max(length, max_y + kernel_size)
        region_of_interest = mask[min_x:max_x, min_y:max_y]
        
        dilated = ndimage.binary_dilation(region_of_interest)
        diff = dilated - region_of_interest.astype(int)
        neig = np.unique(segments_slic[min_x:max_x, min_y:max_y][diff != 0])
        G[seg, neig] = 1
    return G

# greedy algorithm to generate dictionary mapping segments to colors 
def generate_colored_G(G):
    # count degree of all node.
    degree =[]
    for i in range(len(G)):
        degree.append(sum(G[i]))

    # instantiate the possible color
    colorDict = {}
    for i in range(len(G)):
        colorDict[i]=[0,1,2,3,4,5]


    # sort the node depends on the degree
    sortedNode=[]
    indeks = []

    # use selection sort
    for i in range(len(degree)):
        _max = 0
        j = 0
        for j in range(len(degree)):
            if j not in indeks:
                if degree[j] > _max:
                    _max = degree[j]
                    idx = j
        indeks.append(idx)
        sortedNode.append(idx)

    # The main process
    solution={}
    for n in sortedNode: # starting from the node of highest degree
        setTheColor = colorDict[n] # setTheColor = list of available colors
        solution[n] = setTheColor[0] # assign the color for current node to be the first available color
        adjacentNode = G[n] # G[t_[n]] is the corresponding row of the adj matrix for node n
        for j in range(len(adjacentNode)): # for each node in the graph
            if adjacentNode[j]==1 and (setTheColor[0] in colorDict[j]): # if its a neighbour and it currently has the color just used by node n
                colorDict[j].remove(setTheColor[0]) # remove that color from the list so that we don't end up having some colored neighbours
    return solution

# encode superpixelation of input img
def sp_encode(img):
    width, length, _ = np.shape(img)
    segments_slic = slic(img, n_segments=300, compactness=10, sigma=1,
                        start_label=1) - 1 

    # # # Find the adjacency matrix for that superpixelation
    n_sp = len(np.unique(segments_slic))
    kernel_size = 3

    G = create_adj_matrix(segments_slic, kernel_size, n_sp, width, length)
    
    # # # Use that adjacency matrix to generate a "colored graph" - CSP problem
    solution = generate_colored_G(G)

    # Vectorised code
    segments_slic_1d = segments_slic.flatten()
    color_pix = np.array([solution[segment] for segment in segments_slic_1d])
    color_pix = color_pix.reshape(segments_slic.shape)

    out = F.one_hot(torch.tensor(color_pix)).permute(2, 0, 1)

    return out
