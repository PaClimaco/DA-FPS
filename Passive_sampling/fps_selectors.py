import numpy as npaseGraphSelection
import torch
from tqdm import tqdm
import scipy.io
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from numba import njit
from numba import prange
from scipy.spatial import distance
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import os



# -----------
# Pytorch implementations
# -----------

def fps(points_i, initialization, b):
    if b > len(points_i):
        print('Error: number of points to select larger than the number of available points')
        return
    if not torch.is_tensor(points_i):
        points_t = torch.from_numpy(points_i)
    else:
        points_t = points_i  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    points = points_t.to(device)
    centers = initialization
    distances = torch.min(torch.cdist(points, points[centers], p=2), dim=1)[0]
    # Iterate to select additional points until reaching the desired number
    for n in tqdm(range(b - len(initialization))):
        farthest_point = torch.argmax(distances)
        centers.append(farthest_point.item())
        new_distances = torch.min(torch.norm(points - points[farthest_point], dim=1), distances)
        distances = new_distances
    return centers


def da_fps(points_i, initialization, b, d):
    if b > len(points_i):
        print('Error: number of points to select larger than the number of available points')
        return
    # Convert numpy arrays to PyTorch tensors
    points = torch.from_numpy(points_i)
    d = torch.from_numpy(d).to(torch.float32) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = points.to(device)
    centers = [i for i in initialization]
    distances = torch.min(torch.cdist(points, points[centers]), dim=1)[0]
    for _ in tqdm(range(b - len(initialization))):
        neighborhood_to_select = torch.sum((d <= distances.view(-1, 1)), dim=1)
        distances_weights = distances * neighborhood_to_select
        farthest_point = torch.argmax(distances_weights).item() 
        centers.append(farthest_point)
        new_distances = torch.min(torch.norm(points - points[farthest_point], dim=1), distances)  
        distances = new_distances
    return centers

# -----------
# Numpy implementations
# -----------

def fps_np(points, initialization, b):
    if b > len(points):
        print('Error: number of points to select larger than the number of available points')
        return
    centers = [i for i in initialization]
    distances = np.min(cdist(points , points[centers]), axis=1)
    for n in tqdm(range(b - len(initialization))):
        farthest_point = np.argmax(distances)
        centers.append(farthest_point)
        new_distances  = np.minimum(np.linalg.norm(points - points[farthest_point], axis=1), distances)
        distances =  new_distances 
    return centers


def da_fps_np(points, initialization, b, d):
    if b > len(points):
        print('Error: number of points to select larger than the number of available points')
        return
    centers = [i for i in initialization]
    distances = np.min(cdist(points , points[centers]), axis=1)
    for _ in tqdm(range(b - len(initialization))):
        neighborhood_to_select = np.sum((d <= distances[:, np.newaxis]), axis=1)
        distances_weights = distances * neighborhood_to_select
        farthest_point = np.argmax(distances_weights)
        centers.append(farthest_point)
        new_distances  = np.minimum(np.linalg.norm(points - points[farthest_point], axis=1), distances)
        distances =  new_distances 
    return centers

