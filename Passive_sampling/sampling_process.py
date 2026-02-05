import numpy as np
import h5py
from tqdm import tqdm
import random
from apricot import FacilityLocationSelection
from sklearn_extra.cluster import KMedoids
from scipy.spatial import cKDTree
import sys
new_paths = ['../datasets/', './fps_selectors.py']
for path in new_paths:
    sys.path.insert(0, path)
import warnings
warnings.filterwarnings("ignore")
from fps_selectors import fps,  da_fps, fps_np,  da_fps_np
np.random.seed(123)


def sampler(x, save_path, strategies= None, trainig_set_sizes= None, initial_conditions = None, NN = 100, mu = 3 ):
    """
    Sampler function to generate training and test splits using various sampling strategies. The splits are saved in a h5py file. 

    Parameters:
        x (array_like): Input data array (n_samples, n_features).
        save_path (str): Path to save the generated h5py file.
        strategies (list, optional): List of sampling strategies to apply. Default is None. possible strategies are: [ 'FPS', 'RDM', 'FacilityLocation', 'k-medoids++', 'FPS-RDM', 'FPS-FacLoc', 'FPS-k-medoids++', 'DA-FPS', 'DA-FPS'_torch]
        training_set_sizes (list, optional): List of training set sizes. Default is 5%, 10%, 15% and 20% of the available datapoints.
        initial_conditions (list, optional): List of initial conditions. Default random elements.
        NN (int, optional): Number of nearest neighbors to use in DA-FPS. Default is 100.
        mu (int, optional): Parameter value determining the amount of initial samples to select with constant weight when using DA-FPS. Default is 3%.

    Output:
        h5py file containing train/test splits created according to the specified strategies.
    """
    
    if trainig_set_sizes is None:
        trainig_set_sizes =  [int((len(x)/100)*c) for c in [ 5, 10, 15, 20]]
        
    if initial_conditions is None:
        def select_random_indices(n, m, seed=None):
                random.seed(seed)
                num_list = list(range(n))
                random_indices = random.sample(num_list, m)
                return random_indices
        seed = 123  # Set seed for reproducibility
        initial_conditions = select_random_indices(len(x), 5, seed)
    f = h5py.File(save_path, "w")
    nearests = [NN]

    if 'DA-FPS' in strategies:
        print('DA-FPS numpy implementation')
        for nn in nearests:
            tree = cKDTree(x)
            d, i = tree.query(x, nn, workers = -1)
            grp = f.create_group(f'DA-FPS')
            for count, init in  enumerate(initial_conditions,1):
                    subgrup_train = grp.create_group(f'train_Initialize_{count}')
                    subgrup_test = grp.create_group(f'test_Initialize_{count}')
                    idx = fps_np(x, [init], int((len(x)/100)*mu))
                    idx_fps= da_fps_np(x, idx, max(trainig_set_sizes), d)
                    idx_test = list(np.arange(x.shape[0]))
                    for n in trainig_set_sizes:
                                idx_train =  idx_fps[:n]
                                idx_test_selected=  list(set(idx_test).difference(set(idx_train)))  
                                subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                                subgrup_test.create_dataset(f'selected_{n}', data = idx_test_selected)
                                
    if 'DA-FPS_torch' in strategies:
        print('DA-FPS torch implementation')
        for nn in nearests:
            tree = cKDTree(x)
            d, i = tree.query(x, nn, workers = -1)
            grp = f.create_group(f'DA-FPS_torch')
            for count, init in  enumerate(initial_conditions,1):
                    subgrup_train = grp.create_group(f'train_Initialize_{count}')
                    subgrup_test = grp.create_group(f'test_Initialize_{count}')
                    idx = fps(x, [init], int((len(x)/100)*mu))
                    idx_fps= da_fps(x, idx, max(trainig_set_sizes), d)
                    idx_test = list(np.arange(x.shape[0]))
                    for n in trainig_set_sizes:
                                idx_train =  idx_fps[:n]
                                idx_test_selected=  list(set(idx_test).difference(set(idx_train)))  
                                subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                                subgrup_test.create_dataset(f'selected_{n}', data = idx_test_selected)
        
    if 'FPS' in strategies:
        print('Running FPS...')
        grp = f.create_group(f'FPS')
        for count, init in  enumerate(initial_conditions,1):
            subgrup_train = grp.create_group(f'train_Initialize_{count}')
            subgrup_test = grp.create_group(f'test_Initialize_{count}')
            idx_fps = fps_np(x, [init], max(trainig_set_sizes))
            idx_test = list(np.arange(x.shape[0]))
            for n in trainig_set_sizes:
                idx_train = idx_fps[:n]
                idx_test=  list(set(idx_test).difference(idx_train)) 
                subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                subgrup_test.create_dataset(f'selected_{n}', data = idx_test)
        

    if 'RDM' in strategies:
        print('Running RDM...')
        grp = f.create_group(f'RDM')
        for i in range(len(initial_conditions)):
                random.seed(i)
                j = i+1
                subgrup_train = grp.create_group(f'train_Initialize_{j}')
                subgrup_test = grp.create_group(f'test_Initialize_{j}')
                last = 0
                idx_train = []
                idx_test = list(np.arange(x.shape[0]))
                for n in trainig_set_sizes:
                    n_ = n - last
                    idx_train += random.sample(idx_test,n_)
                    idx_test = list(set(idx_test).difference(idx_train))
                    last = n 
                    subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                    subgrup_test.create_dataset(f'selected_{n}', data = idx_test)

    if 'k-medoids++' in strategies:
        print('Running k-medoids++...')
        grp = f.create_group(f'k-medoids++')
        for count, init in  enumerate(initial_conditions,1):
            subgrup_train = grp.create_group(f'train_Initialize_{count}')
            subgrup_test = grp.create_group(f'test_Initialize_{count}')
            idx_test = list(np.arange(x.shape[0]))
            for n in tqdm(trainig_set_sizes):
                kmedoids = KMedoids(n_clusters=n, init = 'k-medoids++', metric = 'euclidean', random_state=count).fit(x)
                select_idx =list(kmedoids.medoid_indices_)
                idx_train = select_idx
                idx_test_selected=  list(set(idx_test).difference(idx_train)) 
                subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                subgrup_test.create_dataset(f'selected_{n}', data = idx_test_selected)
                

    if 'FacilityLocation' in strategies:
        print('Running FacilityLocation...')
        grp = f.create_group(f'FacilityLocation')
        for count, init in  enumerate(initial_conditions,1):
            subgrup_train = grp.create_group(f'train_Initialize_{count}')
            subgrup_test = grp.create_group(f'test_Initialize_{count}')   
            selector = FacilityLocationSelection(max(trainig_set_sizes) , 'euclidean',initial_subset= [init], verbose=True, n_jobs= -1)
            selector.fit(x)
            idx_slctd = selector.ranking
            idx_test = list(np.arange(x.shape[0]))
            for n in trainig_set_sizes:
                    idx_train =  idx_slctd[:n]
                    idx_test=  list(set(idx_test).difference(idx_train))  
                    subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                    subgrup_test.create_dataset(f'selected_{n}', data = idx_test)
                
    if 'FPS-k-medoids++' in strategies:
        print('Running FPS-k-medoids++...')
        grp = f.create_group(f'FPS-k-medoids++')
        for count, init in  enumerate(initial_conditions,1):
            subgrup_train = grp.create_group(f'train_Initialize_{count}')
            subgrup_test = grp.create_group(f'test_Initialize_{count}')
            idx = fps_np(x, [init], int((len(x)/100)*mu))
            idx_test = list(np.arange(x.shape[0]))
            idx_test_selected=  list(set(idx_test).difference(set(idx)))  
            for n in tqdm(trainig_set_sizes):
                kmedoids = KMedoids(n_clusters=n - len(idx), init = 'k-medoids++', metric = 'euclidean', random_state=count).fit(x[idx_test_selected])
                select_idx =kmedoids.medoid_indices_
                idx_train =  list(idx) + list(np.asarray(idx_test_selected)[np.asarray(select_idx)])
                idx_test_selected=  list(set(idx_test).difference(idx_train)) 
                subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                subgrup_test.create_dataset(f'selected_{n}', data = idx_test_selected)  
    
    if 'FPS-FacLoc' in strategies:
        print('Running FPS-FacLoc...')
        grp = f.create_group(f'FPS-FacLoc')
        for count, init in  enumerate(initial_conditions,1):
            subgrup_train = grp.create_group(f'train_Initialize_{count}')
            subgrup_test = grp.create_group(f'test_Initialize_{count}') 
            idx = fps_np(x, [init], int((len(x)/100)*mu))  
            selector = FacilityLocationSelection(max(trainig_set_sizes) - len(idx) , 'euclidean',initial_subset= list(idx), verbose=True, n_jobs= -1)
            selector.fit(x)
            idx_slctd = list(idx) + list(selector.ranking)
            idx_test = list(np.arange(x.shape[0]))
            for n in trainig_set_sizes:
                    idx_train =  idx_slctd[:n]
                    idx_test=  list(set(idx_test).difference(idx_train))  
                    subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                    subgrup_test.create_dataset(f'selected_{n}', data = idx_test)    

    if 'FPS-RDM' in strategies:
        print('Running FPS-RDM...')
        grp = f.create_group(f'FPS-RDM')
        for i in range(len(initial_conditions)):
                random.seed(i)
                j = i+1
                subgrup_train = grp.create_group(f'train_Initialize_{j}')
                subgrup_test = grp.create_group(f'test_Initialize_{j}')
                idx_fps= fps_np(x, [initial_conditions[i]], int((len(x)/100)*mu))
                idx_test = list(set(list(np.arange(x.shape[0]))).difference(idx_fps))
                idx_selected = idx_fps
                idx_selected += random.sample(idx_test,max(trainig_set_sizes)- len(idx_fps))
                idx_test = list(np.arange(x.shape[0]))
                for n in trainig_set_sizes:
                        idx_train = idx_selected[:n]
                        idx_test = list(set(idx_test).difference(idx_train))
                        subgrup_train.create_dataset(f'selected_{n}', data = idx_train) 
                        subgrup_test.create_dataset(f'selected_{n}', data = idx_test)
            
    f.close()