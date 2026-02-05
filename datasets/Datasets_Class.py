import os
import requests
import scipy.io
import pandas as pd
import numpy as np
import bz2
import tarfile
import pickle
import gzip
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from torch_geometric.datasets import QM9
from sklearn.decomposition import PCA
import zipfile
import tarfile




class Dataset:
    def __init__(self, save_path=None):
        self.save_path = save_path if save_path else os.getcwd()


    def unzip_file(self, file_path, extract_to='.'):
        """
        Unzip a compressed file (zip or tar) to a specified directory.
        
        :param file_path: Path to the compressed file.
        :param extract_to: Directory to extract the files to.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f"Extracted {file_path} to {extract_to}")
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
                print(f"Extracted {file_path} to {extract_to}")
        elif file_path.endswith('.tar.bz2'):
            with tarfile.open(file_path, 'r:bz2') as tar_ref:
                tar_ref.extractall(extract_to)
                print(f"Extracted {file_path} to {extract_to}")
        elif file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r:') as tar_ref:
                tar_ref.extractall(extract_to)
                print(f"Extracted {file_path} to {extract_to}")
        else:
            raise ValueError(f"Unsupported file type: '{file_path}'")


    def download_data(self, url, save_path):
        if os.path.exists(save_path):
            return
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Download successful. Data saved to {save_path}")
        else:
            print(f"Failed to download data. Status code: {response.status_code}")

    def get_unique_indices(self, lst):
        # Get unique elements from the list and their corresponding indices
        unique_elements = list(set(lst))
        indices = [lst.index(element) for element in unique_elements]
        return sorted(indices)

    def calculate_mordred_descriptors(self, smiles):
        # Calculate Mordred descriptors for a given SMILES representation of a molecule
        calc = Calculator(descriptors, ignore_3D=True)
        mol = Chem.MolFromSmiles(smiles)
        desc_values = calc(mol)
        return desc_values
    
    def find_labels(self, t):
        return self.labels_delta[self.names.index(t)]

    def QM7_dataset(self, preprocessing= True, normalize = True):
        qm7_url = 'http://quantum-machine.org/data/qm7.mat'
        self.data_qm7_path = os.path.join(self.save_path, 'data_qm7') 
        os.makedirs(self.data_qm7_path, exist_ok=True)
        self.save_path = os.path.join(self.data_qm7_path, 'qm7.mat')
        self.download_data(qm7_url, self.save_path)
        mat_contents = scipy.io.loadmat(self.save_path)
        variable1 = mat_contents['X']
        variable2 = mat_contents['T']
        if preprocessing:
            upper_triangular_entries_all = []
            # Loop through each matrix in variable1
            for matrix in variable1:
                # Get the indices of the upper triangular part
                indices = np.triu_indices(matrix.shape[0])
                # Extract the upper triangular elements using the indices
                upper_triangular_entries = matrix[indices]
                # Store the upper triangular entries in the list
                upper_triangular_entries_all.append(upper_triangular_entries)
            # Convert the list to a numpy array if needed
            features = np.array(upper_triangular_entries_all)
         
        else:
            features = variable1.reshape((variable1.shape[0], variable1.shape[1] ** 2))
        labels = variable2.reshape(-1)
        return features, labels
    
    
        
        
    def PGrid_dataset(self, target_label='stab', normalize = True):
            grid_url = 'https://archive.ics.uci.edu/static/public/471/electrical+grid+stability+simulated+data.zip'
            self.data_grid_path = os.path.join(self.save_path, 'data_grid') 
            
            self.save_path_zip = os.path.join(self.data_grid_path, 'Pgrid.zip')
            self.save_path_csv = os.path.join(self.data_grid_path, 'Pgrid_data/')
           
            if not os.path.exists(self.data_grid_path):
                os.makedirs(self.data_grid_path, exist_ok=True)
                self.download_data(grid_url, self.save_path_zip)
                self.unzip_file(self.save_path_zip, extract_to= self.save_path_csv)
                greed_df = pd.read_csv(os.path.join(self.save_path_csv,'Data_for_UCI_named.csv'))
            else:
                greed_df = pd.read_csv(os.path.join(self.save_path_csv,'Data_for_UCI_named.csv'))
            features_df = greed_df[['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2',
                'g3', 'g4']]
            feature_vectors = features_df.values
            labels = greed_df['stab'].values  
    
            if normalize == True:
                scaler = MinMaxScaler()
                x_not_norm = np.array(feature_vectors)
                features= scaler.fit_transform(x_not_norm)
            else:
                features= feature_vectors
            return features, labels



    def concrete_dataset(self, normalize = True):
            grid_url = 'https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip'
            self.data_grid_path = os.path.join(self.save_path, 'data_concrete') 
            
            self.save_path_zip = os.path.join(self.data_grid_path, 'concrete.zip')
            self.save_path_csv = os.path.join(self.data_grid_path, 'concrete_data/')
           
            if not os.path.exists(self.data_grid_path):
                os.makedirs(self.data_grid_path, exist_ok=True)
                self.download_data(grid_url, self.save_path_zip)
                self.unzip_file(self.save_path_zip, extract_to= self.save_path_csv)
                df = pd.read_excel(os.path.join(self.save_path_csv,'Concrete_Data.xls'))
            else:
                df = pd.read_excel(os.path.join(self.save_path_csv,'Concrete_Data.xls'))
            
            features_df = df[list(df.keys()[:-1])]
            feature_vectors = features_df.values
            labels_all = df['Concrete compressive strength(MPa, megapascals) '].values 
            feature_unique, unique_idx = np.unique(feature_vectors, axis=0, return_index = True)
            labels = labels_all[list(unique_idx)]
            if normalize == True:
                scaler = MinMaxScaler()
                x_not_norm = np.array(feature_unique)
                features= scaler.fit_transform(x_not_norm)
            else:
                features= feature_unique

            return features, labels
               
        
    def ZINC_dataset(self, target_label='logP', preprocessing=True, normalize = True,  pca= 100):
        # Download and process QM8 dataset
        
        self.data_zinc_path = os.path.join(self.save_path, 'data_zinc_24')
        self.subset_zinc_file_path = os.path.join(self.data_zinc_path, 'subset_ZINC.csv')
        self.mordred_features_file_path = os.path.join(self.data_zinc_path, 'mordred_features_zinc.npy')
        os.makedirs(self.data_zinc_path, exist_ok=True)
        if not os.path.exists(self.subset_zinc_file_path):
            zinc_url = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
            self.download_data(zinc_url, os.path.join(self.data_zinc_path, 'ZINC_250.csv'))
            zinc_df = pd.read_csv(os.path.join(self.data_zinc_path,'ZINC_250.csv'))
            subset_df = zinc_df.sample(n=24000, random_state=42)
            subset_df.reset_index(drop=True, inplace=True)
            
            subset_df.to_csv(self.subset_zinc_file_path, index=False)
        else:
            subset_df = pd.read_csv(self.subset_zinc_file_path)

        if not os.path.exists(self.mordred_features_file_path):
            smiles_set = subset_df['smiles']
            mordred_data = []

            for smiles in tqdm(smiles_set):
                gen_descriptors = self.calculate_mordred_descriptors(smiles)
                filtered_descriptors_nan = np.asarray(list(gen_descriptors.values()), dtype='float64')
                filtered_descriptors = np.nan_to_num(filtered_descriptors_nan, nan=0)
                mordred_data.append([smiles, filtered_descriptors])

            tags_features = np.array(mordred_data, dtype=object)
            np.save(self.mordred_features_file_path, mordred_data)
        else:
            mordred_data = np.load(self.mordred_features_file_path, allow_pickle=True)
            tags_features = np.array(mordred_data, dtype=object)

        features = np.asarray([m[1] for m in tags_features])
        labels = np.stack(subset_df[target_label].to_numpy())
        
        if preprocessing:
            selector = VarianceThreshold(0)
            features_raw = selector.fit_transform(features)
            features = features_raw
            if normalize is not None:
                features_raw = features
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features_raw)
            if pca is not None:
                features_raw = features
                pca = PCA(n_components=pca)
                features = pca.fit_transform(features_raw)

        return features, labels

    def QM8_dataset(self, target_label='E1-PBE0', preprocessing=True, normalize = True,  pca= None):
        # Download and process QM8 dataset
        
        self.data_qm8_path = os.path.join(self.save_path, 'data_qm8')
        self.unique_qm8_file_path = os.path.join(self.data_qm8_path, 'unique_qm8.csv')
        self.mordred_features_file_path = os.path.join(self.data_qm8_path, 'mordred_features_qm8.npy')
        os.makedirs(self.data_qm8_path, exist_ok=True)
        if not os.path.exists(self.unique_qm8_file_path):
            qm8_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv"
            self.download_data(qm8_url, os.path.join(self.data_qm8_path, 'qm8.csv'))
            qm8_df = pd.read_csv(os.path.join(self.data_qm8_path, 'qm8.csv'))
            unique_indices = self.get_unique_indices(qm8_df['smiles'].tolist())
            unique_qm8_df = qm8_df.iloc[unique_indices]
            unique_qm8_df.to_csv(self.unique_qm8_file_path, index=False)
        else:
            unique_qm8_df = pd.read_csv(self.unique_qm8_file_path)

        if not os.path.exists(self.mordred_features_file_path):
            smiles_set = unique_qm8_df['smiles']
            mordred_data = []

            print('computing mordreds descriptors for QM8 dataset')
            for smiles in tqdm(smiles_set):
                gen_descriptors = self.calculate_mordred_descriptors(smiles)
                filtered_descriptors_nan = np.asarray(list(gen_descriptors.values()), dtype='float64')
                filtered_descriptors = np.nan_to_num(filtered_descriptors_nan, nan=0)
                mordred_data.append((smiles, filtered_descriptors))

            tags_features = np.array(mordred_data, dtype=object)
            np.save(self.mordred_features_file_path, mordred_data)
        else:
            mordred_data = np.load(self.mordred_features_file_path, allow_pickle=True)
            tags_features = np.array(mordred_data, dtype=object)

        features = np.asarray([m[1] for m in tags_features])
        labels = np.stack(unique_qm8_df[target_label].to_numpy())

        if preprocessing:
            selector = VarianceThreshold(0)
            features_raw = selector.fit_transform(features)
            features = features_raw
            if normalize is not None:
                features_raw = features
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features_raw)
            if pca is not None:
                features_raw = features
                pca = PCA(n_components=pca, svd_solver='full') #Note that if solver not specified a randomized version PCA is used.
                features = pca.fit_transform(features_raw)

        return features, labels
    
    def QM9_dataset(self, preprocessing=True, normalize = False, pca = None):
        self.data_qm9_path = os.path.join(self.save_path, 'data_qm9')
        self.qm9_file_path = os.path.join(self.data_qm9_path, 'dsgdb9nsd.xyz.tar.bz2')
        self.uncharacterized_qm9_file_path = os.path.join(self.data_qm9_path, 'raw/uncharacterized.txt')
        self.qm9_mordreds_path = os.path.join(self.data_qm9_path, 'qm9_mordred_data.pkl.gz')
        os.makedirs(self.data_qm9_path, exist_ok=True)

        QM9_deepchem = QM9(root=self.data_qm9_path)
        if not os.path.exists(self.qm9_file_path):
            qm9_url = 'https://figshare.com/ndownloader/files/3195389'
            self.download_data(qm9_url, self.qm9_file_path)

        # Download uncharacterized molecules file
        if not os.path.exists(self.uncharacterized_qm9_file_path):
            uncharacterized_qm9_url = 'https://figshare.com/ndownloader/files/3195404'
            self.download_data(uncharacterized_qm9_url, self.uncharacterized_qm9_file_path)

        # Process and save Mordred descriptors for QM9 dataset
        if not os.path.exists(self.qm9_mordreds_path):
            # Open the .tar.bz2 file, decompress it, and read contents
            with open(self.qm9_file_path, "rb") as tarbz2_file:
                with bz2.BZ2File(tarbz2_file) as bz2_file:
                    with tarfile.TarFile(fileobj=bz2_file, mode="r") as tar:
                        smiles, tags = [], []

                        for member in tqdm(tar.getmembers()):
                            file_contents = tar.extractfile(member).read().decode("utf-8")
                            lines = file_contents.splitlines()
                            n_atoms = int(lines[0])  # number of atoms
                            smile = lines[n_atoms + 3].split()[1]  # smiles string
                            smiles.append(smile)
                            tags.append(lines[1].split()[0] + "_" + lines[1].split()[1])

            # Exclude uncharacterized molecules
            with open(self.uncharacterized_qm9_file_path, 'r') as f:
                skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

            smiles_consistent = [smiles[i] for i in range(len(smiles)) if i not in skip]
            tags_consistent = [tags[i] for i in range(len(tags)) if i not in skip]

            # Define a function to calculate Mordred descriptors for a given SMILES string on qm8
            def calculate_mordred_descriptors_qm9(smiles):
                calc = Calculator(descriptors, ignore_3D=False)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None, False
                else:
                    desc_values = calc(mol)
                    return desc_values, True

            # Calculate Mordred descriptors and save data
            mordred_data = []
            names_mordreds_not_computed = []
            print('Computing mordreds descriptors for QM9...')
            for i in tqdm(range(len(smiles_consistent))):
                gen_descriptors, flag = calculate_mordred_descriptors_qm9(smiles_consistent[i])

                if flag:
                    filtered_descriptors_nan = np.asarray(list(gen_descriptors.values()), dtype='float64')
                    filtered_descriptors = np.nan_to_num(filtered_descriptors_nan, nan=0)
                    mordred_data.append((tags_consistent[i], filtered_descriptors, smiles_consistent[i]))
                else:
                    names_mordreds_not_computed.append(tags_consistent[i])

            # Remove duplicates and save the unique Mordred descriptors
            unique_indices = self.get_unique_indices([m[2] for m in mordred_data])
            unique_mordreds = [(mordred_data[i][0], mordred_data[i][1], mordred_data[i][2]) for i in unique_indices]

            with gzip.open(self.qm9_mordreds_path, "wb") as file:
                pickle.dump(unique_mordreds, file)

        # Load compressed data using gzip and pickle
        with gzip.open(self.qm9_mordreds_path, "rb") as file:
            tags_mordred = pickle.load(file)

        self.tags = [t[0] for t in tags_mordred]
        features = np.asarray([m[1] for m in tags_mordred])

        # Preprocess data
        if preprocessing:
            selector = VarianceThreshold(0)
            features_raw = selector.fit_transform(features)
            features = features_raw
            if normalize is not None:
                features_raw = features
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features_raw)
            if pca is not None:
                features_raw = features
                pca = PCA(n_components=pca, svd_solver='full')
                features = pca.fit_transform(features_raw)
        
        # Extract labels using multiprocessing
        self.names = [q.name for q in QM9_deepchem]
        self.labels_delta = np.array([q.y[0, 4] for q in QM9_deepchem])  # delta

        with Pool() as mp_pool:
            labels = np.array(list(mp_pool.map(self.find_labels, self.tags)))

        return features, labels


            
            
            
                    


                
                
            
        
    
