import os
from os import listdir
from os.path import isfile, join
import io
import json

import pandas as pd
import numpy as np
from tqdm import tqdm


class SmnHelper:

    def __init__(self):
        self.data_dir = "data/"
        self.meta_dir = "meta/"
        self.parameters_filename = "parameters_list.json"
        self.stations_filename = "stations_list.json"

    def set_paths(self, base_path):
        # Define paths
        self.base_path = base_path
        self.data_path = os.path.join(base_path, self.data_dir)
        self.parameters_path = os.path.join(base_path, self.meta_dir,
                                            self.parameters_filename)
        self.stations_path = os.path.join(base_path, self.meta_dir,
                                          self.stations_filename)

        # Load JSON files
        self.stations = self.__load_json(self.stations_path)
        self.parameters = self.__load_json(self.parameters_path)
    
    def __load_json(self,json_path):
        """
        Helper function to load a json file

        Args:
            json_path (str): The path to the json file
        
        Returns:
            dict : The loaded JSON
            
        """
        with io.open(json_path, mode='r') as json_file:
            data = json.load(json_file)
        return data
    
    
    def load_csv(self, debug=False):
        """
        Helper function to load all CSVs from the SMN dataset in memory.
        This function populates the self.data variable.
        
        Args:
            debug (bool): Optionnal arg to debug (only loads two years)
        
        """
        
        if debug:
            years = ['1999','2000']
        else:
            years = os.listdir(self.data_path)
        
        # Get every file path into one array
        all_files = []
        for y in years:
            year_path = os.path.join(self.data_path, y)
            files = os.listdir(year_path)
            all_files += [os.path.join(year_path, file) for file in files]
            
        # Create a Dict of DataFrames
        data = {}
        for file in tqdm(all_files, desc="Load files "):
            filename = os.path.basename(os.path.normpath(file))
            wmo = filename.split('_')[0]
            csv = pd.read_csv(file, index_col=0)     
            # Populate the Dict
            if wmo in data.keys():
                # Columns get added if different 
                # See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
                data[wmo] = data[wmo].append(csv)
            else:
                data[wmo] = csv
        
        # Make indexes ready for usage
        for k in tqdm(data, desc="Clean indexes "):
            data[k].index = pd.to_datetime(data[k].index)
            data[k] = data[k].sort_index()
        
        self.data = data

        
    def reselect(self, data, columns):
        """
        Select a subset of parameters from the whole dataset
        and apply this to every stations.
        
        Args:
            columns (list): The list of parameters to select
        
        Returns:
            dict: The dict of DataFrames with only the selected parameters
        
        """
        new_data = {}
        for k in data:
            new_data[k] = data[k][columns] 
        return new_data
    
    
    def augment(self, data, augmentations, new_names=False):
        """
        Augment the main dataset by adding informations chosen 
        from the stations. 
        
        Args:
            data (dict<DataFrame>): The data to augment
            augmentations (List<str>): The List of str refering to smn.stations dict
            new_names (List<str>): optionnal param if you want to rename some columns
        
        """
        for k in data:
            for a in augmentations:
                data[k][a] = self.stations[k][a]
                if new_names:
                    renaming = dict(zip(augmentations, new_names))
                    data[k].rename(columns=renaming, inplace=True)    
        return data
    
    
    def unify(self, data):
        """
        Take a dict of DataFrames and put everything 
        in a single DataFrames after having renamed all
        columns.
        
        Returns:
            res: DataFrame of all values from all stations
        
        """
        all_dfs = []
        # Rename columns
        for k in data:
            data[k].rename(columns=lambda x: k+"_"+x, inplace=True)
            all_dfs.append(data[k])
        # Merge all stations DataFrames into one
        res = pd.concat(all_dfs, axis=1)
        return res
