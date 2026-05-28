import torch

class Dataset_VRP20(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs):
        'Initialization'
        #, targets,target_loads
        #self.targets=targets
        #self.target_loads=target_loads
        self.list_IDs = list_IDs
        
    def __len__(self):
        'Denotes total nr of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        ID = self.list_IDs[index]
        
        #if isinstance(ID, str):
        #    x_id = int(ID)
        #else:
        #    x_id = [int(p) for p in ID]
        
        #print(ID)
        # Load data and get label
        #sample=torch.load('VRP20/data_together/+ ID + '.pt')
        #X,y,y_load=sample[0],sample[1],sample[2]
        X = torch.load('VRP20/data_2d_all/' + ID + '.pt')
        y = torch.load('VRP20/data_targ_all/' + ID + '.pt')
        y_load = torch.load('VRP20/data_targloads_all/' + ID + '.pt')
        
        return X,y,y_load
        

class Dataset_VRP50_(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs,X_dat,Y_dat,YLoad_dat):
        'Initialization'
        self.list_IDs = list_IDs
        self.X_dat = X_dat
        self.Y_dat = Y_dat
        self.YLoad_dat = YLoad_dat
        
    def __len__(self):
        'Denotes total nr of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        ID = self.list_IDs[index]
        X = self.X_dat[ID]
        y = self.Y_dat[ID]
        y_load = self.YLoad_dat[ID]
        
        return X,y,y_load