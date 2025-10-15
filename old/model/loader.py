import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

########################################################################################
class SuperResNpyDataset(Dataset):
    def __init__(self, data_folder, lr_files, hr_files,skip=0):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            lr_files (list of str): List of specific low-resolution .npy filenames.
            hr_files (list of str): List of specific high-resolution .npy filenames.
        """
        assert len(lr_files) == len(hr_files), "Mismatch in number of LR and HR files."

        # Load and concatenate the explicitly provided LR and HR data files
        #self.lr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in lr_files], axis=0)
        #self.hr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in hr_files], axis=0)

        #print(f"Loaded LR data shape: {self.lr_data.shape}")
        #print(f"Loaded HR data shape: {self.hr_data.shape}")

         # Load, skip, and concatenate the LR data
        lr_data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded LR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded LR file '{f}' shape after skip: {data.shape}")
            lr_data_list.append(data)
        self.lr_data = np.concatenate(lr_data_list, axis=0)
        print(np.shape(self.lr_data))
        # exit()

        # Load, skip, and concatenate the HR data
        hr_data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded HR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded HR file '{f}' shape after skip: {data.shape}")
            hr_data_list.append(data)
        self.hr_data = np.concatenate(hr_data_list, axis=0)
        print(np.shape(self.hr_data))
        # exit()


        print("X=>",self.lr_data.shape,self.hr_data.shape)

        # High res to low res as input
        reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,6)
        self.lr_data = reshaped_hr.mean(axis=(2, 4))
        print("X=>",self.lr_data.shape,self.hr_data.shape)
        ##########
        self.lr_data = self.lr_data[:,:,:,4] #/ 1.0e-5
        self.hr_data = self.hr_data[:,:,:,4] #/ 1.0e-5

        self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4), order=1)
        #self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4, 1), order=1)
        #self.hr_data = self.hr_data - self.lr_data

        combined = np.concatenate([self.lr_data, self.hr_data], axis=0)  # Shape: [2N, H, W]
        print("minMax = ",combined.min(),combined.max())
        #print("minMax = ",self.lr_data.min(),self.lr_data.max())
        #print("minMax = ",self.hr_data.min(),self.hr_data.max())
        mean = combined.mean()
        std = combined.std()
        print("Mean :: std = ",mean,std)

        x_normalized = (self.lr_data - mean) / std
        y_normalized = (self.hr_data - mean) / std
        # print("Normalization  ",mean,std,combined.min(),combined.max())
        self.lr_data = x_normalized.copy()
        self.hr_data = y_normalized.copy()
        print("12-----------------------------------------------", np.shape(self.lr_data))

        # print("Range ",self.lr_data.max(),self.lr_data.min(),self.lr_data.max()-self.lr_data.min())
        #-7.5,11.8
        # self.lr_data = (self.lr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min())
        # self.hr_data = (self.hr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min())

        print("==>",self.lr_data.shape,self.hr_data.shape)
        # Ensure the number of samples match
        assert self.lr_data.shape[0] == self.hr_data.shape[0], "Mismatch in number of samples between LR and HR data."
        
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx, :, :]  # Low-res sample
        hr = self.hr_data[idx, :, :]  # High-res sample    
        
        # Standardize manually
       
        # Convert to torch tensors and reshape to [C, H, W]
        lr_tensor = torch.tensor(lr, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]
        hr_tensor = torch.tensor(hr, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]

        #assert not torch.isnan(lr_tensor).any(), "NaN in input data"
        #assert not torch.isnan(hr_tensor).any(), "NaN in input data"
        
        return lr_tensor, hr_tensor
    
## 2D  ##################################################################################
class SuperResNpyDataset2D(Dataset):
    def __init__(self, data_folder, lr_files, hr_files,er,skip=0,iMean=None,iStd=None):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            lr_files (list of str): List of specific low-resolution .npy filenames.
            hr_files (list of str): List of specific high-resolution .npy filenames.
        """
        assert len(lr_files) == len(hr_files), "Mismatch in number of LR and HR files."

        # Load and concatenate the explicitly provided LR and HR data files
        #self.lr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in lr_files], axis=0)
        #self.hr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in hr_files], axis=0)

        #print(f"Loaded LR data shape: {self.lr_data.shape}")
        #print(f"Loaded HR data shape: {self.hr_data.shape}")

         # Load, skip, and concatenate the LR data
        lr_data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded LR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded LR file '{f}' shape after skip: {data.shape}")
            lr_data_list.append(data)
        self.lr_data = np.concatenate(lr_data_list, axis=0)

        # Load, skip, and concatenate the HR data
        hr_data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded HR file '{f}' shape before skip: {data.shape}")
            # data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded HR file '{f}' shape after skip: {data.shape}")
            hr_data_list.append(data)
        self.hr_data = np.concatenate(hr_data_list, axis=0)

        self.er_data=er
        print("X1=>",self.lr_data.shape,self.hr_data.shape , self.er_data.shape)



        # Verify before normalization
        print("==== before normalization 6 chanels ======================================================")
        print("before Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        print("before Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        print("before Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        # exit()
        print("========================================================================================")
        # exit()


        #  HLH     ############################################################################################
        ## High res to low res as input
        # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,6)
        # # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,5)
        # self.lr_data = reshaped_hr.mean(axis=(2, 4))
        # # Initialize an empty array for LR data

        # print("X2=>",self.lr_data.shape,self.hr_data.shape,er.shape)
        ##########  [Reshape needs to be afte rcutting extra chanels !!! ]
        #############################################################################################
        self.lr_data = self.lr_data[:,:,:,2:4]   #keep only channels 2 and 3
        self.hr_data = self.hr_data[:,:,:,2:4]
        self.er_data=er

         # 4 
        reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,2)
        # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,5)
        self.lr_data = reshaped_hr.mean(axis=(2, 4))
        print("X2=>",self.lr_data.shape,self.hr_data.shape,self.er_data.shape)

        # 8
        # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0],self.hr_data.shape[1] // 8, 8, self.hr_data.shape[2] // 8, 8,2)
        # self.lr_data = reshaped_hr.mean(axis=(2, 4))
        # exit()
        


        print("==== before normalization 2 chanels ======================================================")
        print("before Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        print("before Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        print("before Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        # exit()
        print("========================================================================================")


        # self.lr_data = self.lr_data[:,:,:,2:4]/ 1.0e-5
        # self.hr_data = self.hr_data[:,:,:,2:4]/ 1.0e-5
        # print("X22=>",self.lr_data.shape,self.hr_data.shape)
        # exit()
        # print(er)
        # print( self.er_data.mean(),self.er_data.max(),self.er_data.min())
        # exit()
############################################################################################################

        # ######   Plot histograms

        # plt.figure(figsize=(10, 6))

        # plt.hist(self.lr_data.flatten(), bins=50, alpha=0.5, label='LR', color='blue', density=True)
        # plt.hist(self.hr_data.flatten(), bins=50, alpha=0.5, label='HR', color='green', density=True)
        # plt.hist(self.er_data.flatten(), bins=50, alpha=0.5, label='eR', color='red', density=True)

        # plt.title('Histograms of LR, HR, and eR Data')
        # plt.xlabel('Value')
        # plt.ylabel('Density')
        # plt.legend()
        # plt.grid(True)
        # plt.xlim(-4, 4)  # Narrow range based on std ~ 1

        # plt.tight_layout()
        # plt.savefig("His-Train.png",dpi=300 )
        # plt.show()

        # print("==== before normalization 2 chanels ======================================================")
        # print("before Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        # print("before Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        # print("before Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        # exit()
        # print("========================================================================================")
####################################################################################################################






        # self.lr_data = self.lr_data[:,:,:,2:4]/ 1.0e-5
        # self.hr_data = self.hr_data[:,:,:,2:4]/ 1.0e-5

        # self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4, 1), order=1)
        # # self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4), order=1)
        # print("X3=>",self.lr_data.shape,self.hr_data.shape,er.shape)
        # print("12-----------------------------------------------", np.shape(self.lr_data))
        # exit()

        #print("minMax = ",combined.min(),combined.max())
        # print("minMax = ",self.lr_data[:,:,0].min(),self.lr_data[:,:,1].min(),self.lr_data[:,:,2].min(),self.lr_data[:,:,0].max(),self.lr_data[:,:,1].max(),self.lr_data[:,:,2].max())
        # print("minMax = ",self.hr_data[:,:,0].min(),self.hr_data[:,:,1].min(),self.hr_data[:,:,2].min(),self.hr_data[:,:,0].max(),self.hr_data[:,:,1].max(),self.hr_data[:,:,2].max())
        # print("minMax = ",self.er_data[:,:,0].min(),self.er_data[:,:,1].min(),self.er_data[:,:,2].min(),self.er_data[:,:,0].max(),self.er_data[:,:,1].max(),self.er_data[:,:,2].max())

        # combined = np.concatenate([self.lr_data, self.hr_data], axis=0)  # Shape: [2N, H, W]
        # mean = np.mean(combined, axis=(0, 1, 2), keepdims=True)
        # std = np.std(combined, axis=(0, 1, 2), keepdims=True)
        # if not iMean is None:
        #     print("Providing Mean & Std")
        #     mean = iMean
        #     std = iStd
        #####################################################################################
        # combined = np.concatenate([self.lr_data*1.01, self.hr_data*0.99], axis=0)  # Shape: [2N, H, W]
        # print("minMax = ",combined.min(),combined.max())
        # #print("minMax = ",self.lr_data.min(),self.lr_data.max())
        # #print("minMax = ",self.hr_data.min(),self.hr_data.max())
        # mean = combined.mean()
        # std = combined.std()
        # print("Mean :: std = ",mean,std)
        #####################################################################################

        # #########################################################################################
        # print("Before ",self.lr_data.max(),self.lr_data.min(),self.lr_data.max()-self.lr_data.min())
        # x_normalized = (self.lr_data - mean) / std
        # y_normalized = (self.hr_data - mean) / std
        # # print("Normalization  ",mean,std,combined.min(),combined.max())
        # self.lr_data = x_normalized.copy()
        # self.hr_data = y_normalized.copy()
        # print("1-----------------------------------------------")
        ######################################################################################
        # Compute mean and std for each dataset separately

        # mean_lr = self.lr_data.mean()
        # std_lr = self.lr_data.std()
        # mean_hr = self.hr_data.mean()
        # std_hr = self.hr_data.std()
        # mean_er = self.er_data.mean()
        # std_er = self.er_data.std()


        # mean_lr = iMean[0]
        # std_lr = iStd[0]
        # mean_hr = iMean[1]
        # std_hr = iStd[1]
        # mean_er = iMean[2]
        # std_er = iStd[2]

        # print("========================================================================================")
        # print(iMean,iStd)
        # print("Mean (LR) :: Std (LR) = ", mean_lr, std_lr)
        # print("Mean (HR) :: Std (HR) = ", mean_hr, std_hr)
        # print("Mean (eR) :: Std (eR) = ", mean_er, std_er)
        # print("========================================================================================")
        # # exit()

        # Normalize separately
        # self.lr_data = (self.lr_data - mean_lr) / (std_lr + 1e-8)
        # self.hr_data = (self.hr_data - mean_hr) / (std_hr + 1e-8)
        # self.er_data = (self.er_data - mean_er) / (std_er + 1e-8)


        # Verify after normalization
        # print(" Verify after normalization   ========================================================================================")

        # print("After Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        # print("After Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        # print("After Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        # exit()
        # print(" Verify after normalization   ========================================================================================")



        ####################################################################################
        # self.lr_data = torch.tensor(self.lr_data)  # Convert to tensor if it's a numpy array
        # self.hr_data = torch.tensor(self.hr_data)  # Convert to tensor if it's a numpy
        # self.lr_data =  torch.clamp(self.lr_data, min=0.0, max=1.0)
        # self.hr_data = torch.clamp(self.hr_data, min=0.0, max=1.0)


        ## SSIM
        # print("Range ",self.lr_data.max(),self.lr_data.min(),self.lr_data.max()-self.lr_data.min())
        # # #-7.5,11.8
        # # Assuming self.lr_data and self.hr_data are numpy arrays
        # # self.lr_data = torch.tensor(self.lr_data)  # Convert to tensor if it's a numpy array
        # # self.hr_data = torch.tensor(self.hr_data)  # Convert to tensor if it's a numpy array
        # # self.lr_data =  torch.clamp(self.lr_data, min=0.0, max=1.0)
        # # self.hr_data = torch.clamp(self.hr_data, min=0.0, max=1.0)


        ########################################################################################################
        # self.lr_data = (self.lr_data - self.lr_data.min()) / (self.lr_data.max() - self.lr_data.min())
        # self.hr_data = (self.hr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min())
        # self.er_data = (self.er_data - self.er_data.min()) / (self.er_data.max() - self.er_data.min())

        # mean_lr = self.lr_data.mean()
        # std_lr = self.lr_data.std()


        # mean_hr = self.hr_data.mean()
        # std_hr = self.hr_data.std()


        # mean_er = self.er_data.mean()
        # std_er = self.er_data.std()
        # print( self.er_data.mean(),self.er_data.max(),self.er_data.min())
        # exit()

        print("After Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        print("After Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        print("After Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        # exit()

        print("==  Plot His======================================================================================")
        # Plot histograms

        # plt.figure(figsize=(10, 6))

        # plt.hist(self.lr_data.flatten(), bins=50, alpha=0.5, label='LR', color='blue', density=True)
        # plt.hist(self.hr_data.flatten(), bins=50, alpha=0.5, label='HR', color='green', density=True)
        # plt.hist(self.er_data.flatten(), bins=50, alpha=0.5, label='eR', color='red', density=True)

        # plt.title('Histograms of LR, HR, and eR Data')
        # plt.xlabel('Value')
        # plt.ylabel('Density')
        # plt.legend()
        # plt.grid(True)
        # plt.xlim(-4, 40)  # Narrow range based on std ~ 1

        # plt.tight_layout()
        # plt.savefig("HIS_Normalization.png",dpi=300 )
        # plt.show()
        # exit()

        # self.lr_data = (self.lr_data - self.lr_data.min()) / (self.lr_data.max() - self.lr_data.min() + 1e-8)
        # self.hr_data = (self.hr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min() + 1e-8)
        # print((self.lr_data < 0).sum(), (self.hr_data < 0).sum())  # Count negative values
        # print(self.lr_data.min(), self.hr_data.min())  # Print true min values
        # print(f"After normalization: lr_data min={self.lr_data.min()}, max={self.lr_data.max()}")
        # print(f"After normalization: hr_data min={self.hr_data.min()}, max={self.hr_data.max()}")
        # exit()



        ##########################################################################################
        

        print("==>  End",self.lr_data.shape,self.hr_data.shape,self.er_data.shape)
        # Ensure the number of samples match
        assert self.lr_data.shape[0] == self.hr_data.shape[0], "Mismatch in number of samples between LR and HR data."
        
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx, :, :,:]  # Low-res sample
        hr = self.hr_data[idx, :, :,:]  # High-res sample
        er = self.er_data[idx, :, :,:]  # High-res sample    
    
        
        # Standardize manually


       
        # Convert to torch tensors and reshape to [C, H, W]
        lr_tensor = torch.tensor(lr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        hr_tensor = torch.tensor(hr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        er_tensor = torch.tensor(er, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]

        
        lr_tensor = lr_tensor.permute(2, 0, 1)    #[C, H, W]
        hr_tensor = hr_tensor.permute(2, 0, 1)    #[C, H, W]
        er_tensor = er_tensor.permute(2, 0, 1)    #[C, H, W]
        



        #assert not torch.isnan(lr_tensor).any(), "NaN in input data"
        #assert not torch.isnan(hr_tensor).any(), "NaN in input data"
        # exit()
        return lr_tensor, hr_tensor , er_tensor  


########################################################################################
class SuperResNpyDataset2(Dataset):
    def __init__(self, data_folder, lr_files, hr_files,skip=0,iMean=None,iStd=None):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            lr_files (list of str): List of specific low-resolution .npy filenames.
            hr_files (list of str): List of specific high-resolution .npy filenames.
        """
        assert len(lr_files) == len(hr_files), "Mismatch in number of LR and HR files."

        # Load and concatenate the explicitly provided LR and HR data files
        #self.lr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in lr_files], axis=0)
        #self.hr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in hr_files], axis=0)

        #print(f"Loaded LR data shape: {self.lr_data.shape}")
        #print(f"Loaded HR data shape: {self.hr_data.shape}")

         # Load, skip, and concatenate the LR data
        lr_data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded LR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded LR file '{f}' shape after skip: {data.shape}")
            lr_data_list.append(data)
        self.lr_data = np.concatenate(lr_data_list, axis=0)

        # Load, skip, and concatenate the HR data
        hr_data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded HR file '{f}' shape before skip: {data.shape}")
            # data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded HR file '{f}' shape after skip: {data.shape}")
            hr_data_list.append(data)
        self.hr_data = np.concatenate(hr_data_list, axis=0)


        print("X1=>",self.lr_data.shape,self.hr_data.shape)




        print("==== before normalization 6 chanels ======================================================")
        print("before Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        print("before Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        # print("before Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        print("========================================================================================")

        # exit()
####################################################################################################################


        #  HLH     ############################################################################################
        ## High res to low res as input
        # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,6)
        # # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,5)
        # self.lr_data = reshaped_hr.mean(axis=(2, 4))
        # print("X2=>",self.lr_data.shape,self.hr_data.shape)
        ##########
        #############################################################################################
        self.lr_data = self.lr_data[:,:,:,2:4]   #keep only channels 2 and 3
        self.hr_data = self.hr_data[:,:,:,2:4]

        #4
        reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,2)
        # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,5)
        self.lr_data = reshaped_hr.mean(axis=(2, 4))


        # low = low[:,:].reshape(2,int(low.shape[1]/4), 4, int(low.shape[2]/4), 4)
        # low = low.mean(axis=(2, 4))



        
        # 8
        # reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0],self.hr_data.shape[1] // 8, 8, self.hr_data.shape[2] // 8, 8,2)
        # self.lr_data = reshaped_hr.mean(axis=(2, 4))
        # print("X2=>",self.lr_data.shape,self.hr_data.shape)
        


        # self.lr_data = self.lr_data[:,:,:,2:4]/ 1.0e-5
        # self.hr_data = self.hr_data[:,:,:,2:4]/ 1.0e-5
        print("X22=>",self.lr_data.shape,self.hr_data.shape)
        # exit()


    # ############################################################################################################

        # ######   Plot histograms

        # plt.figure(figsize=(10, 6))
        # # Flatten and calculate statistics
        # lr_flat = self.lr_data.flatten()
        # hr_flat = self.hr_data.flatten()

        # plt.hist(self.lr_data.flatten(), bins=50, alpha=0.5, label='LR', color='blue', density=True)
        # plt.hist(self.hr_data.flatten(), bins=50, alpha=0.5, label='HR', color='green', density=True)
        # # plt.hist(self.er_data.flatten(), bins=50, alpha=0.5, label='eR', color='red', density=True)

        # plt.title('Test')
        # plt.xlabel('Value')
        # plt.ylabel('Density')
        # plt.legend()
        # plt.grid(True)
        # plt.xlim(-4, 4)  # Narrow range based on std ~ 1

        
        # # Compute stats
        # lr_stats = f"LR:\nMean = {lr_flat.mean():.4f}\nStd = {lr_flat.std():.4f}\nMin = {lr_flat.min():.4f}\nMax = {lr_flat.max():.4f}"
        # hr_stats = f"HR:\nMean = {hr_flat.mean():.4f}\nStd = {hr_flat.std():.4f}\nMin = {hr_flat.min():.4f}\nMax = {hr_flat.max():.4f}"

        # # Add stats as text to plot
        # plt.text(4.05, 0.35, lr_stats, fontsize=9, color='blue', va='top')
        # plt.text(4.05, -0.15, hr_stats, fontsize=9, color='green', va='top')

        # plt.tight_layout()
        # # plt.savefig("His-Train.png",dpi=300 )
        # plt.savefig("His-Test.png",dpi=300 )
        # plt.show()

        # print("==== before normalization 2 chanels ======================================================")
        # print("before Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        # print("before Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        # # print("before Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
        # print("========================================================================================")
        # print("lr minMax = ",self.lr_data.min(),self.lr_data.max())
        # print("hr minMax = ",self.hr_data.min(),self.hr_data.max())
        # print("##########################################################################")
        # exit()


        # self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4, 1), order=1)
        # self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4), order=1)
        print("----- Loader lr & hr shapes       ------------------------------------------")
        print("X3=>",self.lr_data.shape,self.hr_data.shape)
        print("-----------------------------------------------")
 


        #print("minMax = ",combined.min(),combined.max())
        # print("minMax = ",self.lr_data[:,:,0].min(),self.lr_data[:,:,1].min(),self.lr_data[:,:,2].min(),self.lr_data[:,:,0].max(),self.lr_data[:,:,1].max(),self.lr_data[:,:,2].max())
        # print("minMax = ",self.hr_data[:,:,0].min(),self.hr_data[:,:,1].min(),self.hr_data[:,:,2].min(),self.hr_data[:,:,0].max(),self.hr_data[:,:,1].max(),self.hr_data[:,:,2].max())
        
        # combined = np.concatenate([self.lr_data, self.hr_data], axis=0)  # Shape: [2N, H, W]
        # mean = np.mean(combined, axis=(0, 1, 2), keepdims=True)
        # std = np.std(combined, axis=(0, 1, 2), keepdims=True)
        # if not iMean is None:
        #     print("Providing Mean & Std")
        #     mean = iMean
        #     std = iStd
        #####################################################################################
        # combined = np.concatenate([self.lr_data*1.01, self.hr_data*0.99], axis=0)  # Shape: [2N, H, W]
        # print("minMax = ",combined.min(),combined.max())
        # #print("minMax = ",self.lr_data.min(),self.lr_data.max())
        # #print("minMax = ",self.hr_data.min(),self.hr_data.max())
        # mean = combined.mean()
        # std = combined.std()
        # print("Mean :: std = ",mean,std)
        #####################################################################################

        # self.lr_data = (self.lr_data - mean) / (std + 1e-8)
        # self.hr_data = (self.hr_data - mean) / (std + 1e-8)

        # print("Mean before standardization:", mean)
        # print("Std before standardization:", std)
        # print("Mean after standardization:", np.mean(self.lr_data, axis=(0, 1, 2)))
        # print("Std after standardization:", np.std(self.lr_data, axis=(0, 1, 2)))
        # print("-----------------------------------------------")

        # #########################################################################################
        # print("Before ",self.lr_data.max(),self.lr_data.min(),self.lr_data.max()-self.lr_data.min())
        # x_normalized = (self.lr_data - mean) / std
        # y_normalized = (self.hr_data - mean) / std
        # # print("Normalization  ",mean,std,combined.min(),combined.max())
        # self.lr_data = x_normalized.copy()
        # self.hr_data = y_normalized.copy()
        print("-----------------------------------------------")
        ######################################################################################
        # # Compute mean and std for each dataset separately
        mean_lr = self.lr_data.mean()
        std_lr = self.lr_data.std()

        mean_hr = self.hr_data.mean()
        std_hr = self.hr_data.std()


        print("#########################################################################")
        print("############    BEfore Normalization            ###########################")
        print("#########################################################################")

        print("--- lr & hr :  Mean - std  --------------------------------------------")
        print("##########################################################################")

        print("Mean (LR) :: Std (LR) = ", mean_lr, std_lr)
        print("Mean (HR) :: Std (HR) = ", mean_hr, std_hr)

        print("##########################################################################")
        print("--- lr & hr : Min/Max      --------------------------------------------")
        print("##########################################################################")

        print("lr minMax = ",self.lr_data.min(),self.lr_data.max())
        print("hr minMax = ",self.hr_data.min(),self.hr_data.max(),"////////////////////")
        print("##########################################################################")
        # exit()

        # # Normalize separately  ##############################################################################
        # self.lr_data = (self.lr_data - mean_lr) / (std_lr + 1e-8)
        # self.hr_data = (self.hr_data - mean_hr) / (std_hr + 1e-8)

        # # Verify after normalization
        # print("##########################################################################")
        # print("After Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
        # print("After Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        # print("##########################################################################")

        # exit()

        ####################################################################################
        # self.lr_data = torch.tensor(self.lr_data)  # Convert to tensor if it's a numpy array
        # self.hr_data = torch.tensor(self.hr_data)  # Convert to tensor if it's a numpy
        # self.lr_data =  torch.clamp(self.lr_data, min=0.0, max=1.0)
        # self.hr_data = torch.clamp(self.hr_data, min=0.0, max=1.0)


        ## SSIM
        # print("Range ",self.lr_data.max(),self.lr_data.min(),self.lr_data.max()-self.lr_data.min())
        # #-7.5,11.8
        # Assuming self.lr_data and self.hr_data are numpy arrays
        # self.lr_data = torch.tensor(self.lr_data)  # Convert to tensor if it's a numpy array
        # self.hr_data = torch.tensor(self.hr_data)  # Convert to tensor if it's a numpy array
        # self.lr_data =  torch.clamp(self.lr_data, min=0.0, max=1.0)
        # self.hr_data = torch.clamp(self.hr_data, min=0.0, max=1.0)
        # self.lr_data = (self.lr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min())
        # self.hr_data = (self.hr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min())


        ## Min / Max ############################################################################################

        # self.lr_data = (self.lr_data - self.lr_data.min()) / (self.lr_data.max() - self.lr_data.min() + 1e-8)
        # self.hr_data = (self.hr_data - self.hr_data.min()) / (self.hr_data.max() - self.hr_data.min() + 1e-8)

        #         # Verify after normalization
        print("#########################################################################")
        print("############    After Normalization            ###########################")
        print("#########################################################################")       
        print("After Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
        print("##########################################################################")
        print(f"After normalization: lr_data min={self.lr_data.min()}, max={self.lr_data.max()}")
        print(f"After normalization: hr_data min={self.hr_data.min()}, max={self.hr_data.max()}")
        print("##########################################################################")
        print("Negatives in lr and hr",(self.lr_data < 0).sum(), (self.hr_data < 0).sum())  # Count negative values
        print("lrMin and heMin",self.lr_data.min(), self.hr_data.min())  # Print true min values

        # exit()
        ##########################################################################################

    # ############################################################################################################

    #     ######   Plot histograms

    #     plt.figure(figsize=(10, 6))
    #     # Flatten and calculate statistics
    #     lr_flat = self.lr_data.flatten()
    #     hr_flat = self.hr_data.flatten()

    #     plt.hist(self.lr_data.flatten(), bins=50, alpha=0.5, label='LR', color='blue', density=True)
    #     plt.hist(self.hr_data.flatten(), bins=50, alpha=0.5, label='HR', color='green', density=True)
    #     # plt.hist(self.er_data.flatten(), bins=50, alpha=0.5, label='eR', color='red', density=True)

    #     plt.title('Train-MM')
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlim(-2, 2)  # Narrow range based on std ~ 1

        
    #     # Compute stats
    #     lr_stats = f"LR:\nMean = {lr_flat.mean():.4f}\nStd = {lr_flat.std():.4f}\nMin = {lr_flat.min():.4f}\nMax = {lr_flat.max():.4f}"
    #     hr_stats = f"HR:\nMean = {hr_flat.mean():.4f}\nStd = {hr_flat.std():.4f}\nMin = {hr_flat.min():.4f}\nMax = {hr_flat.max():.4f}"

    #     # Add stats as text to plot
    #     plt.text(4.05, 0.7, lr_stats, fontsize=9, color='blue', va='top')
    #     plt.text(4.05, -0.1, hr_stats, fontsize=9, color='green', va='top')

    #     plt.tight_layout()
    #     # plt.savefig("His-Train.png",dpi=300 )
    #     plt.savefig("His-Train-MM.png",dpi=300 )
    #     plt.show()

    #     print("#########################################################################")
    #     print("############    After Normalization            ###########################")
    #     print("#########################################################################")
    #     print("After Normalization (LR): Mean =", self.lr_data.mean(), "Std =", self.lr_data.std())
    #     print("After Normalization (HR): Mean =", self.hr_data.mean(), "Std =", self.hr_data.std())
    #     # print("before Normalization (eR): Mean =", self.er_data.mean(), "Std =", self.er_data.std())
    #     print("========================================================================================")
    #     print("lr minMax = ",self.lr_data.min(),self.lr_data.max())
    #     print("hr minMax = ",self.hr_data.min(),self.hr_data.max())
    #     print("##########################################################################")
    #     exit()

        

        print("==>",self.lr_data.shape,self.hr_data.shape)
        # Ensure the number of samples match
        assert self.lr_data.shape[0] == self.hr_data.shape[0], "Mismatch in number of samples between LR and HR data."
        
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx, :, :,:]  # Low-res sample
        hr = self.hr_data[idx, :, :,:]  # High-res sample    
        
        # Standardize manually


       
        # Convert to torch tensors and reshape to [C, H, W]
        lr_tensor = torch.tensor(lr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        hr_tensor = torch.tensor(hr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        
        lr_tensor = lr_tensor.permute(2, 0, 1)
        hr_tensor = hr_tensor.permute(2, 0, 1)

        #assert not torch.isnan(lr_tensor).any(), "NaN in input data"
        #assert not torch.isnan(hr_tensor).any(), "NaN in input data"
        
        return lr_tensor, hr_tensor        
########################################################################################
class SuperResNpyDataset3(Dataset):
    def __init__(self, data_folder, lr_files, hr_files,skip=0):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            lr_files (list of str): List of specific low-resolution .npy filenames.
            hr_files (list of str): List of specific high-resolution .npy filenames.
        """
        assert len(lr_files) == len(hr_files), "Mismatch in number of LR and HR files."

        # Load and concatenate the explicitly provided LR and HR data files
        #self.lr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in lr_files], axis=0)
        #self.hr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in hr_files], axis=0)

        #print(f"Loaded LR data shape: {self.lr_data.shape}")
        #print(f"Loaded HR data shape: {self.hr_data.shape}")

         # Load, skip, and concatenate the LR data
        lr_data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded LR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded LR file '{f}' shape after skip: {data.shape}")
            lr_data_list.append(data)
        self.lr_data = np.concatenate(lr_data_list, axis=0)

        # Load, skip, and concatenate the HR data
        hr_data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded HR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded HR file '{f}' shape after skip: {data.shape}")
            hr_data_list.append(data)
        self.hr_data = np.concatenate(hr_data_list, axis=0)


        print("X=>",self.lr_data.shape,self.hr_data.shape)

        # High res to low res as input
        reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,6)
        self.lr_data = reshaped_hr.mean(axis=(2, 4))
        print("X=>",self.lr_data.shape,self.hr_data.shape)
        ##########
        self.lr_data = self.lr_data[:,:,:,2:5]
        self.hr_data = self.hr_data[:,:,:,2:5]

        self.lr_data[:,:,:,2] = self.lr_data[:,:,:,2] / 1.0e-5
        self.hr_data[:,:,:,2] = self.hr_data[:,:,:,2] / 1.0e-5

        self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4, 1), order=1)

        #print("minMax = ",combined.min(),combined.max())
        print("minMax = ",self.lr_data[:,:,0].min(),self.lr_data[:,:,1].min(),self.lr_data[:,:,2].min(),self.lr_data[:,:,0].max(),self.lr_data[:,:,1].max(),self.lr_data[:,:,2].max())
        print("minMax = ",self.hr_data[:,:,0].min(),self.hr_data[:,:,1].min(),self.hr_data[:,:,2].min(),self.hr_data[:,:,0].max(),self.hr_data[:,:,1].max(),self.hr_data[:,:,2].max())
        
        combined = np.concatenate([self.lr_data, self.hr_data], axis=0)  # Shape: [2N, H, W]
        mean = np.mean(combined, axis=(0, 1, 2), keepdims=True)
        std = np.std(combined, axis=(0, 1, 2), keepdims=True)

        self.lr_data = (self.lr_data - mean) / (std + 1e-8)
        self.hr_data = (self.hr_data - mean) / (std + 1e-8)

        print("Mean before standardization:", mean)
        print("Std before standardization:", std)
        print("Mean after standardization:", np.mean(self.lr_data, axis=(0, 1, 2)))
        print("Std after standardization:", np.std(self.lr_data, axis=(0, 1, 2)))
        

        print("==>",self.lr_data.shape,self.hr_data.shape)
        # Ensure the number of samples match
        assert self.lr_data.shape[0] == self.hr_data.shape[0], "Mismatch in number of samples between LR and HR data."
        
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx, :, :,:]  # Low-res sample
        hr = self.hr_data[idx, :, :,:]  # High-res sample    
        
        # Standardize manually
       
        # Convert to torch tensors and reshape to [C, H, W]
        lr_tensor = torch.tensor(lr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        hr_tensor = torch.tensor(hr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        
        lr_tensor = lr_tensor.permute(2, 0, 1)
        hr_tensor = hr_tensor.permute(2, 0, 1)

        #assert not torch.isnan(lr_tensor).any(), "NaN in input data"
        #assert not torch.isnan(hr_tensor).any(), "NaN in input data"
        
        return lr_tensor, hr_tensor
########################################################################################
class SuperResNpyDataset42(Dataset):
    def __init__(self, data_folder, lr_files, hr_files,skip=0,iMean=None,iStd=None):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            lr_files (list of str): List of specific low-resolution .npy filenames.
            hr_files (list of str): List of specific high-resolution .npy filenames.
        """
        assert len(lr_files) == len(hr_files), "Mismatch in number of LR and HR files."

        # Load and concatenate the explicitly provided LR and HR data files
        #self.lr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in lr_files], axis=0)
        #self.hr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in hr_files], axis=0)

        #print(f"Loaded LR data shape: {self.lr_data.shape}")
        #print(f"Loaded HR data shape: {self.hr_data.shape}")

         # Load, skip, and concatenate the LR data
        lr_data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded LR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded LR file '{f}' shape after skip: {data.shape}")
            lr_data_list.append(data)
        self.lr_data = np.concatenate(lr_data_list, axis=0)

        # Load, skip, and concatenate the HR data
        hr_data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded HR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded HR file '{f}' shape after skip: {data.shape}")
            hr_data_list.append(data)
        self.hr_data = np.concatenate(hr_data_list, axis=0)


        print("X=>",self.lr_data.shape,self.hr_data.shape)

        # High res to low res as input
        reshaped_hr = self.hr_data.reshape(self.hr_data.shape[0], int(self.hr_data.shape[1]/4), 4, int(self.hr_data.shape[2]/4), 4,6)
        self.lr_data = reshaped_hr.mean(axis=(2, 4))
        print("X=>",self.lr_data.shape,self.hr_data.shape)
        ##########
        self.lr_data[:,:,:,2:4] = self.lr_data[:,:,:,2:4]
        self.hr_data = self.hr_data[:,:,:,2:4]

        self.lr_data = zoom(self.lr_data, zoom=(1, 4, 4, 1), order=1)
        
        combined = np.concatenate([self.lr_data[:,:,:,2:4], self.hr_data], axis=0)  # Shape: [2N, H, W]
        mean = np.mean(combined, axis=(0, 1, 2), keepdims=True)
        std = np.std(combined, axis=(0, 1, 2), keepdims=True)
        if not iMean is None:
            print("Providing Mean & Std")
            self.lr_data[:,:,:,2] = (self.lr_data[:,:,:,2] - iMean[0]) / (iStd[0] + 1e-8)
            self.hr_data[:,:,:,0] = (self.hr_data[:,:,:,0] - iMean[0]) / (iStd[0] + 1e-8)
            self.lr_data[:,:,:,3] = (self.lr_data[:,:,:,3] - iMean[1]) / (iStd[1] + 1e-8)
            self.hr_data[:,:,:,1] = (self.hr_data[:,:,:,1] - iMean[1]) / (iStd[1] + 1e-8)            
        else:
            self.lr_data[:,:,:,2] = (self.lr_data[:,:,:,2] - mean[0,0,0,0]) / (std[0,0,0,0] + 1e-8)
            self.hr_data[:,:,:,0] = (self.hr_data[:,:,:,0] - mean[0,0,0,0]) / (std[0,0,0,0] + 1e-8)
            self.lr_data[:,:,:,3] = (self.lr_data[:,:,:,3] - mean[0,0,0,1]) / (std[0,0,0,1] + 1e-8)
            self.hr_data[:,:,:,1] = (self.hr_data[:,:,:,1] - mean[0,0,0,1]) / (std[0,0,0,1] + 1e-8)

        print("Mean before standardization:", mean)
        print("Std before standardization:", std)
        #print("Mean after standardization:", np.mean(self.lr_data, axis=(0, 1, 2)))
        #print("Std after standardization:", np.std(self.lr_data, axis=(0, 1, 2)))
        
        self.lr_data = self.lr_data[:,:,:,0:4]

        print("==>",self.lr_data.shape,self.hr_data.shape)
        # Ensure the number of samples match
        assert self.lr_data.shape[0] == self.hr_data.shape[0], "Mismatch in number of samples between LR and HR data."
        
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx, :, :,:]  # Low-res sample
        hr = self.hr_data[idx, :, :,:]  # High-res sample    
        
        # Standardize manually
       
        # Convert to torch tensors and reshape to [C, H, W]
        lr_tensor = torch.tensor(lr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        hr_tensor = torch.tensor(hr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        
        lr_tensor = lr_tensor.permute(2, 0, 1)
        hr_tensor = hr_tensor.permute(2, 0, 1)

        #assert not torch.isnan(lr_tensor).any(), "NaN in input data"
        #assert not torch.isnan(hr_tensor).any(), "NaN in input data"
        
        return lr_tensor, hr_tensor
########################################################################################
class SuperResNpyDatasetLH(Dataset):
    def __init__(self, data_folder, lr_files, hr_files,skip=0,iMean=None,iStd=None):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            lr_files (list of str): List of specific low-resolution .npy filenames.
            hr_files (list of str): List of specific high-resolution .npy filenames.
        """
        assert len(lr_files) == len(hr_files), "Mismatch in number of LR and HR files."

        # Load and concatenate the explicitly provided LR and HR data files
        #self.lr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in lr_files], axis=0)
        #self.hr_data = np.concatenate([np.load(os.path.join(data_folder, f)) for f in hr_files], axis=0)

        #print(f"Loaded LR data shape: {self.lr_data.shape}")
        #print(f"Loaded HR data shape: {self.hr_data.shape}")

         # Load, skip, and concatenate the LR data
        lr_data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded LR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded LR file '{f}' shape after skip: {data.shape}")
            lr_data_list.append(data)
        self.lr_data = np.concatenate(lr_data_list, axis=0)

        # Load, skip, and concatenate the HR data
        hr_data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))
            print(f"Loaded HR file '{f}' shape before skip: {data.shape}")
            data = data[skip:]  # Skip the first 'skip' items
            print(f"Loaded HR file '{f}' shape after skip: {data.shape}")
            hr_data_list.append(data)
        self.hr_data = np.concatenate(hr_data_list, axis=0)

        highPrev = np.load("../data/100/window_Prev_2012.npy")

        self.lr_data = self.lr_data[:4000,:,:,:]
        self.hr_data = self.hr_data[:4000,:,:,:]
        highPrev = highPrev[:4000,:,:,:]
        
        #-- Zoom LRes to HRes dimensions
        zoom_factors = (1, 4, 4, 1)
        self.lr_data = zoom(self.lr_data, zoom_factors, order=1)
        print("LR shape after bicubic = ",self.lr_data.shape)
        #-- add last time step SR
        self.lr_data[:,:,:,0:2] = highPrev[:,:,:,2:4]
        self.lr_data = self.lr_data[:,:,:,0:4]
        self.hr_data = self.hr_data[:,:,:,2:4]

        print("Final shape of LR & HR = ",self.lr_data.shape,self.hr_data.shape)
        
        combinedU = np.concatenate([self.lr_data[:,:,:,0],self.lr_data[:,:,:,2], self.hr_data[:,:,:,0]], axis=0)  # Shape: [2N, H, W]
        combinedV = np.concatenate([self.lr_data[:,:,:,1],self.lr_data[:,:,:,3], self.hr_data[:,:,:,1]], axis=0)  # Shape: [2N, H, W]
        meanU,meanV = np.mean(combinedU),np.mean(combinedV)
        stdU,stdV = np.std(combinedU),np.std(combinedV)
        if not iMean is None:
            print("Providing Mean & Std")
            meanU,meanV = iMean[0],iMean[1]
            stdU,stdV = iStd[0],iStd[1]

        self.lr_data[:,:,:,0] = (self.lr_data[:,:,:,0] - meanU) / (stdU + 1e-8)
        self.lr_data[:,:,:,2] = (self.lr_data[:,:,:,2] - meanU) / (stdU + 1e-8)
        self.hr_data[:,:,:,0] = (self.hr_data[:,:,:,0] - meanU) / (stdU + 1e-8)
        self.lr_data[:,:,:,1] = (self.lr_data[:,:,:,1] - meanV) / (stdV + 1e-8)
        self.lr_data[:,:,:,3] = (self.lr_data[:,:,:,3] - meanV) / (stdV + 1e-8)
        self.hr_data[:,:,:,1] = (self.hr_data[:,:,:,1] - meanV) / (stdV + 1e-8)

        print("Mean before standardization:", meanU,meanV)
        print("Std before standardization:", stdU,stdV)
        

        print("==>",self.lr_data.shape,self.hr_data.shape)
        # Ensure the number of samples match
        assert self.lr_data.shape[0] == self.hr_data.shape[0], "Mismatch in number of samples between LR and HR data."
        
    def __len__(self):
        return self.lr_data.shape[0]
    
    def __getitem__(self, idx):
        lr = self.lr_data[idx, :, :,:]  # Low-res sample
        hr = self.hr_data[idx, :, :,:]  # High-res sample    
        
        # Standardize manually
       
        # Convert to torch tensors and reshape to [C, H, W]
        lr_tensor = torch.tensor(lr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        hr_tensor = torch.tensor(hr, dtype=torch.float32) #.unsqueeze(0)  # Shape: [1, H, W]
        
        lr_tensor = lr_tensor.permute(2, 0, 1)
        hr_tensor = hr_tensor.permute(2, 0, 1)

        #assert not torch.isnan(lr_tensor).any(), "NaN in input data"
        #assert not torch.isnan(hr_tensor).any(), "NaN in input data"
        
        return lr_tensor, hr_tensor   
########################################################################################