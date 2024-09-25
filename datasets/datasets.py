import numpy as np
import pandas as pd
from PIL import Image
import os

import torch
from torchvision import transforms



class FacialKeyPoints(Dataset):
    def __init__(self,csv_path, split='training', device=torch.device('cpu'), model_input_size = 224):
        super(FacialKeyPoints).__init__()
        self.csv_path = csv_path
        self.split = split
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.model_input_size = 224
        self.normalize = transforms.Normalize(
            mean = [0.485,0.456,0.406],
            std = [0.229,0.224,0.225]
        )
        
    def __len__(self):
        return len(self.df)        
    
    def __getitem__(self,index):
        img,original_size = self.get_image(index)
        keypoints = self.get_keypoints(index,original_size=original_size)
        return img,keypoints
    
    def get_image(self,index):
        image_path = os.path.join(os.getcwd(), 'dataset', self.split, self.df.iloc[index,0])
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        #image preprocessing
        img = img.resize((model_input_size,model_input_size))
        img = np.asarray(img,dtype=np.float32) / 255.0
        img = torch.tensor(img).permute(2,0,1).float()
        img = self.normalize(img)
        return img.to(self.device),original_size
        
    def get_keypoints(self,index,original_size):
        kp = self.df.iloc[index, 1:].to_numpy().astype(np.float32)
        kp_x = kp[0::2] / original_size[0]
        kp_y = kp[1::2] / original_size[1]
        kp = np.concatenate([kp_x,kp_y])
        return torch.tensor(kp, dtype=torch.float32).to(self.device)
        
    def load_image(self,index):
        image_path = os.path.join(os.getcwd(), 'dataset', self.split, self.df.iloc[index,0])
        img = Image.open(image_path).convert('RGB')
        img = img.resize((model_input_size,model_input_size))
        return(np.asarray(img) / 255.0)

training_csv_path = '/kaggle/working/dataset/training_frames_keypoints.csv'
test_csv_path='/kaggle/working/dataset/test_frames_keypoints.csv'

training_data = FacialKeyPoints(training_csv_path,device=device)
test_data = FacialKeyPoints(test_csv_path, split='test', device=device)