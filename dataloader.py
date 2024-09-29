# Dataloader for custom PhraseCut Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class PhraseCutDataset_(Dataset):
    
    def __init__(self, split = None):
        super(PhraseCutDataset_self, self).__init__()
        
        if split == 'mini':
            yx = pd.read_csv("data_train.csv")
            
            self.size = 100
            
            self.phrases = yx['phrase'][0:100]
            self.tasks = yx['task_id'][0:100]
            self.input_images = []
            self.outputs = []
            for i in range(100):
                temp_ip = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/images_train/%s.jpg" % yx.iloc[i]['image_id'])
                temp_op = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/output_train/%s.jpg" % yx.iloc[i]['task_id'])
                temp_ip = temp_ip.resize((224, 224))
                temp_op = temp_op.resize((224, 224))
                self.input_images.append(np.array(temp_ip, dtype=np.float32))
                self.outputs.append(np.array(temp_op, dtype = np.float32)/255)
                
        elif split == 'train_subset_pt1':
            yx = pd.read_csv("data_train.csv")
            
            self.size = 500
            
            self.phrases = yx['phrase'][0:500]
            self.tasks = yx['task_id'][0:500]
            self.input_images = []
            self.outputs = []
            for i in range(500):
                temp_ip = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/images_train/%s.jpg" % yx.iloc[i]['image_id'])
                temp_op = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/output_train/%s.jpg" % yx.iloc[i]['task_id'])
                temp_ip = temp_ip.resize((224, 224))
                temp_op = temp_op.resize((224, 224))
                self.input_images.append(np.array(temp_ip, dtype=np.float32))
                self.outputs.append(np.array(temp_op, dtype = np.float32)/255)
                
        elif split == 'val_subset':
            yx = pd.read_csv("data_val.csv")
            
            self.size = 100
            
            self.phrases = yx['phrase'][0:100]
            self.tasks = yx['task_id'][0:100]
            self.input_images = []
            self.outputs = []
            for i in range(100):
                temp_ip = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/images_val/%s.jpg" % yx.iloc[i]['image_id'])
                temp_op = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/output_val/%s.jpg" % yx.iloc[i]['task_id'])
                temp_ip = temp_ip.resize((224, 224))
                temp_op = temp_op.resize((224, 224))
                self.input_images.append(np.array(temp_ip, dtype=np.float32))
                self.outputs.append(np.array(temp_op, dtype=np.float32)/255)
                
        elif split == 'test_model':
            yx = pd.read_csv("data_train.csv")
            
            self.size = 2000
            
            self.phrases = yx['phrase'][0:2000]
            self.tasks = yx['task_id'][0:2000]
            self.input_images = []
            self.outputs = []
            for i in range(2000):
                temp_ip = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/images_train/%s.jpg" % yx.iloc[i]['image_id'])
                temp_op = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/output_train/%s.jpg" % yx.iloc[i]['task_id'])
                temp_ip = temp_ip.resize((224, 224))
                temp_op = temp_op.resize((224, 224))
                self.input_images.append(np.array(temp_ip, dtype=np.float32))
                self.outputs.append(np.array(temp_op, dtype=np.float32)/255)
                 
        elif split == 'train_subset_pt2':
            yx = pd.read_csv("data_train.csv")
            
            self.size = 15000
            
            self.phrases = yx['phrase'][0:15000]
            self.tasks = yx['task_id'][0:15000]
            self.input_images = []
            self.outputs = []
            for i in range(15000):
                temp_ip = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/images_train/%s.jpg" % yx.iloc[i]['image_id'])
                temp_op = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/output_train/%s.jpg" % yx.iloc[i]['task_id'])
                temp_ip = temp_ip.resize((224, 224))
                temp_op = temp_op.resize((224, 224))
                self.input_images.append(np.array(temp_ip, dtype=np.float32))
                self.outputs.append(np.array(temp_op, dtype=np.float32)/255)
            
        else:
            yx = pd.read_csv(f"data_{split}.csv")
            
            self.size = len(yx)
            
            self.phrases = yx['phrase'][0:]
            self.tasks = yx['task_id'][0:]
            self.input_images = []
            self.outputs = []
            for i in range(len(yx)):
                temp_ip = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/images_%s/%s.jpg" % (split, yx.iloc[i]['image_id']))
                temp_op = Image.open("PhraseCutDataset/data/VGPhraseCut_v0/output_%s/%s.jpg" % (split, yx.iloc[i]['task_id']))
                temp_ip = temp_ip.resize((224, 224))
                temp_op = temp_op.resize((224, 224))
                self.input_images.append(np.array(temp_ip, dtype=np.float32))
                self.outputs.append(np.array(temp_op, dtype=np.float32)/255)
        
    
    def __len__(self):
        
        return self.size
    
    def __getitem__(self, idx):
        
        return self.phrases[idx], self.input_images[idx], self.outputs[idx], self.tasks[idx]
