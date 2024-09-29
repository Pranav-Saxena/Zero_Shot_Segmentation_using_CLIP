from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from PhraseCutDataset.utils.file_paths import img_fpath, img_fpath2
from PhraseCutDataset.utils.refvg_loader import RefVGLoader
plt.switch_backend('agg')
import pandas as pd

def generator(split = None, pred_mask = 'Greys'):
    
    if split is not None:
        
        import os
        
        refvg = RefVGLoader(split=split)
        df_temp = pd.read_csv(f'data_{split}.csv')
        
        list_ids = list(set(df_temp['image_id'][1:]))
        
        for i in range(len(list_ids)):
            
            img_id = list_ids[i]
            
            img_ref_data = refvg.get_img_ref_data(img_id)
            
            Polygons = img_ref_data['gt_Polygons']
            
            task_ids = img_ref_data['task_ids']
            
            j = 0
            
            for task_i, task_id in enumerate(task_ids):
                
                img = Image.open(os.path.join(img_fpath[split],
                                              '%d.jpg' % img_id))
                
                img_n = np.array(img)

                h, w = img_n.shape[1], img_n.shape[0]
                
                polygon_list = Polygons[task_i]
                
                mask = Image.new('L', (h, w), 0)
                
                for polygon_coords in polygon_list:
                    
                    p_m = []
                    
                    for p in polygon_coords:
                        
                        for x, y in p:
                        
                            p_m.append((int(x), int(y)))
    
                        ImageDraw.Draw(mask, mode = 'L').polygon(p_m, fill = 255, outline = 1)
                
                mask.save(f"PhraseCutDataset/data/VGPhraseCut_v0/output_{split}/{task_ids[j]}.jpg")
                j += 1
    print("done")

generator(split = 'val')
generator(split = 'train')