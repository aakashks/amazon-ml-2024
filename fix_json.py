import pandas as pd
import json
import os
from tqdm import tqdm

df = pd.read_csv("fracdata/dataset_part0.csv")

# for every json file in directory

dir  = "output/0/"

all_data = []

for filename in tqdm(os.listdir(dir)):
    with open(os.path.join(dir, filename), "r") as f:
        data = json.load(f)
        
    for item in data:
        item['image_link'] = df.loc[df['index'] == item['index'], 'image_link'].values[0]
        
    all_data.extend(data)
    
with open("output/qwen_0.json", "w") as f:
    json.dump(all_data, f, indent=4)
