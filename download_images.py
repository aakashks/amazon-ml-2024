import pandas as pd
import os
import urllib.request
from tqdm import tqdm

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = os.path.basename(image_link)
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
    
    create_placeholder_image(image_save_path) 

def create_placeholder_image(image_save_path):
    try:
        from PIL import Image
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

df_test = pd.read_csv('student_resource 3/dataset/test.csv')

download_folder = 'images'
os.makedirs(download_folder, exist_ok=True)

image_links = df_test['image_link'].tolist()

for img_link in tqdm(image_links, desc="Downloading images"):
    download_image(img_link, download_folder)

print(f"Images have been downloaded to {download_folder}")
