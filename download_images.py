import pandas as pd
import os
import aiohttp
import aiofiles
import asyncio
from tqdm.asyncio import tqdm_asyncio
from PIL import Image

async def download_image(image_link, session, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return
    
    filename = os.path.basename(image_link)
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return
    
    for _ in range(retries):
        try:
            async with session.get(image_link) as response:
                if response.status == 200:
                    async with aiofiles.open(image_save_path, 'wb') as f:
                        await f.write(await response.read())
                    return
                else:
                    print(f"Error downloading {image_link}: Status {response.status}")
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
            await asyncio.sleep(delay)
    
    await create_placeholder_image(image_save_path) 

async def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        async with aiofiles.open(image_save_path, 'wb') as f:
            placeholder_image.save(f, format='JPEG')
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

async def main():
    df_test = pd.read_csv('student_resource 3/dataset/test.csv')
    download_folder = 'images'
    os.makedirs(download_folder, exist_ok=True)

    image_links = df_test['image_link'].tolist()
    
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(img_link, session, download_folder) for img_link in image_links]
        await tqdm_asyncio.gather(*tasks, desc="Downloading images")

    print(f"Images have been downloaded to {download_folder}")

if __name__ == '__main__':
    asyncio.run(main())