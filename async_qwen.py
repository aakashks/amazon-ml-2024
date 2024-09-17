import asyncio
import aiohttp
import pandas as pd
import json
from tqdm.asyncio import tqdm_asyncio
import os
import gc

# Configuration
IMAGE_FOLDER = "/scratch/be205_29/images_test"
OUTPUT_FOLDER = "outputt/"
CSV_PATH = "fracdata/dataset_part0.csv"

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://0.0.0.0:8080/v1"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

CONCURRENT_REQUESTS = 64  # Number of concurrent API requests
BATCH_SIZE = 1000         # Number of images to process before writing to disk

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load DataFrame
df_test = pd.read_csv(CSV_PATH)
image_paths = df_test['image_link'].tolist()  # Assuming there's an 'image_path' column

# Prompt Template
PROMPT = """
ONLY ONE IMAGE IS PROVIDED TO YOU. Extract textual features notably height, Depth, Width, Maximum Weight Recommendation, Item Weight, Voltage, Wattage, Item_volume whichever visible STRICTLY.
"""

async def fetch(session, semaphore, image_url, index, entity_name):
    """
    Asynchronously fetch the API response for a single image.
    """
    async with semaphore:
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            }

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            async with session.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Adjust based on actual response structure
                    vlm_output = data['choices'][0]['message']['content']
                else:
                    vlm_output = f"Error: {response.status}"
        except Exception as e:
            vlm_output = f"Exception: {str(e)}"

        return {
            'index': int(index),
            'entity_name': entity_name,
            'vlm_output': vlm_output
        }

async def process_images(session, semaphore, images, start_idx):
    """
    Process a list of images asynchronously.
    """
    tasks = []
    for i, image_url in enumerate(images):
        idx = start_idx + i
        entity_name = df_test.at[idx, 'entity_name']
        tasks.append(fetch(session, semaphore, image_url, df_test.at[idx, 'index'], entity_name))
    
    results = await asyncio.gather(*tasks)
    return results

async def main_async():
    """
    Main asynchronous function to process all images.
    """
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=None)  # Adjust timeout as needed

    cumulative_results = []
    batch_counter = 1

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in tqdm_asyncio(range(0, len(image_paths), BATCH_SIZE), desc="Processing Batches"):
            end = min(start + BATCH_SIZE, len(image_paths))
            batch_images = image_paths[start:end]
            batch_results = await process_images(session, semaphore, batch_images, start)

            cumulative_results.extend(batch_results)

            # Write to file
            output_path = os.path.join(OUTPUT_FOLDER, f'batch_output_{batch_counter}.json')
            with open(output_path, 'w') as outfile:
                json.dump(batch_results, outfile, indent=4)
            
            print(f"----- Batch {batch_counter} saved with {len(batch_results)} results. -----")
            batch_counter += 1

            # Clear memory
            cumulative_results = []
            gc.collect()

    print("All batches processed successfully.")

if __name__ == "__main__":
    asyncio.run(main_async())
    