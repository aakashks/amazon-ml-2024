import threading
import time
from PIL import Image
import os

def io_task(file):
    print("Task started")
    print(f"Processing file: {file}")
    img = Image.open(file)
    print(f"Image size: {img.size}")
    img_list.append(img)
    print("Task completed")
    

threads = []

dir = "images_test"


if __name__ == '__main__':
    time_start = time.time()
    img_list = []
    
    all_files = os.listdir(dir)
    
    for img in all_files[:1000]:
        t = threading.Thread(target=io_task, args=(os.path.join(dir, img),))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()

        
    print(f"Time taken = {time.time() - time_start}")
