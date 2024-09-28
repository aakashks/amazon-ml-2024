import multiprocessing
import time

def cpu_bound_task(n):
    print(f"Task {n} started")
    time.sleep(1)

if __name__ == '__main__':
    start = time.time()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    lst = range(100)
    result = pool.map(cpu_bound_task, lst)
    pool.close()
    pool.join()
    
    print(f"Time taken = {time.time() - start}")