from tqdm import tqdm
import time

n = 10
m = 100
tot = n*m
with tqdm(total = tot) as pbar:
    for i in range(n):
        for j in range(m):
            time.sleep(0.1)
            # print(i, j)
            pbar.update(1)