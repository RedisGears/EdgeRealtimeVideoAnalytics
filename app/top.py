# RedisEdge realtime video analytics video realtime performance metrics
import argparse
import math
import os
import pprint
import time
import redis
from urllib.parse import urlparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', help='Refresh period (sec)', type=float, default=0.5)
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('-v', '--video', help='Video stream key name', type=str, default='camera:0')
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    metrics = [
        'in_fps',
        'out_fps',
        'prf_read',
        'prf_resize',
        'prf_model',
        'prf_script',
        'prf_boxes',
        'prf_store',
        'prf_total',
    ]
    samples = {
    }
    for m in metrics:
        samples[m] = None

    while True:
        now = int(time.time())
        p = conn.pipeline(transaction=False)
        for m in metrics:
            p.execute_command('TS.RANGE', f'{args.video}:{m}', now - 2, now - 1)
        res = p.execute()
        res.reverse()
        for m in metrics:
            r = res.pop()
            if r:
                ts = r[0][0]
                metric = round(float(r[0][1].decode('utf-8')),1)
            else:
                ts = 'N/A'
                metric = 'N/A'
            samples[m] = metric
        line = ' '.join([f'{m}: {samples[m]}' for m in metrics])
        line = f'ts: {ts} {line}'
        print(line)
        time.sleep(1.0)