# RedisEdge realtime video analytics initialization script
import argparse
import redis
from urllib.parse import urlparse

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='CPU or GPU', type=str, default='CPU')
    parser.add_argument('-i', '--camera_id', help='Input video stream key camera ID', type=str, default='0')
    parser.add_argument('-p', '--camera_prefix', help='Input video stream key prefix', type=str, default='camera')
    parser.add_argument('-u', '--url', help='RedisEdge URL', type=str, default='redis://127.0.0.1:6379')
    args = parser.parse_args()

    # Set up some vars
    input_stream_key = '{}:{}'.format(args.camera_prefix, args.camera_id)  # Input video stream key name
    initialized_key = '{}:initialized'.format(input_stream_key)

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    # Check if this Redis instance had already been initialized
    initialized = conn.exists(initialized_key)
    if initialized:
        print('Discovered evidence of a privious initialization - skipping.')
        exit(0)

    # Load the RedisAI model
    print('Loading model - ', end='')
    with open('models/tiny-yolo-voc.pb', 'rb') as f:
        model = f.read()
        res = conn.execute_command('AI.MODELSTORE', 'yolo:model', 'TF', args.device, 'INPUTS', 1,  'input', 'OUTPUTS', 1, 'output', 'BLOB', model)
        print(res)

    # Load the PyTorch post processing boxes script
    print('Loading script - ', end='')
    with open('yolo_boxes.py', 'rb') as f:
        script = f.read()
        res = conn.execute_command('AI.SCRIPTSTORE', 'yolo:script', args.device, 'ENTRY_POINTS', 1, 'boxes_from_tf', 'SOURCE', script)
        print(res)

    print('Creating timeseries keys and downsampling rules - ', end='')
    res = []                                                             # RedisTimeSeries replies list
    labels = ['LABELS', args.camera_prefix, args.camera_id, '__name__']  # A generic list of timeseries keys labels
    # Create the main timeseries key
    res.append(conn.execute_command('TS.CREATE', '{}:people'.format(input_stream_key), *labels, 'people'))
    # Set up timeseries downsampling keys and rules
    wins = [1, 5, 15]             # Downsampling windows
    aggs = ['avg', 'min', 'max']  # Downsampling aggregates
    for w in wins:
        for a in aggs:
            res.append(conn.execute_command('TS.CREATE', '{}:people:{}:{}m'.format(input_stream_key, a, w), *labels, 'people_{}_{}m'.format(a, w)))
            res.append(conn.execute_command('TS.CREATERULE', '{}:people'.format(input_stream_key), '{}:people:{}:{}m'.format(input_stream_key, a, w), 'AGGREGATION', a, w*60))
    # Set up fps timeseries keys
    res.append(conn.execute_command('TS.CREATE', '{}:in_fps'.format(input_stream_key), *labels, 'in_fps'))
    res.append(conn.execute_command('TS.CREATE', '{}:in_fps_count'.format(input_stream_key), *labels, 'in_fps_count'))
    res.append(conn.execute_command('TS.CREATERULE', '{}:in_fps'.format(input_stream_key), '{}:in_fps_count'.format(input_stream_key), 'AGGREGATION', 'count', 1000))
    res.append(conn.execute_command('TS.CREATE', '{}:out_fps'.format(input_stream_key), *labels, 'out_fps'))
    res.append(conn.execute_command('TS.CREATE', '{}:out_fps_count'.format(input_stream_key), *labels, 'out_fps_count'))
    res.append(conn.execute_command('TS.CREATERULE', '{}:out_fps'.format(input_stream_key), '{}:out_fps_count'.format(input_stream_key), 'AGGREGATION', 'count', 1000))

    # Set up profiler timeseries keys
    metrics = ['read', 'resize', 'model', 'script', 'boxes', 'store', 'total']
    for m in metrics:
        res.append(conn.execute_command('TS.CREATE', '{}:prf_{}'.format(input_stream_key,m), *labels, 'prf_{}'.format(m)))
    print(res)

    # Load the gear
    print('Loading gear - ', end='')
    with open('gear.py', 'rb') as f:
        gear = f.read()
        res = conn.execute_command('RG.PYEXECUTE', gear, "REQUIREMENTS", "opencv-python-headless<4.5", "Pillow", "numpy")
        print(res)

    # Lastly, set a key that indicates initialization has been performed
    print('Flag initialization as done - ', end='') 
    print(conn.set(initialized_key, 'most certainly.'))
