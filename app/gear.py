# A Redis gear for orchestrating realtime video analytics
import io
import cv2
import redisAI
import numpy as np
from time import time
from PIL import Image

from redisgears import executeCommand as execute

# Globals for downsampling
_mspf = 1000 / 10.0      # Msecs per frame (initialized with 10.0 FPS)
_next_ts = 0             # Next timestamp to sample a frame

def xlog(*args):
    redisgears.executeCommand('xadd', 'log', '*', 'text', ' '.join(map(str, args))) 

class SimpleMovingAverage(object):
    ''' Simple moving average '''
    def __init__(self, value=0.0, count=7):
        '''
        @value - the initialization value
        @count - the count of samples to keep
        '''
        self.count = int(count)
        self.current = float(value)
        self.samples = [self.current] * self.count

    def __str__(self):
        return str(round(self.current, 3))

    def add(self, value):
        ''' Adds the next value to the average '''
        v = float(value)
        self.samples.insert(0, v)
        o = self.samples.pop()
        self.current = self.current + (v-o)/self.count

class Profiler(object):
    ''' Mini profiler '''
    names = []  # Steps names in order
    data = {}   # ... and data
    last = None
    def __init__(self):
        pass

    def __str__(self):
        s = ''
        for name in self.names:
            s = '{}{}:{}, '.format(s, name, self.data[name])
        return(s[:-2])

    def __delta(self):
        ''' Returns the time delta between invocations '''
        now = time()*1000       # Transform to milliseconds
        if self.last is None:
            self.last = now
        value = now - self.last
        self.last = now
        return value

    def start(self):
        ''' Starts the profiler '''
        self.last = time()*1000

    def add(self, name):
        ''' Adds/updates a step's duration '''
        value = self.__delta()
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def assign(self, name, value):
        ''' Assigns a step with a value '''
        if name not in self.data:
            self.names.append(name)
            self.data[name] = SimpleMovingAverage(value=value)
        else:
            self.data[name].add(value)

    def get(self, name):
        ''' Gets a step's value '''
        return self.data[name].current

'''
The profiler is used first and foremost for keeping track of the total (average) time it takes to process
a frame - the information is required for setting the FPS dynamically. As a side benefit, it also provides
per step metrics.
'''
prf = Profiler()

def downsampleStream(x):
    ''' Drops input frames to match FPS '''
    global _mspf, _next_ts
    execute('TS.INCRBY', 'camera:0:in_fps', 1, 'RESET', 1)  # Store the input fps count
    ts, _ = map(int, str(x['id']).split('-'))         # Extract the timestamp part from the message ID
    sample_it = _next_ts <= ts
    if sample_it:                                           # Drop frames until the next timestamp is in the present/past
        _next_ts = ts + _mspf
    return sample_it

def process_image(img, height):
    ''' Utility to resize a rectangular image to a padded square (letterbox) '''
    color = (127.5, 127.5, 127.5)
    shape = img.shape[:2]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (height - new_shape[0]) / 2    # Width padding
    dh = (height - new_shape[1]) / 2    # Height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img = np.asarray(img, dtype=np.float32)
    img /= 255.0                        # Normalize 0..255 to 0..1.00
    return img

def runYolo(x):
    ''' Runs the model on an input image from the stream '''
    global prf
    IMG_SIZE = 416     # Model's input image size
    prf.start()        # Start a new profiler iteration

    # log('read')

    # Read the image from the stream's message
    buf = io.BytesIO(x['value']['image'])
    pil_image = Image.open(buf)
    numpy_img = np.array(pil_image)
    prf.add('read')

    # log('resize')
    # Resize, normalize and tensorize the image for the model (number of images, width, height, channels)
    image = process_image(numpy_img, IMG_SIZE)
    # log('tensor')
    img_ba = bytearray(image.tobytes())
    image_tensor = redisAI.createTensorFromBlob('FLOAT', [1, IMG_SIZE, IMG_SIZE, 3], img_ba)
    prf.add('resize')

    # log('model')
    # Create the RedisAI model runner and run it
    modelRunner = redisAI.createModelRunner('yolo:model')
    redisAI.modelRunnerAddInput(modelRunner, 'input', image_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output')
    model_replies = redisAI.modelRunnerRun(modelRunner)
    model_output = model_replies[0]
    prf.add('model')

    # log('script')
    # The model's output is processed with a PyTorch script for non maxima suppression
    scriptRunner = redisAI.createScriptRunner('model', 'boxes_from_tf')
    redisAI.scriptRunnerAddInput(scriptRunner, model_output)
    redisAI.scriptRunnerAddOutput(scriptRunner)
    script_reply = redisAI.scriptRunnerRun(scriptRunner)
    prf.add('script')

    # log('boxes')
    # The script outputs bounding boxes
    shape = redisAI.tensorGetDims(script_reply[0])
    buf = redisAI.tensorGetDataAsBlob(script_reply[0])
    boxes = np.frombuffer(buf, dtype=np.float32).reshape(shape)

    # Iterate boxes to extract the people
    ratio = float(IMG_SIZE) / max(pil_image.width, pil_image.height)  # ratio = old / new
    pad_x = (IMG_SIZE - pil_image.width * ratio) / 2                  # Width padding
    pad_y = (IMG_SIZE - pil_image.height * ratio) / 2                 # Height padding
    boxes_out = []
    people_count = 0
    for box in boxes[0]:
        if box[4] == 0.0:  # Remove zero-confidence detections
            continue
        if box[-1] != 14:  # Ignore detections that aren't people
            continue
        people_count += 1

        # Descale bounding box coordinates back to original image size
        x1 = (IMG_SIZE * (box[0] - 0.5 * box[2]) - pad_x) / ratio
        y1 = (IMG_SIZE * (box[1] - 0.5 * box[3]) - pad_y) / ratio
        x2 = (IMG_SIZE * (box[0] + 0.5 * box[2]) - pad_x) / ratio
        y2 = (IMG_SIZE * (box[1] + 0.5 * box[3]) - pad_y) / ratio

        # Store boxes as a flat list
        boxes_out += [x1,y1,x2,y2]
    prf.add('boxes')

    return x['id'], people_count, boxes_out

def storeResults(x):
    ''' Stores the results in Redis Stream and TimeSeries data structures '''
    global _mspf, prf
    ref_id, people, boxes= x[0], int(x[1]), x[2]
    ref_msec = int(str(ref_id).split('-')[0])

    # Store the output in its own stream
    res_id = execute('XADD', 'camera:0:yolo', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)

    # Add a sample to the output people and fps timeseries
    res_msec = int(str(res_id).split('-')[0])
    execute('TS.ADD', 'camera:0:people', ref_msec, people)
    execute('TS.INCRBY', 'camera:0:out_fps', 1, 'RESET', 1)

    # Adjust mspf to the moving average duration
    total_duration = res_msec - ref_msec
    prf.assign('total', total_duration)
    avg_duration = prf.get('total')
    _mspf = avg_duration * 1.05  # A little extra leg room

    # Record profiler steps
    for name in prf.names:
        current = prf.data[name].current
        execute('TS.ADD', 'camera:0:prf_{}'.format(name), ref_msec, current)

    prf.add('store')
    # Make an arithmophilial homage to Count von Count for storage in the execution log
    if people == 0:
        return 'Now there are none.'
    elif people == 1:
        return 'There is one person in the frame!'
    elif people == 2:
        return 'And now there are are two!'
    else:
        return 'I counted {} people in the frame! Ah ah ah!'.format(people)

# Create and register a gear that for each message in the stream
gb = GearsBuilder('StreamReader')
gb.filter(downsampleStream)  # Filter out high frame rate
gb.map(runYolo)              # Run the model
gb.map(storeResults)         # Store the results
gb.register('camera:0')
