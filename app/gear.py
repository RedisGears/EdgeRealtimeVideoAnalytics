# A Redis gear for orchestrating realtime video analytics
import cv2
import redisAI
import numpy as np
import traceback
from PIL import Image
from PIL import ImageDraw

# Configuration
FPS = 4.0           # Maximum number of frames per second to process TODO: move to config key

# Globals for downsampling
_mspf = 1000 / FPS  # Msecs per frame
_next_ts = 0        # Next timestamp to sample

def downsampleStream(x):
    ''' Drops input frames to match FPS '''
    global _mspf, _next_ts

    execute('TS.INCRBY', 'camera:0:in_fps', 1, 'RESET', 1)  # Store the input fps count
    ts, _ = map(int, str(x['streamId']).split('-'))         # Extract the timestamp part from the message ID
    if _next_ts <= ts:                                      # Drop frames until the next timestamp is in the present/past
        _next_ts = ts + _mspf
        return True
    return False

def process_image(img, height):
    ''' Resizes a rectangular image to a padded square '''
    color = (127.5, 127.5, 127.5)
    shape = img.shape[:2]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (height - new_shape[0]) / 2    # Width padding
    dh = (height - new_shape[1]) / 2    # Height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img = np.asarray(img, dtype=np.float32) 
    img /= 255.0
    return img

def runYolo(x):
    ''' Runs the model on an input image using RedisAI '''
    IMG_SIZE = 736

    # Read the image from the stream's message
    buf = io.BytesIO(x['image'])
    pil_image = Image.open(buf)
    numpy_img = np.array(pil_image)
    image = process_image(numpy_img, IMG_SIZE)

    # Prepare the image and shape tensors as model inputs
    image_tensor = redisAI.createTensorFromBlob('FLOAT', [1,IMG_SIZE,IMG_SIZE,3], image.tobytes())
    shape_tensor = redisAI.createTensorFromValues('FLOAT', [2], [IMG_SIZE,IMG_SIZE])

    # Create yolo's RedisAI model runner and run it
    modelRunner = redisAI.createModelRunner('yolo:model')
    redisAI.modelRunnerAddInput(modelRunner, 'input_1', image_tensor)
    redisAI.modelRunnerAddInput(modelRunner, 'input_image_shape', shape_tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'concat_13')
    redisAI.modelRunnerAddOutput(modelRunner, 'concat_12')
    redisAI.modelRunnerAddOutput(modelRunner, 'concat_11')
    model_reply = redisAI.modelRunnerRun(modelRunner)

    # Get the model's outputs
    classes_tensor = model_reply[0]
    shape = redisAI.tensorGetDims(classes_tensor)
    buf = redisAI.tensorGetDataAsBlob(classes_tensor)
    classes = np.frombuffer(buf, dtype=np.float32).reshape(shape)
    people_count = np.unique(classes, return_counts=True)[1][0]  # 0 is people
    boxes_tensor = model_reply[2]
    shape = redisAI.tensorGetDims(boxes_tensor)
    buf = redisAI.tensorGetDataAsBlob(boxes_tensor)
    boxes = np.frombuffer(buf, dtype=np.float32).reshape(shape)

    boxes_out = []
    ratio = float(IMG_SIZE) / max(pil_image.width, pil_image.height)  # ratio = old / new
    pad_x = (IMG_SIZE - pil_image.width * ratio) / 2                  # Width padding
    pad_y = (IMG_SIZE - pil_image.height * ratio) / 2                 # Height padding
    for ind, class_val in enumerate(classes):
        if class_val == 0:  # 0 is people
            top, left, bottom, right = boxes[ind]
            # Descale coordinates back to original image size
            x1 = (left - pad_x) / ratio
            x2 = (right - pad_x) / ratio
            y1 = (top - pad_y) / ratio
            y2 = (bottom - pad_y) / ratio
            boxes_out += [x1,y1,x2,y2]
    return x['streamId'], people_count, boxes_out

def storeResults(x):
    ''' Stores the results in Redis Stream and TimeSeries data structures '''
    ref_id, people, boxes= x[0], int(x[1]), x[2]
    ref_msec = int(str(ref_id).split('-')[0])

    # Store the output in its own stream
    res_id = execute('XADD', 'camera:0:yolo', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)

    # Add a sample to the output people and fps timeseries
    res_msec = int(str(res_id).split('-')[0])
    execute('TS.ADD', 'camera:0:people', ref_msec/1000, people)
    execute('TS.INCRBY', 'camera:0:out_fps', 1, 'RESET', 1)

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
