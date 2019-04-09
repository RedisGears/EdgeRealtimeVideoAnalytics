# RedisEdge realtime video analytics video capture script
import argparse
import cv2
import redis
from urllib.parse import urlparse

MAX_DIM = 640
class Webcam:
    def __init__(self, infile=0, fps=30.0):
        self.infile = infile
        self.cam = cv2.VideoCapture(self.infile)
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        # Read image
        ret_val, img0 = self.cam.read()
        if not ret_val:  # Try to stupidly loop over video files
            self.cam = cv2.VideoCapture(self.infile)
            ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'

        # Preprocess
        img = cv2.flip(img0, 1)

        return self.count, img

    def __len__(self):
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Input file (leave empty to use webcam)', nargs='?', type=str, default=None)
    parser.add_argument('-o', '--output', help='Output stream key name', type=str, default='camera:0')
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    parser.add_argument('-w', '--webcam', help='Webcam device number', type=int, default=0)
    parser.add_argument('-v', '--verbose', help='Verbose output', type=bool, default=False)
    parser.add_argument('--fmt', help='Frame storage format', type=str, default='.jpg')
    parser.add_argument('--fps', help='Frames per second (webcam)', type=float, default=15.0)
    parser.add_argument('--maxlen', help='Maximum length of output stream', type=int, default=10000)
    args = parser.parse_args()

    # Set up Redis connection
    url = urlparse(args.url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    if not conn.ping():
        raise Exception('Redis unavailable')

    # Choose video source
    if args.infile is None:
        loader = Webcam(infile=args.webcam, fps=args.fps)  # Default to webcam
    else:
        loader = Webcam(infile=args.infile, fps=args.fps)  # Unless an input file (image or video) was specified

    for (count, img) in loader:
        _, data = cv2.imencode(args.fmt, img)
        msg = {
            'count': count,
            'image': data.tobytes()
        }
        _id = conn.xadd(args.output, msg, maxlen=args.maxlen)
        if args.verbose:
            print('frame: {} id: {}'.format(count, _id))
