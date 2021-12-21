import cv2
from glob import glob
import os

img_folder = os.path.join(os.getcwd(), 'output1')
frames_len = len(glob(img_folder+r'\*.jpg'))

vidwrite = cv2.VideoWriter('output_video1.mov', -1, 144, (720,1280))

for idx in range(frames_len):
  image = cv2.imread(os.path.join(img_folder, f'frame{idx}.jpg'))
  vidwrite.write(image) # write frame into video

vidwrite.release()
