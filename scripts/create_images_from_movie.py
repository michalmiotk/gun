import cv2
import numpy as np
import os

filename = '/home/m/Pobrane/składanka.mp4'
output_dir = '/home/m/images_from_video'
save_every_n_frames = 20


cap = cv2.VideoCapture(filename)



# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_nr = 0
os.makedirs(output_dir, exist_ok = True)

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
  
    if frame_nr%save_every_n_frames == 0:
        image_name = os.path.join(output_dir, str(int(frame_nr/save_every_n_frames))+'.png')
        cv2.imwrite(image_name, frame)
    frame_nr +=1

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

