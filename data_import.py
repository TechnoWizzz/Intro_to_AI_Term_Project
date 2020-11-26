import imageio
import json
import numpy as np
from os import listdir

#Create file location string, change to match your local minecraft data folder
files_path = "C:/Users/Kio/Desktop/INTRO_TO_AI/PROJECT/data/MineRLNavigateDense-v0"
#Get the file names of the path directory
video_files_list = listdir(files_path)

video_path = video_files_list[0]

reader = imageio.get_reader(files_path + "/" + video_path + "/recording.mp4")
f = open(files_path + "/" + video_path + "/metadata.json")
metadata = json.load(f)
frames = []
for i,im in enumerate(reader):
    frames.append(im)
#Loops through video file name list and import data from each recording folder (video recording and metadata)
# for file in video_files_list:
    # reader = imageio.get_reader(files_path + "/" + file + "/recording.mp4")
    # f = open(files_path + "/" + file + "/metadata.json")
    # metadata = json.load(f)
    # metadata_set.append(metadata)
    # frame_groups = []
    # for i,im in enumerate(reader):
        # frame_groups.append(im)
    # video_set.append(frame_groups)

#The data formatting is two lists containing an array
#Each video has an entry in the outer data structure

#Number of frames for a certain video
print(f'Number of frames: {len(frames)}, {metadata["true_video_frame_count"]}')

#Number of RGB pixels in a frame, 64 by 64, with an int (between 0 and 255) RBG number in each entry
print(f'Shape if frame data enry, a 64 by 64 array of RGB values: {frames[0].shape}')

#Values and keys stored in list in metadata_set
for key, value in metadata.items() :
    print (f'Key: {key}\n Value: {value}')
    
#Can import matplotlib to use imshow to display the images of each frame, if needed
import matplotlib.pyplot as plt
frame = frames[0]
plt.imshow(frame)
plt.show()