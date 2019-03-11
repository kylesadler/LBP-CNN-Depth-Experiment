import os
import numpy as np
import cv2
import h5py


def main():

    save_size = 256
    frames_per_vid = 60
    batch_size = 32
    save_dir = "cropped_samples"
    dataset_path = "SiW/SiW_release/Train"
    
    
    if(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    
    face_list = []
    label_list = []

    for folder in os.listdir(dataset_path): # subject folders
        
        # process all videos in subject folder
        for video in get_videos(os.path.join(dataset_path, folder)):
            
            dataset_name = video.split("/")[-1].split(".")[0]
            
            if(dataset_name.split("-")[2] == "1"):
                video_label = [1, 0]
            else:
                video_label = [0, 1]
                #[live, spoofed]
            
            face_points = open(video[:-3]+"face", "r")
            vidcap = cv2.VideoCapture(video)
            
            success, frame = vidcap.read()
            count = 0
            
            while(count < frames_per_vid and success):
           
                try:
                    assert(len(frame.shape) == 3)    
                    assert(frame.shape[2] == 3)
                        
                    frame = crop_image(frame, parse_line(face_points.readline()))
                    frame = cv2.resize(frame, (save_size, save_size))
                    
                    if(count % 27 == 18):
                        cv2.imsave(os.path.join(save_dir, dataset_name+"_frame"+str(count)),frame)
                    
                    #cv2.imshow('image',frame)
                    #cv2.waitKey(27)
                    
                    # add frame channels to face_list
                    for channel in range(3):
                        
                        face_list.append(frame[:,:,channel])
                        label_list.append(video_label)
                        
                        if(len(face_list) >= batch_size):
                            h5py_img_file = h5py.File("none.h5",  'a')
                            h5py_label_file = h5py.File("labels.h5",  'a')
                            
                            h5py_img_file.create_dataset(dataset_name+str(count), data=np.asarray(face_list))
                            h5py_label_file.create_dataset(dataset_name+str(count), data=np.asarray(label_list))
                            
                            h5py_img_file.close()
                            h5py_label_file.close()
                            
                            face_list = []
                            label_list = []
                    
                    
                except AssertionError:
                    print("could not process frame " +str(count)+" in video "+dataset_name)
                except Exception as e:
                    raise e
                finally:
                    success, frame = vidcap.read()
                    count += 1
            
            
            print(video)
      
    if(len(face_list) > 0):    
        h5py_img_file = h5py.File("none.h5",  'a')
        h5py_label_file = h5py.File("labels.h5",  'a')
        
        h5py_img_file.create_dataset(dataset_name+str(count), data=np.asarray(face_list))
        h5py_label_file.create_dataset(dataset_name+str(count), data=np.asarray(label_list))
        
        h5py_img_file.close()
        h5py_label_file.close()

def parse_line(line):
    """ returns list length 4 [startX, startY, endX, endY] """
    try:
        points = line.split(" ")
        assert(len(points) == 4)
        points = [int(x) for x in points]
        return points
    except:
        raise AssertionError
    
def get_file_pairs(directory):
    """ return list of .mov files and the corresponding list of .face files in given dir """ 
    videos = []
    faces = []
    
    for file in os.listdir(directory):
        if(file[-4:] == "face"):
            faces.append(os.path.join(directory, file))
        elif(file[-3:] == "mov" or file[-3:] == "mp4"):
            videos.append(os.path.join(directory, file))
            
    videos.sort()
    faces.sort()
    
    return videos, faces
    
def get_videos(directory):
    """ return list of .mov and .mp4 files in given directory """ 
    videos = []
    
    for file in os.listdir(directory):
        if(file[-3:] == "mov" or file[-3:] == "mp4"):
            videos.append(os.path.join(directory, file))

    return videos

def crop_image(frame, bbox):
    """ bbox is a valid bbox as detemined by parse_line """
    startY = max(bbox[1], 0)
    endY = min(bbox[3], img.shape[0])
    startX = max(bbox[0], 0)
    endX = min(bbox[2], img.shape[1])
    return img[ startY:endY, startX : endX ]
    
if(__name__ == "__main__"):
    main()
