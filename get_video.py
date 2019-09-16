import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
import random

#from lanenet_repo.tools import test_lanenet

ROOT_PATH = ''


def get_video_frames(src, fpv, frame_height, frame_width):
    # print('reading video from', src)
    cap = cv2.VideoCapture(src)
    #filename = src.split('/')[3][:-4]

    frames = []
    if not cap.isOpened():
        cap.open(src)
    ret = True
    while(True and ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

    rnd_idx = random.randint(5,len(frames)-5)
    rnd_frame = frames[rnd_idx]
    rnd_frame = cv2.resize(rnd_frame,(224,224)) # Needed for Densenet121-2d

    step = len(frames)//fpv

    # get 16 frames from each half
    #frames_fhalf = frames[:len(frames)//2]
    #avg_frames_fhalf = frames_fhalf[::step]
    #frames_shalf = frames[len(frames)//2:]
    #avg_frames_shalf = frames_shalf[::step]
    #avg_frames = avg_frames_fhalf + avg_frames_shalf

    avg_frames = frames[::step]
    avg_frames = avg_frames[:fpv]
    avg_resized_frames = []
    for af in avg_frames:
        rsz_f = cv2.resize(af, (frame_width, frame_height))
        avg_resized_frames.append(rsz_f)

    # call lanenet for lane detection!
    #avg_resized_lanenet_frames = []
    #weights_path = '/home/yagiz/Sourcebox/git/T3D-keras/lanenet_repo/model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt'
    #counts = 0
    #for image in avg_resized_frames:
    #    lanenet_frame = test_lanenet.test_lanenet(image, weights_path, filename, counts)
    #    counts += 1
    #    avg_resized_lanenet_frames.append(lanenet_frame)
    return np.asarray(rnd_frame)/255.0,np.asarray(avg_resized_frames)


def get_video_and_label(index, data, frames_per_video, frame_height, frame_width):
    # Read clip and appropiately send the behavior class
    frame, clip = get_video_frames(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()), frames_per_video, frame_height, frame_width)
    behavior_class = data['class'].values[index]

    frame = np.expand_dims(frame, axis=0)
    clip = np.expand_dims(clip, axis=0)

    # print('Frame shape',frame.shape)
    # print('Clip shape',clip.shape)

    return frame, clip, behavior_class


def video_gen(data, frames_per_video, frame_height, frame_width, channels, num_classes, batch_size=4):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])

        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            clip = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)
            frame = np.empty([0, 224, 224, 3], dtype=np.float32)

            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get frames and its corresponding color for an traffic light
                single_frame, single_clip, behavior_class = get_video_and_label(
                    i, data, frames_per_video, frame_height, frame_width)

                # Appending them to existing batch
                try:
                    frame = np.append(frame, single_frame, axis=0)
                    clip = np.append(clip, single_clip, axis=0)
                except Exception as e:
                    print(e, single_frame.shape, single_clip.shape, frame.shape, clip.shape)
                y_train = np.append(y_train, [behavior_class])
            y_train = to_categorical(y_train, num_classes=num_classes)

            yield ([frame, clip], y_train)
