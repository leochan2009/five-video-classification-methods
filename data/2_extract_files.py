"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call
def writeGroundTruthCSV(generated_files_truth, outputFileName):
    with open(outputFileName, 'a') as fout:
        writer = csv.writer(fout)
        for file_truth in generated_files_truth:
            writer.writerow(file_truth)

def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['train', 'test']
    classes = ["Blood","Clarity", "Particles"]
    for folder in folders:
        videos = glob.glob(os.path.join(folder, '*.mp4'))

        for video in videos:
            for name in classes:
                # Get the parts of the file.
                #video_parts = get_video_parts(video_path)
                #train_or_test, classname, filename_no_ext, filename = video_parts
                parts = video.split(os.path.sep)
                train_or_test = parts[0]
                filename = parts[1]
                validFrames = []
                validFramesGroundtruth = []
                outputFileName= './'+ train_or_test + '/' + train_or_test + '-' + name +'.csv'
                with open(outputFileName, 'w') as fout:
                    writer = csv.writer(fout)
                    writer.writerow(['id', 'label'])
                with open(os.path.join(train_or_test, filename[:-4]+ name+".csv"), 'r') as fin:
                    reader = csv.reader(fin, delimiter = ',')
                    for row in reader:
                        if float(row[1])>-1:
                            validFrames.append(int(row[0]))
                            validFramesGroundtruth.append(float(row[1]))
                    filename_no_ext = filename.split('.')[0]
                    # Only extract if we haven't done it yet. Otherwise, just get
                    # the info.
                    if not bool(os.path.exists(os.path.join(train_or_test, filename_no_ext + '-00000.png'))):
                        #if not check_already_extracted(video_parts):
                        # Now extract it.
                        src = os.path.join(train_or_test, filename)
                        dest = os.path.join(train_or_test,
                                            filename_no_ext + '-%05d.png')
                        call(["ffmpeg", "-i", src, "-start_number", "0", dest])

                    # Now get how many frames it is.
                    #nb_frames = get_nb_frames_for_video(video_parts)
                    generated_files = sorted(glob.glob(os.path.join(train_or_test, filename_no_ext + '*.png')))
                    nb_frames = len(generated_files)
                    # delete the unused frames
                    generated_files_truth = []
                    for frame in range(nb_frames):
                        # if not (frame in validFrames):
                        #     os.remove(os.path.join(generated_files[frame])) no need to remove files as blood, particle or clarity need different frames
                        if  frame in validFrames:
                            generated_files_truth.append([generated_files[frame].split(os.path.sep)[1], validFramesGroundtruth[validFrames.index(frame)]])
                    writeGroundTruthCSV(generated_files_truth, outputFileName)
                    data_file.append([train_or_test, filename_no_ext, len(generated_files_truth)])



                    print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(train_or_test, classname,
                                filename_no_ext + '*.png'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
