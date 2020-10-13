import csv, os
import numpy as np
import json
# path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
#             '-' + data_type + '.npy')
fileNames = ['20180802-094306_912-MatchedToMP4.csv', '20180802-094523_456-MatchedToMP4.csv', '20180802-095434_390-MatchedToMP4.csv',
             '20180802-100436_497-MatchedToMP4.csv', '20180802-134041_790-MatchedToMP4.csv', '20180802-141251_987-MatchedToMP4.csv']
def readCSV(cname, lastColumn= -1):
    print("Reading: ", cname)
    with open(cname, "r") as file:
        legend = file.readline()
        print("File legend: " + legend)
        if lastColumn == 0:
            akeys = [k.strip() for k in legend.split(',')[:]]
        else:
            akeys = [k.strip() for k in legend.split(',')[:lastColumn]]
        ametrics = np.empty([0,len(akeys)])
        row = np.zeros([1,len(akeys)])
        iL = 0
        started = False;
        for line in file:
            toks = line.split(',')
            sec = float(toks[0])
            if sec < .2:            # Can have bad samples at the beginning w/ the end time from a previous video file
                started=True
            if started:
                iL+=1
                if iL < 4 or iL % 1500 == 0: print("    Time offset: " + toks[0])
                if lastColumn == 0:
                    row = [float(t) for t in toks[:]]
                else:
                    row = [float(t) for t in toks[:-1]]
                ametrics = np.vstack([ametrics,row])
    return ametrics

seq_length = 30
for fileName in fileNames:
    metrics = readCSV(os.path.join(os.getcwd(), 'data', fileName),0)
    metricsTrimmed = np.hstack((metrics[:,2:8] , metrics[:, 9:]))

    annotation_file = fileName[:fileName.find('-MatchedToMP4')] + '.json' #'ourdata_ordinalRegression.json'
    labels = None
    classnames = {}
    if not os.path.isfile(os.path.join(os.getcwd(),'data', annotation_file)):
        print('No annotation found at %s' % annotation_file)
    else:
        with open(os.path.join(os.getcwd(),'data', annotation_file), 'r') as json_file:
            try:
                annotations = json.load(json_file)
                print('Existing annotation found: %d items' % len(annotations))
            except json.JSONDecodeError:
                print('Unable to load annotations from %s' % annotation_file)

        # Check for absolute/relative paths of annotated videos and
        # make sure that labels are valid
        for anno in annotations:
            # If the annotation has a relative path, it is relative to the
            # annotation file's folder
            clipName = anno['video']
            clipIndex = int(clipName[clipName.rfind('_clip_') + 6:clipName.rfind('.mp4')])
            if anno['label'] == "clarity 25":
                classnames[str(clipIndex)] = 1
            elif anno['label'] == "clarity 50":
                classnames[str(clipIndex)] = 2
            elif anno['label'] == "clarity 75":
                classnames[str(clipIndex)] = 3
            elif anno['label'] == "clarity 100":
                classnames[str(clipIndex)] = 4


    data_file = []
    for i in range(int(metrics.shape[0]/30) + 1):
        if i < int(metricsTrimmed.shape[0]/30):
            metrictobesave= metricsTrimmed[i*30:(i+1)*30,:]
        else:
            metrictobesave = metricsTrimmed[i * 30:, :]
        numpyfileName = fileName[:-4] + '-' + str(i) +'-' + str(seq_length) + '-features' + '.npy'
        np.save(os.path.join(os.getcwd(), 'data', 'sequences', numpyfileName), metrictobesave)
        if classnames.get(str(i)):
            data_file.append(["train", classnames[str(i)], fileName[:-4] + '-' + str(i), metrictobesave.shape[0]])

    with open(os.path.join(os.getcwd(),'data', 'data_file_ordinal_logistic_regression.csv'), 'a') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

