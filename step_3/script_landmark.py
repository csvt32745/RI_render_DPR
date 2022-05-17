from detect_landmark import *
import numpy as np
import cv2

detect_landmark = detect_landmark()


faceList = []
with open('../data.list') as f:
    for line in f:
        faceList.append(line.strip())

defected_list = []
with open('../defected.list') as f:
    for line in f:
        defected_list.append(line.strip())

imgPath = '../data'
savePath = '../result'

# print(len(faceList))
# print(len(defected_list))

# if not os.path.exists(savePath):
#     os.makedirs(savePath)
# TODO: split data that cannot detect landmark
for item in list(faceList):
    print(item)
    imgName = item.split('.')[0]
    subFolder = os.path.join(savePath, imgName, 'render')
    if os.path.exists(subFolder):
        if os.path.exists(os.path.join(subFolder, 'albedo_3DDFA.png')):
            print('existing')
            continue
        img = cv2.imread(os.path.join(subFolder, 'albedo.png'))
        albedo_landmark = detect_landmark.detect(img)
        if albedo_landmark is None:
            faceList.remove(item)
            defected_list.append(item)
            continue
        else:
            detect_landmark.save_landmark(albedo_landmark, 
                    os.path.join(subFolder, 'albedo_detected.txt'))
            detect_landmark.draw_landmark(albedo_landmark, img, 
                    os.path.join(subFolder, 'albedo_3DDFA.png'))
    else:
        print('no file {}'.format(imgName))

print(len(faceList))
print(len(defected_list))

with open('../defected.list', mode='w') as f:
    for n in defected_list:
        f.write(n+"\n")

with open('../data.list', mode='w') as f:
    for n in faceList:
        f.write(n+"\n")