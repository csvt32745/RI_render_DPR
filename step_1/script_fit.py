from fit_3DDFA import *
import cv2
import os

def read_faceList():
    faceList = []
    with open('../data.list') as f:
        for line in f:
            faceList.append(line.strip())
    return faceList
faceList = read_faceList()

imgPath = '../data/'
savePath = '../result'
if not os.path.exists(savePath):
    os.makedirs(savePath)

fit_3DMM = fit_3DDFA('gpu')

# for item in faceList:
#     imgName = item.split('.')[0]
#     subFolder = os.path.join(savePath, imgName)
#     if not os.path.exists(subFolder):
#         os.makedirs(subFolder)
#     img = cv2.imread(os.path.join(imgPath, item))
#     fit_3DMM.forward(img, subFolder, item.split('.')[0])

batch = 10
# exist_faces = [i+".png" for i in os.listdir(savePath)]
# faceList = set(faceList)
# faceList = (faceList ^ set(exist_faces))

faceList = list(faceList)
print(f"# of face: {len(faceList)}")

defected_list = []
for i in range(0, len(faceList), batch):
    items = faceList[i:i+batch]
    imgNames = [item.split('.')[0] for item in items]
    subFolders = [os.path.join(savePath, imgName) for imgName in imgNames]
    imgs = [cv2.imread(os.path.join(imgPath, item)) for item in items]
    defected_list += fit_3DMM.forward(imgs, subFolders, imgNames, items)

print(f"There is {len(defected_list)} defected images")
new_faceList = list(set(read_faceList()) ^ set(defected_list))
# print(len(orig_faceList))
with open('../data.list', mode='w') as f:
    for n in new_faceList:
        f.write(n+"\n")

with open('../defected.list', mode='w') as f:
    for n in defected_list:
        f.write(n+"\n")
