import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import numpy
import os
import SimpleITK as sitk

output_size =[368, 368, 96]

def covert_h5():
    listt = glob("G:\DATASET/NIH/NII\image/*.nii")
    print(listt)
    for item in tqdm(listt):

        image =read_img(item).transpose(1,2,0)
        label = read_img(item.replace('image', 'label')).transpose(1,2,0)
        #label = numpy.zeros(image.shape)
        # print(label.shape)

        w, h, d = label.shape
        tempL = np.nonzero(image)
        #tempL = np.array([256,256,20])
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        arr=[[minx,maxx],[miny,maxy],[minz,maxz]]
        print(arr)
        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy,minz:maxz]
        label = label[minx:maxx, miny:maxy,minz:maxz]
        print(label.shape)

        f = h5py.File(item.replace('.nii', '_h5.h5').replace("image","h5"), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()
def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data
if __name__ == '__main__':
    covert_h5()

