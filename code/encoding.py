import os
from retinaface import Retinaface
def putAll(dir):
    '''
    在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
    '''
    retinaface = Retinaface(1)

    list_dir = os.listdir(dir)
    image_paths = []
    names = []
    for name in list_dir:
        image_paths.append(dir + "/" + name)
        names.append(name.split(".")[0])

    retinaface.encode_face_dataset(image_paths, names)