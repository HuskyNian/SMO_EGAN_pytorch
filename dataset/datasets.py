import os
import sys
import random
import numpy as np
import h5py
import lmdb
import io 
import torch
import sklearn
#set modules path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','utils'))
#db utils
from utils import config
from utils.sysfile import unpickle, exists_or_download, create_dir
#image utils
from PIL import Image
from matplotlib.pyplot import imshow, imsave
from lib.data_utils import convert_img, convert_img_back, processing_img

LINK_CHIFAR10="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
LINK_CELEB_A="0B7EVK8r0v71pZjFTYXZWM3FlRnM"
LINK_BEDROOM = "http://dl.yf.io/lsun/scenes/{}_{}_lmdb.zip".format('bedroom', 'train')

def toy_dataset_stdev(DATASET='8gaussians'):
    if DATASET == '25gaussians':
        return 2.828 #0.05
    elif DATASET == 'swissroll':
        return 7.5
    elif DATASET == '8gaussians':
        return 1.414 #0.02
    else:
        return None

def toy_dataset(DATASET='8gaussians', size=256):

    if DATASET == '25gaussians':
        dataset = []
        for i in range(int(size/25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        # while True:
        #    for i in range(len(dataset)/BATCH_SIZE):
        #        yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':
        data = sklearn.datasets.make_swiss_roll(
            n_samples=size,
            noise=0.25
        )[0]
        dataset = data.astype('float32')[:, [0, 2]]
        dataset /= 7.5  # stdev plus a little

    elif DATASET == '8gaussians':
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x, scale*y) for x, y in centers]
        dataset = []
        for i in range(size):
            point = np.random.randn(2)*.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev

    return dataset

def cifar10():
    def load_data(dbpath):
        xs = []
        ys = []
        for j in range(5):
            batch = "data_batch_"+str(j+1)
            fullpath = os.path.join(dbpath,batch)
            d = unpickle(fullpath)
            x = d['data']
            y = d['labels']
            xs.append(x)
            ys.append(y)

        batch = "test_batch"
        fullpath = os.path.join(dbpath,batch)
        d = unpickle(fullpath)
        xs.append(d['data'])
        ys.append(d['labels'])

        x = np.concatenate(xs)/np.float32(127.5)
        y = np.concatenate(ys)
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

        # subtract per-pixel mean      0 - 255 => -1 - 1
        #pixel_mean = np.mean(x[0:50000],axis=0)
        #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
        #x -= pixel_mean
        x -= 1. 

        # create mirrored images
        X_train = x[0:50000,:,:,:]
        Y_train = y[0:50000]
        X_train_flip = X_train[:,:,:,::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train,X_train_flip),axis=0)
        Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

        X_test = x[50000:,:,:,:]
        Y_test = y[50000:]

        return dict(X_train=torch.FloatTensor(X_train),
                    Y_train=Y_train.astype('int32'), 
                    X_test=torch.FloatTensor(X_test),
                    Y_test =Y_test.astype('int32'),)
    create_dir(config.SAVE)
    exists_or_download(os.path.join(config.SAVE,"cifar-10-python.tar.gz"), LINK_CHIFAR10, unpack=True)
    return load_data(os.path.join(config.SAVE,"cifar-10-batches-py"))

def faces(img_size=64, 
          start=0, 
          stop=202599):

    img_shape=[3, img_size, img_size]
    def load_data():
        C, H, W = img_shape
        pathfiles = os.path.join(config.SAVE,"img_align_celeba")
        files_name = os.listdir(pathfiles)

        X = np.zeros(((stop-start), C, H, W), dtype='uint8')
        for i in range(start, stop):
            image_path = os.path.join(pathfiles, files_name[i])
            if not os.path.exists(image_path):
                print("warning, ",image_path)
                continue

            z = 0
            img = Image.open(image_path)

            if img.size[1] != H or img.size[0] != W:
                img = img.resize((W, H), Image.ANTIALIAS)

            img = np.array(img)
            if len(np.shape(img)) < C:
                img = [img]*C
                img = np.array(img)
                img = img.reshape([C, H, W*2])
                img = convert_img_back(img)

            #scale [0,1], center [1,-1], HWC to CHW
            img = processing_img(img, center=False, scale=False, convert=True)

            X[(i-start), :, :, :] = img
        return X, files_name[start:stop]
    
    #where is the db
    dbpath = os.path.join(config.SAVE,"img_align_celeba_{}.hdf5".format(img_size))
    #generate
    if not os.path.isfile(dbpath):
        create_dir(config.SAVE)
        print("Download faces dataset")
        exists_or_download(os.path.join(config.SAVE,"img_align_celeba.zip"), LINK_CELEB_A, unpack=True, google_drive=True)
        print("Create faces dataset")
        X, fnames = load_data()
        print("Save faces dataset")
        dbfile = h5py.File(dbpath,'w')
        dbfile.create_dataset("data",data=X)
    else:
        dbfile = h5py.File(dbpath,'r')
        X = dbfile.get("data").value
    return X

def bedrooms(img_size=128, 
             start=0, 
             stop=202599):

    img_shape=[3, img_size, img_size]
    def load_data():
        C, H, W = img_shape
        pathfiles = os.path.join(config.SAVE,"bedroom_train_lmdb/")
        env = lmdb.open(pathfiles, map_size=1099511627776, max_readers=100, readonly=True)
        X = np.zeros(((stop-start), C, H, W), dtype='uint8')
        i = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for bkey, val in cursor:
                if stop <= i:
                    break
                if i < start:
                    i += 1
                    continue

                img = Image.open(io.BytesIO(val))
                if img.size[1] != H or img.size[0] != W:
                    img = img.resize((W, H), Image.ANTIALIAS)
                #scale [0,1], center [1,-1], HWC to CHW
                img = processing_img(img, center=False, scale=False, convert=True)
                #save
                X[i, :, :, :] = img
                i += 1
        return X
    
    #where is the db
    dbpath = os.path.join(config.SAVE,"bedroom_train_{}.hdf5".format(img_size))
    #generate
    if not os.path.isfile(dbpath):
        create_dir(config.SAVE)
        print("Download bedrooms dataset")
        exists_or_download(os.path.join(config.SAVE,"bedroom_train_lmdb.zip"), LINK_BEDROOM, unpack=True)
        print("Create bedrooms dataset")
        X = load_data()
        print("Save bedrooms dataset")
        dbfile = h5py.File(dbpath,'w')
        dbfile.create_dataset("data",data=X)
    else:
        dbfile = h5py.File(dbpath,'r')
        X = dbfile.get("data").value
    return X
