import h5py
import os
import numpy as np
import PIL.Image

DIR = '/ais/gobi3/u/shikhar/fhp/data'

text_fn = os.path.join(DIR, 'train.txt')
h5_fn   = os.path.join(DIR, 'trainrating.h5')

with open(text_fn, 'r') as f:
    lines = f.readlines()
    imgpaths = []
    ratings = []
    for line in lines:
        imgpaths.append(line.split(' ')[0])
        ratings.append(float(line.split(' ')[1]))
    ratings = np.array(ratings)*0.001

with h5py.File(h5_fn, 'w') as f:
    f['label'] = ratings

text_fn = os.path.join(DIR, 'trainrating.txt')
with open(text_fn, 'w') as f:
    print >>f,h5_fn

text_fn = os.path.join(DIR, 'valid.txt')
h5_fn   = os.path.join(DIR, 'validrating.h5')

with open(text_fn, 'r') as f:
    lines = f.readlines()
    imgpaths = []
    ratings = []
    for line in lines:
        imgpaths.append(line.split(' ')[0])
        ratings.append(float(line.split(' ')[1]))
    ratings = np.array(ratings)*0.001

with h5py.File(h5_fn, 'w') as f:
    f['label'] = ratings

text_fn = os.path.join(DIR, 'validrating.txt')
with open(text_fn, 'w') as f:
    print >>f,h5_fn

text_fn = os.path.join(DIR, 'test.txt')
h5_fn   = os.path.join(DIR, 'testrating.h5')

with open(text_fn, 'r') as f:
    lines = f.readlines()
    imgpaths = []
    ratings = []
    for line in lines:
        imgpaths.append(line.split(' ')[0])
        ratings.append(float(line.split(' ')[1]))
    ratings = np.array(ratings)*0.001

with h5py.File(h5_fn, 'w') as f:
    f['label'] = ratings

text_fn = os.path.join(DIR, 'testrating.txt')
with open(text_fn, 'w') as f:
    print >>f,h5_fn
