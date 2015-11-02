import caffe
import lmdb
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import numpy as np

lmdb_env = lmdb.open('_temp/features')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

prediction = []
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    for d in data:
            prediction.append(d[0][0])
prediction = np.array(prediction).reshape([len(prediction),])

lmdb_env = lmdb.open('_temp/label')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

labels = []
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    for d in data:
            labels.append(d[0][0])
labels = np.array(labels).reshape([len(labels),])

topfraction = 0.1
predtop = np.argsort(prediction)[::-1]
realtop = np.argsort(labels)[::-1]

predtopidx = predtop[:int(topfraction*len(prediction))]
realtopidx = realtop[:int(topfraction*len(labels))]
intersection = np.intersect1d(predtopidx,realtopidx)

print '----------------------------------------------------------'
print 'Predicting top',int(topfraction*len(prediction)),'images out of',len(prediction),'images'
print len(intersection),'images predicted correctly'
print 'Ratio:',len(intersection)*1.0/len(realtopidx)
print '----------------------------------------------------------'
