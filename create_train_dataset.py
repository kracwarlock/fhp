from fivehundredpx.client import FiveHundredPXAPI
from os import environ
import numpy as np
import csv 

CONSUMER_KEY = environ['PX_CONSUMER_KEY']
api = FiveHundredPXAPI()

feature_list = ['popular','highest_rated','upcoming','editors','fresh_today','fresh_yesterday','fresh_week']
maxlimit = 1000

allurls    = np.chararray((maxlimit*len(feature_list),),itemsize=300)
allratings = np.zeros((maxlimit*len(feature_list),))

for idx, feature_type in enumerate(feature_list):
    photos = api.photos(consumer_key=CONSUMER_KEY,
                    feature=feature_type,
                    image_size=21,
                    sort_direction='asc',
                    page=1,
                    rpp=maxlimit)

    urls    = map(lambda x: x['image_url'], photos['photos'])
    ratings = map(lambda x: x['highest_rating'], photos['photos'])
    allurls[idx*maxlimit:(idx+1)*maxlimit]    = urls
    allratings[idx*maxlimit:(idx+1)*maxlimit] = ratings

_, idx = np.unique(allurls, return_index=True)

img = allurls[idx]
rat = allratings[idx]
sz  = len(img)

p = np.random.permutation(len(img))
img = img[p]
rat = rat[p]
split = ['valid','train']
trl = np.zeros((len(img),))
trl[:0.9*len(img)] = 1
t = np.random.permutation(len(img))
trl = trl[t]

#img_url,label,_split
with open('/ais/gobi3/u/shikhar/fhp/dataset.csv','wb') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(["image_url","label","_split"])
    for i in xrange(len(img)):
        writer.writerow([img[i],"{0:.3f}".format(rat[i]/100.0),split[int(trl[i])]])
