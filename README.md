## Prerequisites
 - Caffe
 - pycaffe
 - pandas
 - skimage
 - numpy
 - python-500px
 - lmdb

## Instructions
 - Change the filepaths in the following files according to you setup:
   - `create_train_dataset.py`
   - `create_test_dataset.py`
   - `assemble_train_data.py`
   - `assemble_test_data.py`
   - `ratings_to_hdf5.py`
   - `solver.prototxt`
   - `train_val.prototxt`
   - `test.prototxt`
   - `evaluate.py`

- Now run the scripts in this order

    ```
    $ python create_train_dataset.py
    ```

    ```
    $ python create_test_dataset.py
    ```

    ```
    $ python assemble_train_data.py
    ```

    ```
    $ python assemble_test_data.py
    ```
 
    ```
    $ python ratings_to_hdf5.py
    ```

  This will download and prepare your dataset.

- To fine-tune the pre-trained ConvNet go to your Caffe directory
  
    ```
    $ build/tools/caffe train -solver /<path-to-fhp>/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0
    ```
  
- To extract the predictions for the test dataset

    ```
    $ build/tools/extract_features /<path-to-fhp>/models/fhp/fhp_final_iter_3100.caffemodel /<path-to-fhp>/test.prototxt fc8_flickr /<path-to-fhp>/_temp/features 10 lmdb GPU 0
    ```

- To evaluate the model

    ``` 
    $ python evaluate.py
    ----------------------------------------------------------
    Predicting top 100 images out of 1000 images
    32 images predicted correctly
    Ratio: 0.32
    ----------------------------------------------------------
    ```

## Short Description
  This code does regression on the ratings users give to images. The convnet used here is the pretrained Caffenet
  which I finetune using the image ratings for the 500px data subset. 
  After predicting ratings on the test set the code selects the top-k images according to it out of the test set.
  We then see how many of these images are present in the top-k images voted by users. This metric is not a very
  good metric since the premise was that there are some images which are great but aren't rated by many users and 
  thus their rating isn't representative of their beauty. However, if we can learn to predict what ratings users
  give to images which have received above a certain threshold of number of votes then it is quite possible that
  the model understands the users' notion of beauty and can also rate those images accurately which did not get
  featured due to less votes.0.0231359

## Analysis
 1. The model's recall in the top-100 images was 0.32. Since beauty is difficult to model as a metric and is
    subjective I think this is reasonable. There was some overfitting as the dataset size was very small (~6k)
    and even after doing mirroring transformations it wasn't very big but the overfitting was not substantial.
    The training MSE was 0.0114491 while the validation MSE was 0.0231359.
 2. Top rated images according to my model:
    - https://drscdn.500px.org/photo/127163753/h%3D600_k%3D1_a%3D1/0db877b44a70b881689e305183a38ca7
    - https://drscdn.500px.org/photo/127163479/h%3D600_k%3D1_a%3D1/737e085594b902910bb1ae18ddd604e1
    - https://drscdn.500px.org/photo/127163191/h%3D600_k%3D1_a%3D1/45ea5f25dca1d54890227c0497696629
    - https://drscdn.500px.org/photo/127162263/h%3D600_k%3D1_a%3D1/a9fd301d3607b2fb74c90113171e585f
    - https://drscdn.500px.org/photo/127163285/h%3D600_k%3D1_a%3D1/6080d68e61cf0b0d32ce55d464a1c3d7
    - https://drscdn.500px.org/photo/127163605/h%3D600_k%3D1_a%3D1/53fde68398ffea07cc3cfbed24bfe2cf
    - https://drscdn.500px.org/photo/127161905/h%3D600_k%3D1_a%3D1/8576a11ba586c5cc831f779c4f117681
    Bottom rated images according to my model:
    - https://drscdn.500px.org/photo/127162071/h%3D600_k%3D1_a%3D1/265abe7df5ae9a473435172eac715d25
    - https://drscdn.500px.org/photo/127163617/h%3D600_k%3D1_a%3D1/f99e75682a6e5c39cb03eadc0b3a48a3
    - https://drscdn.500px.org/photo/127163043/h%3D600_k%3D1_a%3D1/3919b016369bc5596d9fd090eb737aeb
    - https://drscdn.500px.org/photo/127161265/h%3D600_k%3D1_a%3D1/0ecd6ffc49d2568eb4509f6f22e687f1
    - https://drscdn.500px.org/photo/127164305/h%3D600_k%3D1_a%3D1/77c56604d6f71eea79906c29f036004a
    - https://drscdn.500px.org/photo/127163687/h%3D600_k%3D1_a%3D1/8cd95945a8b36ef5ced30fda6f421928
    - https://drscdn.500px.org/photo/127163493/h%3D600_k%3D1_a%3D1/b565c1e757a3ccea17cace946e3246ab
    
   It's difficult to find a trend in these images but I think that in the images it rates high there is not a
   large variation in the colors present in the image (the range of colours in each image is small) and the images
   are bright. The images it dislikes also seem to be mostly dark or they have large dark regions.

## Code Attribution
This repository uses code from the 500px API sample and http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html
