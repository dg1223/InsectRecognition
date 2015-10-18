import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import colorsys

from annotation import get_annotation, get_bbs
##from tools import dispims
from pprint import pprint

from scipy.misc import imresize

def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 = int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * np.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = np.vstack((
                            np.hstack((np.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*np.ones((height,border),dtype=float))),
                            bordercolor*np.ones((border,width+border),dtype=float)
                            ))
    plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest', **kwargs)
    plt.show()



plt.ion()

target_height = 32  # including margin
target_width = 32  # including margin
margin_height = 8
margin_width = 8

train_path = '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'
test_path = '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Test Set'

jpg_train = [f for f in os.listdir(train_path) if f.find('.jpg') > 0]

def crop_and_rescale(im, xy, width, height, target_width=32, target_height=32):
    yx = xy[::-1]
    shape = (height, width)
    small_dim = np.argmin(shape)  # pad smaller dimension
    large_dim = 1 - small_dim

    # pad up small dim so that we have square image
    pad_size = shape[large_dim] - shape[small_dim]

    pad_before = pad_size / 2
    pad_after = (pad_size / 2) + (pad_size % 2)  # extra goes at end

    small_bounds = (yx[small_dim] - pad_before,
                 yx[small_dim] + shape[small_dim] + pad_after)

    # bounds checking: did padding mean we exceed image dimensions?
    # if so, make window tight up against boundary
    if small_bounds[0] < 0:
        small_bounds = (0, shape[large_dim])

    if small_bounds[1] > im.shape[small_dim]:
        small_bounds = (im.shape[small_dim] - shape[large_dim],
                        im.shape[small_dim])

    # the min here is a fix for at least one of the annotations
    # which exceeds the image bounds
    large_bounds = (yx[large_dim], min(yx[large_dim] + shape[large_dim],
                                       im.shape[large_dim]))

    im_crop = np.take(np.take(im, np.arange(*small_bounds), axis=small_dim),
            np.arange(*large_bounds), axis=large_dim)

    im_resized = imresize(im_crop, (target_height, target_width))

    return im_resized

moths = []
moths_resized = []

for i, j in enumerate(jpg_train):
    try:
        im = plt.imread(os.path.join(train_path, j), 'r')
    except IOError:
        print "There was a problem reading the jpg: %s." % j
        continue

    # the rollaxis command rolls the last (-1) axis back until the start
    # do a colourspace conversion
    im_y, im_i, im_q = colorsys.rgb_to_yiq(*np.rollaxis(im[...,:3], axis=-1))
    ann_file = j.split('.')[0]
    ann_path = ann_path = os.path.join(train_path, ann_file)
    annotation = get_annotation(ann_path)

    # get all bbs for this image
    bbs = get_bbs(annotation)
    for xy, width, height in bbs:
        x, y = xy
        # remember y is indexed first in image
        moth = im_y[y:(y + height), x:(x + width)]
        moths.append(moth)
        #print moth.shape

        moth_resized = crop_and_rescale(im_y, xy, width, height,
                         target_width=32, target_height=32)    

        moths_resized.append(moth_resized)
        n_moths = len(moths_resized)

m = np.asarray(moths_resized).reshape((n_moths,
                                           target_height * target_width))
dispims(m.T,
        target_height, target_width, border=2)
