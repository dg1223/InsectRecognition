## This script is modelled to run in iPyhton

import numpy as np
import matplotlib.pylab as plt
import PIL
import Image
import ImageOps as io
import scipy
import ndimage
import _imaging
import glob
import json
from pprint import pprint

### Perform contrast normalization on all the images if necessary
### Need to make it as a function which takes file path name, formatname, cutoff, targetpath, fmt 
##
##cd 'your file path name here'
##
##list_of_files = glob.glob('*.formatname')
##list_of_files = np.array(list_of_files)
##
##for x in range(shape(list_of_files)[0]):
##    image = Image.open(list_of_files[x])
##    image = io.autocontrast(image cutoff = 0.1) # or any other cutoff percentage
##    image.save('targetpath/' + list_of_files[x], 'fmt') # fmt = JPEG, PNG etc


# go to working directory

cd '/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set'

