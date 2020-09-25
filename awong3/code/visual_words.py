import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color

from opts import get_opts
import sklearn
from multiprocessing import Pool
import sklearn.cluster
import matplotlib as plt


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    # Image properties 
    img = skimage.img_as_float32(img) # convert incoming image that are non floating point data type into floating point data type
    if len(img.shape) < 3: # Check image channel (No more than 3 channels in input images)
        img = np.stack((img,)*3,axis=-1) # stacks 3 HxW arrays into HxWx3 array
    img = skimage.color.rgb2lab(img) # Convert image into lab color image    

    # Filter properties
    scaleSize = len(filter_scales) # number of filter scales
    numFilter = 4 # number of filters
    filterBank = scaleSize*numFilter # filter bank 
    numChannels = 3 # number of channels in an image or video
    
    # Filter images
    filter_responses = np.empty([img.shape[0], img.shape[1], 3*filterBank]) # Instantiate empty filter result
    for scale in range(scaleSize):
        for channel in range(numChannels):
            gaussFilter = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma = filter_scales[scale]) # apply gaussian filter 
            laplaceGaussFilter = scipy.ndimage.gaussian_laplace(img[:,:,channel], sigma = filter_scales[scale]) # apply gaussian laplace filter 
            gaussXFilter = scipy.ndimage.gaussian_filter(img[:,:,channel],sigma = filter_scales[scale], order = (1,0)) #apply first order gaussian filter in x direction
            gaussYFilter = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma = filter_scales[scale], order = (0,1)) #apply first order gaussian filter in y direction
            
            # Save filter responses
            iteration = 3*scale*numFilter + channel # filter iteration
            filter_responses[:,:,iteration] = gaussFilter
            filter_responses[:,:,iteration + 3] = laplaceGaussFilter
            filter_responses[:,:,iteration + 6] = gaussXFilter
            filter_responses[:,:,iteration + 9] = gaussYFilter
    
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    # Instantiate arguments
    opts = get_opts()
    i, alpha, train_files = args    # inputs: i-index of training image, alpha-number of random pixel samples, and img_path-path of image
    alpha = int(alpha)

    # Instantiate images from image path
    img = Image.open("../data/"+train_files)
    img = np.array(img).astype(np.float32)/255  # Make image is a floating point format type

    filter_responses = extract_filter_responses(opts,img) #extract the responses
    randSampleX = np.random.choice(filter_responses.shape[0], alpha) # random x pixel sampling of alpha size
    randSampleY = np.random.choice(filter_responses.shape[1], alpha) # random y pixel sampling of alpha size
   
    filter_responses = filter_responses[randSampleX,randSampleY,:] #Extract the random pixels of size alpha*3*F
    np.save(os.path.join("../feat/", str(i)+'.npy'), filter_responses)

def compute_dictionary(opts, n_worker=8):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    print('in compute dictionary')
    print('L is', opts.L)
    print('K is', opts.K)
    print('alpha is', opts.alpha)

    # Instantiate variables
    alpha = opts.alpha
    train_data_num = len(train_files) # number of training data
    trainDataList = np.arange(train_data_num) # iterate training data 1 by 1
    alphaList = alpha*np.ones(train_data_num) # multiply each training data with some sort of random sampling alpha
    filter_responses = []
    args=[]

    # Instantiate multiprocessing
    with multiprocessing.Pool(n_worker) as p:
        args = list(zip(trainDataList, alphaList, train_files))
        p.map(compute_dictionary_one_image, args)

    for i in range(train_data_num):
        filter_responses.append(np.load("../feat/"+str(i)+'.npy')) 
    filter_responses = np.concatenate(filter_responses, axis=0)

    # Instantiate kmeans
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    
    # Save dictionary in a file
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_responses = extract_filter_responses(opts, img) # filtered response
    filter_responses_resize = filter_responses.reshape(filter_responses.shape[0]*filter_responses.shape[1], -1) 

    # Compute Eucliden distance of each pixel in wordmap to the closest visual word of the filter response
    distance = scipy.spatial.distance.cdist(filter_responses_resize, dictionary, 'euclidean') # distance between pixel and dictionary word
    wordmap = np.argmin(distance, axis = 1) # find the closest match in dictionary using euclidean distance (-1 for min idex)
    wordmap = wordmap.reshape(filter_responses.shape[0], filter_responses.shape[1])    
    
    return wordmap

