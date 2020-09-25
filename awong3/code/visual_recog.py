import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from os.path import join
from opts import get_opts

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    # Instantiate dictionary
    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary) 

    # Histogram computation
    hist = np.histogram(wordmap, dict_size) # returns 2 values: values appearance and bins
    hist = np.array(hist[0])

    # normalize the histogram
    hist = hist/np.sum(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    #  Load dictionary
    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary) 

    # Instantiate variables
    h, w = wordmap.shape[0], wordmap.shape[1]
    hist_all = []
    cell_hist = []
    weight = []

    for layer in range(L-1, -1, -1): 
        cell_h = math.ceil(h/(2**layer)) # cell height
        cell_w  = math.ceil(w/(2**layer)) # cell width
        cell_num = 2**layer
        hist_img = []
        x, y = 0, 0
        vertical = np.array_split(wordmap, cell_num, axis=0)
        for i in vertical:
            x = 0
            horizontal = np.array_split(i, cell_num, axis=1) 
            for j in horizontal:
                if layer > 1:
                    weight = 2**(layer-L-1)
                else:
                    weight = 2**(-L)
                cell_hist = get_feature_from_wordmap(opts,wordmap[x:x+cell_h, y:y+cell_w])
                hist_img = np.append(hist_img, weight*cell_hist)
                x += cell_h
            y += cell_w
        hist_img = np.reshape(hist_img, (dict_size, cell_num, cell_num))
        hist_img = hist_img/np.sum(hist_img)
        hist_all = np.append(hist_all, hist_img)
    
    hist_all = hist_all/np.sum(hist_all)
    return hist_all

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    img = Image.open("../data/"+ (img_path)) # load image
    wordmap = visual_words.get_visual_words(opts,img,dictionary) # find the wordmap for the image
    feature = get_feature_from_wordmap_SPM(opts,wordmap) # compute SPM

    return feature

def build_recognition_system(opts, n_worker=4):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    print('in build recognition files')
    # Instantiate variables
    train_files_num = len(train_files) # number of training data
    features = []
    args = []

    for filename in range(train_files_num):
        args.append((opts,train_files[filename], dictionary))

    with multiprocessing.Pool(n_worker) as p:
        features = p.starmap(get_image_feature, args)
    features = np.array(features)

    # save learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    # minimum histogram similarity (at intersection) and sum because same number of bins
    sim = np.sum(np.minimum(word_hist,histograms),axis=1) 

    return sim

def evaluate_recognition_system(opts, n_worker=8):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    print('in evaluation')
    # Instantiate variables
    train_features = trained_system['features']
    train_labels = trained_system['labels']
    test_file_num = len(test_files)
    conf = np.zeros((8,8))
    predicted_labels = []
    wrong_labels = []

    for filename in range(test_file_num):
        test_img = Image.open("../data/"+ (test_files[filename])) # load test images
        test_wordmap = visual_words.get_visual_words(opts,test_img,dictionary) # compute wordmap
        test_features = get_feature_from_wordmap_SPM(opts,test_wordmap) # compute SPM
        predicted_feature = np.argmax(distance_to_set(test_features, train_features)) # predicted features 
        predicted_label = train_labels[predicted_feature] # predicted label per predicted feature 
        predicted_labels.append(predicted_label) # all predicted labels
        label = test_labels[filename]
        conf[label,predicted_label] += 1
        if label != predicted_label: # append wrong labels
            wrong_labels = test_files[filename]
            # print(wrong_labels, label, predicted_label)
    accuracy = np.trace(conf)/np.sum(conf)

    return conf, accuracy