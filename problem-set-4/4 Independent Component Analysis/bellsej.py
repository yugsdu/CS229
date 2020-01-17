### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix # each column corresponding to one of the mixed signals xi

def play(vec):
    sd.play(vec, Fs, blocking=True)

# first define its cumulative distribution function/cdf
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def unmixer(X):
    M, N = X.shape#MxN
    W = np.eye(N) #NxN, memory error if eye(M)??

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    #update the parameter
    for alpha in anneal:
        for x in X:
            # compute array and array to get matrix
            # method 1: use np.mat() changing array to matrix
            #deltW = (1-2*sigmoid(np.dot(W, np.mat(x).T))).dot(np.mat(x))+\
            #np.linalg.inv(W.T)
            # method2: use np.outer() to get array, both of them are of shape (len,)
            deltW = np.outer(1-2*sigmoid(np.dot(W, x.T)),x)+np.linalg.inv(W.T) # np.dot((2,3),(3,))==(2,)
            W += alpha*deltW
        
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = X.dot(W.T)
    ##################################
    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i]) # each column corresponding to one of the mixed signals xi

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i]) # each column corresponding to one of the unmixed signals xi

if __name__ == '__main__':
    main()
