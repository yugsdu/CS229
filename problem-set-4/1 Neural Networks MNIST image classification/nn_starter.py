import numpy as np
import matplotlib.pyplot as plt

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE
    # x: BXK, batchsizeXnumber of classes
    classes = np.max(x, axis=1, keepdims=True)
    num = np.exp(x-classes) #numerical stability
    deno = np.sum(num, axis=1,keepdims=True)
    s = num / deno
    ### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    x = x.astype(np.float32)
    s = 1/(1+np.exp(-x))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    data: BX784
    labels: BX10
    """
    W1 = params['W1'] # 784XH
    b1 = params['b1'] # BXH
    W2 = params['W2'] # HX10
    b2 = params['b2'] # BX10

    ### YOUR CODE HERE
    z1 = np.dot(data,W1)+b1
    h = sigmoid(z1)
    z2 = np.dot(h,W2)+b2
    y = softmax(z2)
    
    loss = -np.multiply(labels, np.log(y+1e-12)).sum()
    loss /= data.shape[0]
    ### END YOUR CODE
    return h, y, cost

def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h, y, cost = forward_prop(data, labels, params)
    
    delta1 = y-labels
    gradW2 = np.dot(h.T, delta1)
    gradb2 = np.sum(delta1, axis=0,keepdims=True)
    
    delta2 = np.dot(delta1,W2.T)*h*(1-h)#multiply
    gradW1 = np.dot(data.T, delta2)
    gradb1 = np.sum(delta2, axis=0,keepdims=True)
    
    # consider regularization
    lam = params['lambda']
    gradW2 += lam*W2
    gradW1 += lam*W1
    
    B = data.shape[0]
    gradW2 /= B
    gradb2 /= B 
    gradW1 /= B 
    gradb1 /= B 
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

def gradient_descent(params, grad, learning_rate):
    params['W1'] -= learning_rate*grad['W1']
    params['b1'] -= learning_rate*grad['b1']
    params['W2'] -= learning_rate*grad['W2']
    params['b2'] -= learning_rate*grad['b2']
    
def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    # add extra parameter options
    batch_size = 1000
    num_epochs = 30
    reg_strength=0
    params = {}

    ### YOUR CODE HERE
    K = trainLabels.shape[1] 
    H = num_hidden 
    B = batch_size
    
    params['W1'] = np.random.normal((n,H))
    params['b1'] = np.zeros((1,H),dtype=float)
    params['W2'] = np.random.normal((H,K))
    params['b2'] = np.zeros((1,K),dtype=float)
    params['lambda'] = reg_strength
    
    num_iter = int(m/B) # number of iterations per epoch
    train_loss, train_acc, dev_loss,dev_acc = [],[],[],[]
    for i in range(num_epochs):
        print('epoch = %d\n',i)
        for j in range(num_iter):
            batch_data = trainData[j*B:(j+1)*B]
            batch_labels = trainLabels[j*B:(j+1)*B]
            grad = backward_prop(batch_data,batch_labels,params)
            gradient_descent(params,grad,learning_rate)
        # calculate accuray per epoch
        _, y, cost = forward_prop(trainData,trainLabels,params)
        train_loss.append(cost)
        train_acc.append(compute_accuracy(y, trainLabels))
        _, y, cost = forward_prop(devData,devLabels,params)
        dev_loss.append(cost)
        dev_acc.append(compute_accuracy(y, devLabels))
        
        ### END YOUR CODE

    return params, train_loss, train_acc, dev_loss, dev_acc

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def prepare_data()
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    return trainData, trainLabels, devData, devLabels, testData, testLabels
    
def main():
    np.random.seed(100)
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    
    params, train_loss, train_acc, dev_loss, dev_acc\
    = nn_train(trainData, trainLabels, devData, devLabels)


    readyForTesting = False
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
    print 'Test accuracy: %f' % accuracy

if __name__ == '__main__':
    main()
