from keras.datasets import cifar10
def get_cifar10_labels():
    return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  
def get_cifar10_data(preprocess=False):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = len(np.unique(y_train))
    
    class_names = get_cifar10_labels()
    fig = plt.figure(figsize=(8,3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        features_idx = X_train[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = features_idx[img_num]
        ax.set_title(class_names[i])
        plt.imshow(im)
    plt.show()
    if preprocess:
      X_train = X_train.astype('float32')
      X_test = X_test.astype('float32')
      X_train /= 255
      X_test /= 255
      # convert class labels to binary class labels
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    print ('Classes:', num_classes)
    print ('Shape:', X_train.shape)
    return num_classes, X_train, Y_train, X_test, Y_test
