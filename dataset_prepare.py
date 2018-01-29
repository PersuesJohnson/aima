import numpy
import pickle
import os
import cv2
import time


def file_loader(img_add='./data', numpy_add='./numpydata', data_file='data.txt'):
    path_dir = os.listdir(img_add)
    n_count = 0
    mean_value = numpy.zeros(3)
    mean_v_temp = numpy.zeros(3)
    dirs_temp = os.listdir('.')
    writeTxt = True
    if 'data.txt' in dirs_temp:
        writeTxt = False
    if 'mean_value.npy' in dirs_temp:
        mean_value = numpy.load('mean_value.npy')
        standard = numpy.load('standard.npy')
        return mean_value,standard
    for dirs in path_dir:
        n_count += 1
        child = os.path.join('%s/%s' % (img_add, dirs))
        origin_dir = dirs.split('.')
        npy_file = origin_dir[0] + '_' + origin_dir[1]
        print('Processing file ' + child + ', this is file ' + str(n_count))
        img = cv2.imread(child)
        img = cv2.resize(img,(224,224))
        img = img.astype('float')
        img = numpy.transpose(img,(2,0,1))
        numpy_dir = numpy_add + '/' + npy_file + '.npy'
        #for i in range(3):
        #    mean_value[i] += numpy.mean(img[:,:,i])
        numpy.save(numpy_dir, img)
        if writeTxt:
            with open(data_file, 'a') as file:
                file.write(numpy_dir + ' ')
                file.write('0\n')
    mean_value = numpy.array([0.485, 0.456, 0.406])
    standard = numpy.array([0.229, 0.224, 0.225])
    numpy.save('standard.npy',standard)
    numpy.save('mean_value.npy',mean_value)
    return mean_value,standard

def value_preprocess(img_add='./data', numpy_add='./numpydata', mean_value=numpy.array([0,0,0]), standard=numpy.array([1,1,1])):
    path_dir = os.listdir(img_add)
    n_count = 0
    for dirs in path_dir:
        n_count += 1
        child = os.path.join('%s/%s' % (img_add, dirs))
        origin_dir = dirs.split('.')
        npy_file = origin_dir[0] + '_' + origin_dir[1]
        numpy_dir = numpy_add + '/' + npy_file + '.npy'
        img = numpy.load(numpy_dir)
        img = img.astype('float')
        img = img / 256
        for i in range(3):
            img[i] = img[i] - mean_value[i]
            img[i] = img[i] / standard[i]
        numpy.save(numpy_dir, img)
        print('Processing file ' + str(n_count) + ' for mean value...')


if __name__ == '__main__':
    mean_value, standard = file_loader()
    time.sleep(1)
    print('Max value for each channel is:',mean_value)
    time.sleep(1)
    value_preprocess(mean_value = mean_value,standard = standard)
    print('Done.')
    
