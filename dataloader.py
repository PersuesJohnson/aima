import numpy
import csv
import torch.utils.data as data
import time

def dataloader_dl(cross = 10,num_of_cross = 0,file_list = 'data.txt'):
    dataset = []
    labels = []
    with open(file_list, 'r') as f:
        for lines in f.readlines():
            lines = lines.split(' ')
            dataset.append(lines[0])
            labels.append(lines[1])
    # dataset = dataset.astype('float32')
    dataset = numpy.array(dataset)
    label = numpy.array(labels)
    #print(type(dataset))
    #print(type(dataset[1]))
    #print(dataset[1])
    label_col = numpy.split(label,cross)
    dataset_col = numpy.split(dataset,cross)
    val_dataset = None
    val_label = None
    if num_of_cross >= 0:
        val_dataset = dataset_col[num_of_cross]
        val_label = label_col[num_of_cross]
    train_dataset = None
    train_label = None
    for i in range(cross):
        if i == num_of_cross:
            continue
        if train_dataset is None:
            train_dataset = dataset_col[i]
            train_label = label_col[i]
        else:
            train_dataset = numpy.concatenate((train_dataset,dataset_col[i]),axis = 0)
            train_label = numpy.concatenate((train_label,label_col[i]),axis = 0)
    #print('Data for No.',num_of_cross+1,'is ready.')
    #print('Size for training:',numpy.shape(train_dataset),'labels for training:',numpy.shape(train_label))
    #print('Size for validation:',numpy.shape(val_dataset),'labels for validation:',numpy.shape(val_label))
    return {'train_data':train_dataset,
            'train_label':train_label,
            'val_data':val_dataset,
            'val_label':val_label,
            'full_data':dataset,
            'full_label':label}

class img_Dataset(data.Dataset):
    def __init__(self,cross = 10,num_of_cross = 0,Training = True):
        self.cross = cross
        self.num_of_cross = num_of_cross
        self.training = Training
        self.origin_data = dataloader_dl(cross,num_of_cross)
        
    def __getitem__(self,index):
        if self.training:
            #print(self.origin_data['train_data'])
            training_data = self.loader(self.origin_data['train_data'][index]).astype('float32')
            return training_data,self.origin_data['train_label'][index].astype('int64')
        else:
            eval_data = self.loader(self.origin_data['val_data'][index]).astype('float32')
            txt_n = []
            otd = self.origin_data['val_data'][index].split('.')
            #print(otd)
            fxk = otd[1].split('/')
            fxk = fxk[-1]
            #print(fxk)
            return eval_data,self.origin_data['val_label'][index].astype('int64'),int(fxk)
            
    def loader(self,filepath):
        return numpy.load(filepath)
            
    def __len__(self):
        if self.training:
            return len(self.origin_data['train_label'])    
        else:
            return len(self.origin_data['val_label']) 
        
if __name__ == '__main__':
    dataloader_dl()
