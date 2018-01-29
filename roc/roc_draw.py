import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###����roc��auc
from sklearn import cross_validation

# Import some data to play with
iris = datasets.load_iris()
#X = iris.data
#y = iris.target
X = np.load('train_pred.npy')[1:]
y = np.load('train_real.npy')[1:]
print(X)
print(y)
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y, X) ###���������ʺͼ�����
roc_auc = auc(fpr,tpr) ###����auc��ֵ

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###������Ϊ�����꣬������Ϊ������������
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
