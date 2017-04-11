import sys
sys.path.insert(0,'../caffe-fr-chairs-deepsim/python')
import caffe
import numpy as np
import cv2
import scipy.io
import scipy.misc

nz = 100
img_size = 64
batch_size = 64

caffe.set_mode_gpu()
gen_net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
#data_loader=caffe.Net('data_gen.prototxt',sys.argv[2],caffe.TEST)
#data_loader=caffe.Net('data_test.prototxt',sys.argv[2],caffe.TEST)
data_loader=caffe.Net('data_dual.prototxt',sys.argv[2],caffe.TEST)
# Fix the seed to debug
#np.random.seed(0) 
#
#img = cv2.imread(sys.argv[3])
#img = np.array(img,dtype=np.float32)
#img = cv2.resize(img,(img_size,img_size))
#img -= np.array((104, 117, 123))
#img = img.transpose(2,0,1)
#img_batch = np.zeros((batch_size,3,img_size,img_size))
#for i in xrange(batch_size):
#    img_batch[i,:] = img[i]
#gen_net.blobs['feat'].data[...] = img_batch
data_loader.forward_simple()
img_batch = data_loader.blobs['data'].data

gen_net.blobs['feat'].data[...] =img_batch 
gen_net.forward_simple()

generated_img = gen_net.blobs['generated'].data

print generated_img.shape

max_val, min_val = np.max(generated_img[0]), np.min(generated_img[0])
max_val_, min_val_ = np.max(img_batch[0]), np.min(img_batch[0])


# Concat all images into a big 8*8 image
flatten_img = ((generated_img.transpose((0,2,3,1)))[:] - min_val) / (max_val-min_val)
flatten_img_batch = ((img_batch.transpose((0,2,3,1)))[:] - min_val_) / (max_val_-min_val_)
print flatten_img[0,:]
print flatten_img.shape
for i in xrange(batch_size):
    cv2.imwrite('test_result/'+str(i)+'.jpg',255*flatten_img[i,:])
    cv2.imwrite('test_result/'+str(i)+'_ori.jpg',255*flatten_img_batch[i,:])
    

#cv2.imwrite('test1.jpg', 255*flatten_img.reshape(8,8,img_size,img_size,3).swapaxes(1,2).reshape(8*img_size,8*img_size, 3))
#cv2.imwrite('test_ori.jpg', 255*flatten_img_batch.reshape(8,8,img_size,img_size,3).swapaxes(1,2).reshape(8*img_size,8*img_size, 3))

#cv2.imshow('test', ((generated_img.transpose((0,2,3,1)))[2] - min_val) / (max_val-min_val))
#cv2.waitKey()
