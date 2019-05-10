# https://gluon-cv.mxnet.io/build/examples_classification/demo_cifar10.html
# Gluoncv Tutorials
# Image Classification
# 1. Getting Started with Pre-trained Model on CIFAR10
import matplotlib.pyplot as plt
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model

# download the example image
def download(url):
    im_fname = utils.download(url)
    img = image.imread(im_fname)  
#    plt.imshow(img.asnumpy())   # show the example image
#    plt.show()                  # show the example image
    return img

# transformations for the image
# make the image more "model-friendly", instead of "human-friendly"
def transf(img):
    ## resize and crop the image to 32x32
    ## transpose it to num_channels*height*width(used for mxnet but not tf)
    ## normalize with mean and standard deviation calculated across all CIFAR10 images
    transform_fn = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    img_T = transform_fn(img)
#     plt.imshow(nd.transpose(img_T, (1,2,0)).asnumpy())   # show the example image
#     plt.show()                                           # show the example image
    return img_T

# load a pre-trained model
# here we just load the pretrained resnet100_v1 for the CIFAR10 dataset
def load_model(model_input = 'cifar_resnet110_v1', classes_input = 10, pretrained_or_not = True):
    net = get_model(name=model_input, classes=classes_input, pretrained=pretrained_or_not)
    return net

# finally, prepare the image and feed it to the model
def main(pic_url):
    url = pic_url
    img = transf(download(url))
    net = load_model()
    pred = net(img.expand_dims(axis=0))
    ## class names defining
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ind = nd.argmax(pred, axis=1).astype('int')
    print('The input image is classified as [%s], with probability [%.3f].' % (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))

# run the program
if __name__  == '__main__':
# the example image url, it is a plane
    url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/plane-draw.jpeg'
    main(url)