from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from utility import get_parent_function_name
import os

gf_dim = 64     #filters number of first layer in G
df_dim = 64     #filters number of first layer in D
gfc_dim = 1024 * 2  
dfc_dim = 1024  
img_dim = 28  

c_dim = 3  
y_dim = 1  
output_height = 64
output_width = 64

use_cudnn = True
if 'ce_mode' in os.environ:
    use_cudnn = False

def bn(x, name=None, act='relu'):   #batch_norm
    if name is None:
        name = get_parent_function_name()
    #return fluid.layers.leaky_relu(x)
    return fluid.layers.batch_norm(
        x,
        param_attr=name + '1',
        bias_attr=name + '2',
        moving_mean_name=name + '3',
        moving_variance_name=name + '4',
        name=name,
        act=act)


def conv(x, num_filters, name=None, act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.nets.simple_img_conv_pool(    
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        pool_stride=2,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        use_cudnn=use_cudnn,
        act=act)


def fc(x, num_filters, name=None, act=None): 
    if name is None:
        name = get_parent_function_name()
    return fluid.layers.fc(input=x,
                           size=num_filters,
                           act=act,
                           param_attr=name + 'w',
                           bias_attr=name + 'b')


def deconv(x,       
           num_filters,
           name=None,
           filter_size=5,
           stride=2,
           dilation=1,
           padding=2,
           output_size=None,
           act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.layers.conv2d_transpose(
        input=x,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        num_filters=num_filters,
        output_size=output_size,
        filter_size=filter_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        use_cudnn=use_cudnn,
        act=act)


def D(x):  
    x = fluid.layers.reshape(x=x, shape=[-1, 1, 28, 28])
    x = conv(x, df_dim, act='leaky_relu')
    x = bn(conv(x, df_dim * 2), act='leaky_relu')
    x = bn(fc(x, dfc_dim), act='leaky_relu')
    x = fc(x, 1, act='sigmoid')
    return x


def G(x): 
    x = bn(fc(x, gfc_dim))
    x = bn(fc(x, gf_dim * 2 * img_dim // 4 * img_dim // 4))
    x = fluid.layers.reshape(x, [-1, gf_dim * 2, img_dim // 4, img_dim // 4])
    x = deconv(x, gf_dim * 2, act='relu', output_size=[14, 14])
    x = deconv(x, 1, filter_size=5, padding=2, act='tanh', output_size=[28, 28])
    x = fluid.layers.reshape(x, shape=[-1, 28 * 28])
    return x
