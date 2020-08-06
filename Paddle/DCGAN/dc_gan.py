# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import argparse
import functools
import matplotlib
import six
import numpy as np
import paddle
import time
import requests
import paddle.fluid as fluid
from utility import get_parent_function_name, plot, check, add_arguments, print_arguments
from network import G, D
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil
from reader import reader_creator, batch


NOISE_SIZE = 100
LEARNING_RATE = 2e-4
IMG_SIZE = 64
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   128,          "Minibatch size.")
add_arg('epoch',             int,   1,        "The number of epoched to be trained.") #20
add_arg('output',            str,   "./output_dcgan", "The directory the model and the test result to be saved to.")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
add_arg('run_ce',           bool,  False,       "Whether to ce.")
# yapf: enable


def loss(x, label):
    return fluid.layers.mean(
        fluid.layers.sigmoid_cross_entropy_with_logits(
            x=x, label=label))


def train(args):

    d_program = fluid.Program()
    dg_program = fluid.Program()
    
    #image = fluid.layers.data(name='image', shape=[3, 32, 32], dtype='float32')
    
    with fluid.program_guard(d_program):
        img = fluid.layers.data(name='img', shape=[IMG_SIZE*IMG_SIZE], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='float32')
        
        d_logit = D(img) 
        ################################################################
        d_loss = loss(d_logit, label) 
        ################################################################
    with fluid.program_guard(dg_program):
        noise = fluid.layers.data(
            name='noise', shape=[NOISE_SIZE, 1, 1], dtype='float32')
        g_img = G(x=noise)  #generate fake image

        g_program = dg_program.clone()
        g_program_test = dg_program.clone(for_test=True)

        dg_logit = D(g_img)  
        ################################################################
        dg_loss = loss(
            dg_logit,
            fluid.layers.fill_constant_batch_size_like(
                input=noise, dtype='float32', shape=[-1, 1], value=1.0))  
        ################################################################
    opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE, beta1=0.5)

    opt.minimize(loss=d_loss) 
    parameters = [p.name for p in g_program.global_block().all_parameters()]

    opt.minimize(loss=dg_loss, parameter_list=parameters) 

    exe = fluid.Executor(fluid.CPUPlace())
    if args.use_gpu:
        exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(fluid.default_startup_program())


    #download data
    data_dir = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data_dir = os.path.join(data_dir, 'data65')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print("start to download image data...")
    image_filename = os.path.join(BASE_DIR, 'data/data65/train-images-idx3-ubyte.gz')
    if not os.path.exists(image_filename):
        r = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        with open(image_filename, 'wb') as f:
            f.write(r.content)
    print("start to downlaod image data...")
    label_filename = os.path.join(BASE_DIR, 'data/data65/train-labels-idx1-ubyte.gz')
    if not os.path.exists(label_filename):
        r = requests.get('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        with open(label_filename, 'wb') as f:
            f.write(r.content)
    print("finished download")

    train_reader = reader_creator(image_filename, label_filename, 500)
    train_reader = batch(train_reader, args.batch_size, drop_last=False)
    NUM_TRAIN_TIMES_OF_DG = 2
    const_n = np.random.uniform(
        low=-1.0, high=1.0,
        size=[args.batch_size, NOISE_SIZE, 1, 1]).astype('float32')

    t_time = 0
    losses = [[], []]
    for pass_id in range(args.epoch):
        for batch_id, data in enumerate(train_reader()):
            if len(data) != args.batch_size:
                continue
            noise_data = np.random.uniform(
                low=-1.0, high=1.0,
                size=[args.batch_size, NOISE_SIZE, 1, 1]).astype('float32')
            real_image = np.array(list(map(lambda x: x[0], data))).reshape(
                -1, IMG_SIZE*IMG_SIZE).astype('float32')
            real_labels = np.ones(
                shape=[real_image.shape[0], 1], dtype='float32') 
            fake_labels = np.zeros(
                shape=[real_image.shape[0], 1], dtype='float32')  
            total_label = np.concatenate([real_labels, fake_labels])
            s_time = time.time()
            generated_image = exe.run(g_program,
                                      feed={'noise': noise_data},
                                      fetch_list=[g_img])[0]  #generate fake image

            total_images = np.concatenate([real_image, generated_image])

            d_loss_1 = exe.run(d_program,
                               feed={
                                   'img': generated_image,
                                   'label': fake_labels,
                               },
                               fetch_list=[d_loss])[0][0]  #tell fake images

            d_loss_2 = exe.run(d_program,
                               feed={
                                   'img': real_image,
                                   'label': real_labels,
                               },
                               fetch_list=[d_loss])[0][0]#tell real image
            ##############two losses above to trian D
            d_loss_n = d_loss_1 + d_loss_2
            losses[0].append(d_loss_n)
            for _ in six.moves.xrange(NUM_TRAIN_TIMES_OF_DG):
                noise_data = np.random.uniform(
                    low=-1.0, high=1.0,
                    size=[args.batch_size, NOISE_SIZE, 1, 1]).astype('float32')
                dg_loss_n = exe.run(dg_program,
                                     feed={'noise': noise_data},
                                     fetch_list=[dg_loss])[0][0] ### G_loss
                losses[1].append(dg_loss_n)
            t_time += (time.time() - s_time)
            if batch_id % 200 == 0 and not args.run_ce:
                # if not os.path.exists(args.output):
                #     os.makedirs(args.output)
                # generate image each batch
                generated_images = exe.run(g_program_test,
                                           feed={'noise': const_n},
                                           fetch_list=[g_img])[0]
                total_images = np.concatenate([real_image, generated_images])
                fig = plot(total_images)
                msg = "Epoch ID={0} Batch ID={1} D-Loss={2} DG-Loss={3}\n gen={4}".format(
                    pass_id, batch_id,
                    d_loss_n, dg_loss_n, check(generated_images))
                print(msg)
                #plt.title(msg)
                #plt.savefig(
                #    os.path.join(BASE_DIR, 'result/dcgan/{:04d}_{:04d}.png'.format(pass_id,
                #                                  batch_id)),
                #    bbox_inches='tight')
                #plt.close(fig)
                
        #save_path = os.path.join(BASE_DIR, 'model/dcgan_model')
        # delete old model file
        #shutil.rmtree(save_path, ignore_errors=True)
        #os.makedirs(save_path)
        # save prediction model
        #fluid.io.save_inference_model(main_program=g_program_test, dirname=save_path, feeded_var_names=['noise'], target_vars=g_img, executor=exe)

    if args.run_ce:
        print("kpis,dcgan_d_train_cost,{}".format(np.mean(losses[0])))
        print("kpis,dcgan_g_train_cost,{}".format(np.mean(losses[1])))
        print("kpis,dcgan_duration,{}".format(t_time / args.epoch))

    result_dir = os.path.join(BASE_DIR, 'result')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    RESULT_FILE = os.path.join(result_dir, 'results_dcgan.txt')
    with open(RESULT_FILE, 'a') as f:
        f.write('\n\n\n\ dcgan results:\n')
        f.write("kpis,dcgan_d_train_cost,{}".format(np.mean(losses[0])))
        f.write("kpis,dcgan_g_train_cost,{}".format(np.mean(losses[1])))
        f.write("kpis,dcgan_duration,{}".format(t_time))

    index = 0
    while os.path.exists(os.path.join(result_dir, 'minst_dcgan_tf_epoch_{}_{}.png'.format(args.epoch, index))):
        index += 1
    imgname = os.path.join(result_dir,  'minst_dcgan_tf_epoch_{}_{}.png'.format(args.epoch, index))
    generated_images = exe.run(g_program_test,
                                feed={'noise': const_n},
                                fetch_list=[g_img])[0]
    total_images = np.concatenate([real_image, generated_images])
    fig = plot(total_images)
    plt.savefig(
        imgname,
        bbox_inches='tight')
    plt.close(fig)
 

if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    train(args)
