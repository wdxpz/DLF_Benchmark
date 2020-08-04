import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import paddle.fluid as fluid
from utility import get_parent_function_name, plot, check, add_arguments, print_arguments
import functools
import numpy as np
import os

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('batch_size',        int,   9,          "Minibatch size.")
add_arg('output',            str,   "./output_dcgan", "The directory the model and the test result to be saved to.")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
NOISE_SIZE = 100


def infer():
   
    exe = fluid.Executor(fluid.CPUPlace())
    if args.use_gpu:
        exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(fluid.default_startup_program())
    
    noise_data = np.random.uniform(
                low=-1.0, high=1.0,
                size=[args.batch_size, NOISE_SIZE]).astype('float32')
                
    save_path = 'freeze_model'
    
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
    
    generated_image = exe.run(infer_program,
                                      feed={feeded_var_names[0]: noise_data},
                                      fetch_list=target_var)[0]
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    total_images = generated_image
    fig = plot(total_images)
    plt.savefig(
        '{}/generated_image.png'.format(args.output),
        bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    args = parser.parse_args()
    infer()