import paddle.fluid as fluid

#define operation
w = fluid.data(name='w', shape=[None,1], dtype='float32')
w.stop_gradient=False
loss = w * w
grad = fluid.gradients([loss], w)
#Define Exector
cpu = fluid.core.CPUPlace() 
exe = fluid.Executor(cpu) 
exe.run(fluid.default_startup_program()) 
#Prepare data
x = numpy.ones((1, 1))
x= x.astype('float32')
#Run computing
outs = exe.run(feed={'w':x},fetch_list=[loss, grad])
print('loss: {}, grad: {}'.format(outs[0][0], outs[1][0]))
