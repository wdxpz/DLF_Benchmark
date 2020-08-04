import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import Linear


train_data=np.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
y_true = np.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32')

with fluid.dygraph.guard(fluid.core.CPUPlace()):
    #define predict function 
    x = fluid.dygraph.to_variable(train_data)
    y = fluid.dygraph.to_variable(y_true)

    linear = Linear(1, 1)
    y_predict = linear(x) 
    #define optimizer
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=linear.parameters())
    #define loss function 
    cost = fluid.layers.square_error_cost(input=y_predict,label=y)
    avg_cost = fluid.layers.mean(cost)
    avg_cost.backward()
    
    sgd_optimizer.minimize(avg_cost)
    print(y_predict)
