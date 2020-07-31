import numpy as np
import paddle.fluid as fluid


data = np.ones([1]).astype('float32')
with fluid.dygraph.guard(fluid.core.CPUPlace()):
    w = fluid.dygraph.to_variable(data)
    w.stop_gradient=False
    loss = w * w
    print(loss) #=> name tmp_1, dtype: VarType.FP32 shape: [1L] lod: {}
                #=> dim: 1, layout: NCHW, dtype: float, data: [1]
    loss.backward()
    print(w.gradient()) # => [2.]