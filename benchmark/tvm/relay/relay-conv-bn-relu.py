from __future__ import absolute_import

import numpy as np
import tvm
import tvm.contrib.graph_runtime as runtime
from tvm import relay 


def batch_norm_infer(data,
                     gamma=None,
                     beta=None,
                     moving_mean=None,
                     moving_var=None,
                     **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data,
                               gamma=gamma,
                               beta=beta,
                               moving_mean=moving_mean,
                               moving_var=moving_var,
                               **kwargs)[0]


def conv2d(data, weight=None, **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)


def conv_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1),
               padding=(1, 1), epsilon=1e-5):
    conv = conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout='NCHW',
        name=name+'_conv')
    bn = batch_norm_infer(data=conv, epsilon=epsilon, name=name + '_bn')
    act = relay.nn.relu(data=bn)
    return act


if __name__ == "__main__":
    data_shape = (1, 3, 224, 224)
    kernel_shape = (32, 3, 3, 3)
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    act = conv_block(data, "graph", 32, strides=(2, 2))
    func = relay.Function(relay.ir_pass.free_vars(act), act)
    print(func)
    
    target = "cuda"
    ctx = tvm.context(target, 0)
    # data = np.random.uniform(-1, 1, size=data_shape).astype(dtype)
    # kernel = np.random.uniform(-1, 1, size=kernel_shape).astype(dtype)
    # intrp = relay.create_executor("graph", ctx=ctx, target=target)
    # op_res = intrp.evaluate(func)(data, kernel)
    # print(op_res)
    
    net = relay.ir_pass.infer_type(func)
    shape_dict = {
        v.name_hint : v.checked_type for v in net.params}
    # net.astext()
    # np.random.seed(0)
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, ctx=ctx)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(net, target, params=params)
    
    # print(graph)

    module = runtime.create(graph, lib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(-1, 1, size=data_shape)).astype(dtype))
    module.set_input('data', data_tvm)
    module.set_input(**params)

    module.run()
    # print(module.get_output(0))

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=30)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))