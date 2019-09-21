import tvm  
import nnvm 
from nnvm import symbol as sym 

"""
This implementation of MobileNet V2 is modified 
from https://github.com/tonylins/pytorch-mobilenet-v2
"""


def generate_parameters(obj):
    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            if isinstance(value, sym.Symbol):
                yield value
            elif isinstance(value, Layer):
                for ret in generate_parameters(value):
                    yield ret
            elif isinstance(value, (list, tuple)):
                for ele in value:
                    for ret in generate_parameters(ele):
                        yield ret
            elif isinstance(value, dict):
                for k, v in value.items():
                    for ret in generate_parameters(v):
                        yield ret


class Layer(object):
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return generate_parameters(self)


def check_array(data, whole_type, inner_type, length):
    assert isinstance(length, int)
    assert isinstance(data, whole_type)
    assert len(data) == length
    for ele in data:
        assert isinstance(ele, inner_type)

    
def compose(f, *args):
    if len(args) == 0:
        return f 
    return compose(lambda *a: args[0](f(*a)), *args[1:])


def identity(*args):
    return args


class Sequential(Layer):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._call_list = args

    def forward(self, inputs):
        # return compose(identity, *self._call_list)(*inputs)
        if len(self._call_list) == 0:
            return inputs 
        else:
            ret = self._call_list[0](inputs)
            for f in self._call_list[1:]:
                ret = f(ret)
            return ret


class Conv2d(Layer):
    def __init__(self, name, in_channel, out_channel, kernel_size, strides=1, padding=1, dilation=1, group=1, use_bias=False):
        if isinstance(strides, int):
            strides = (strides, strides) 
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        assert isinstance(group, int) 
        assert in_channel % group == 0 and out_channel % group == 0
        check_array(strides, tuple, int, 2)
        check_array(kernel_size, tuple, int, 2)
        check_array(padding, tuple, int, 2)
        check_array(dilation, tuple, int, 2)

        self.name = name
        self._in_channel = in_channel
        self._out_channel = out_channel
        self._kernel_size = kernel_size
        self._padding = padding 
        self._strides = strides
        self._dilation = dilation 
        self._group = group 
        self._use_bias = use_bias 

        self.weight = sym.Variable("%s_weight" % name, shape=[out_channel, in_channel, *kernel_size])
        if use_bias:
            self.bias = sym.Variable("%s_bias" % name, shape=[out_channel])

    def forward(self, inputs):
        if self._use_bias:
            return sym.conv2d(data=inputs, weight=self.weight, bias=self.bias, channels=self._out_channel, 
                    kernel_size=self._kernel_size, padding=self._padding, strides=self._strides, 
                    dilation=self._dilation, groups=self._group, use_bias=self._use_bias)
        else:
            return sym.conv2d(data=inputs, weight=self.weight, channels=self._out_channel, 
                    kernel_size=self._kernel_size, padding=self._padding, strides=self._strides, 
                    dilation=self._dilation, groups=self._group, use_bias=self._use_bias)


class BatchNorm(Layer):
    def __init__(self):
        super(BatchNorm, self).__init__()

    def forward(self, inputs):
        return sym.batch_norm(inputs)


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inputs):
        return sym.relu(inputs)


class Dropout(Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self._rate = rate

    def forward(self, inputs):
        return sym.dropout(data=inputs, rate=self._rate)


class Dense(Layer):
    def __init__(self, name, in_dim, units, use_bias=False):
        super(Dense, self).__init__()
        self.name = name
        self._units = units
        self.weight = sym.Variable("%s_weight" % name, shape=[units, in_dim])
        if use_bias:
            self.bias = sym.Variable("%s_bias" % name, shape=[units])
        self._use_bias = use_bias

    def forward(self, inputs):
        if self._use_bias:
            return sym.dense(data=inputs, weight=self.weight, bias=self.bias, units=self._units)
        else:
            return sym.dense(data=inputs, weight=self.weight, units=self._units)


def conv_bn(name, in_channel, out_channel, kernel_size, strides=1, padding=1, dilation=1, group=1, use_bias=False):
    return Sequential(
        Conv2d(name, in_channel, out_channel, kernel_size, strides, padding, dilation, group, use_bias),
        BatchNorm(),
        ReLU()
    )


def conv_3x3_bn(name, in_channel, out_channel, strides):
    return conv_bn(name, in_channel, out_channel, 3, strides)


def conv_1x1_bn(name, in_channel, out_channel):
    return conv_bn(name, in_channel, out_channel, 1, padding=0)


class InvertedResidual(Layer):
    def __init__(self, name, in_channel, out_channel, strides, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.strides = strides 
        assert strides in [1, 2]

        hidden_dim = round(in_channel * expand_ratio)
        self.use_res_connect = self.strides == 1 and in_channel == out_channel

        if expand_ratio == 1:
            self.conv = Sequential(
                # depthwise
                Conv2d("%s_depthwise_1" % name, in_channel, hidden_dim, 3, strides=strides, padding=1, group=hidden_dim),
                BatchNorm(),
                ReLU(),
                # pointwise
                Conv2d("%s_pointwise_1" % name, hidden_dim, out_channel, 1, strides=1, padding=0),
                BatchNorm()
            )
        else:
            self.conv = Sequential(
                # pointwise
                Conv2d("%s_pointwise_1" % name, in_channel, hidden_dim, 1, strides=1, padding=0),
                BatchNorm(),
                ReLU(),
                # depthwise
                Conv2d("%s_depthwise_1" % name, hidden_dim, hidden_dim, 3, strides=strides, padding=1, group=hidden_dim),
                BatchNorm(),
                ReLU(),
                # pointwise
                Conv2d("%s_pointwise_2" % name, hidden_dim, out_channel, 1, strides=1, padding=0),
                BatchNorm()
            )

    def forward(self, inputs):
        if self.use_res_connect:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)


class MobileNetV2(Layer):
    def __init__(self, name, n_class=1000, input_size=224, width_mult=1):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # expand_ratio, c, n, stride
            [1, 16, 1, 1],
            # [6, 24, 2, 2],
            # [6, 32, 3, 2],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1]
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_3x3_bn("%s_first_conv3x3_bn" % name, 3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block("%s_residual_%d" % (name, i+1), 
                                    input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block("%s_residual_%d" % (name, i+1), 
                                    input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                
        self.features.append(conv_1x1_bn("%s_last_conv1x1_bn" % name, input_channel, self.last_channel))
        self.features = Sequential(*self.features)

        self.classifier = Sequential(
            Dropout(0.2),
            Dense("%s_dense" % name, self.last_channel, n_class)
        )

    def forward(self, inputs):
        y = self.features(inputs)
        y = sym.mean(y, axis=[2, 3])
        y = self.classifier(y)
        return y 


if __name__ == "__main__":
    net = MobileNetV2("mobilenet-v2")
    
    inputs = sym.Variable("inputs", shape=[111, 3, 224, 224])
    output = net(inputs)

    compute_graph = nnvm.graph.create(output)
    print(compute_graph.ir())

    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target="cuda")
    print(deploy_graph.ir())