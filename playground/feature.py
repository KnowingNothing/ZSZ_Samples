# this is what I want
import my_module as m
import nn 


class CustomBlock(m.Compute):
    def __init__(self, input_shape, out_channel, hidden_channel):
        batch, in_channel, H, W = input_shape
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.__compute__ = self.define_compute

        self.weight1 = m.placeholder(shape=[in_channel, hidden_channel // in_channel, 3, 3], dtype="float32")
        self.weight2 = m.placeholder(shape=[out_channel, hidden_channel], dtype="float32")

    def define_compute(self, inputs):
        input_shape = inputs.shape
        batch, in_channel, H, W = input_shape
        hidden_channel = self.hidden_channel
        out_channel = self.out_channel

        ri, rj = m.reduce_axis((0, 3)), m.reduce_axis((0, 3))
        factor = hidden_channel // in_channel
        out1 = m.compute(
            (batch, hidden_channel, H, W), 
            lambda b, c, i, j: 
                m.sum(
                    inputs[b, c // factor, i + ri, j + rj] * 
                    self.weight1[c // factor, c % factor, ri, rj], 
                    axis=[ri, rj]
                    )
        )
        rc = m.reduce_axis((0, hidden_channel))
        out2 = m.compute(
            (batch, out_channel, H, W),
            lambda b, c, i, j:
                m.sum(out1[b, rc, i, j] * self.weight2[c, rc], axis=[rc])
        )
        return out2

    def forward(self, inputs):
        return self.__compute__(inputs)


inputs = m.placeholder(shape=[1, 3, 224, 224], dtype="float32")

# this should use the library such as cuDNN
conv = nn.Conv2d(inputs, out_channel=32, kernel=[3, 3], stride=2, padding=1)

# this is a user-defined function block
block = CustomBlock(conv.shape, 32, 64)

result = block.forward(conv)

# create the computation graph with optimization
compute_graph = m.create_graph(result)

# link to library and generate code for user-defined functions
runnable = m.deploy(compute_graph, target="cuda")

# run the graph
runnable.run()