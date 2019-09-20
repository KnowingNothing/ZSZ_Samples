import nnvm
from nnvm import symbol as sym


if __name__ == "__main__":
    x = sym.Variable("x", shape=[4, 5, 7, 9])
    y = sym.Variable("y", shape=[6, 5, 3, 3])
    z = sym.conv2d(name="z", channels=6, kernel_size=(1, 3), strides=(1, 1), padding=(1, 1), data=x)
    a = sym.batch_norm(z)

    compute_graph = nnvm.graph.create(a)
    print(compute_graph.ir())

    deploy_graph, lib, params = nnvm.compiler.build(
        compute_graph, target="cuda")
    print(deploy_graph.ir())

    print(lib.imported_modules[0].get_source())