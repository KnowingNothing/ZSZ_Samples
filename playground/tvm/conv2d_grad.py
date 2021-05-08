import tvm


def grad_to_img(C, H, W, K, R, S):
    P = H - R + 1
    Q = W - S + 1
    grad_to_output = tvm.te.placeholder([K, P, Q], name="grad_to_output")
    input = tvm.te.placeholder([C, H, W], name="input")
    weight = tvm.te.placeholder([K, C, R, S], name="weight")
    rk = tvm.te.reduce_axis([0, K], name="rk")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")
    padding = tvm.te.compute(
        [C, H + R - 1, W + S - 1],
        lambda c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= R - 1, h < P + R - 1, w >= S - 1, w < W + S - 1),
                grad_to_output[c, h - R + 1, w - S + 1],
                0.0
            )
    )
    grad_to_input = tvm.te.compute(
        [C, H, W],
        lambda c, h, w:
            tvm.te.sum(padding[rk, h + rr, w + rs] * weight[rk, c, R - rr + 1, S - rs + 1] * (1 - input[c, h, w] * input[c, h, w]), axis=[rk, rr, rs]),
            name="grad_to_input"
    )

    sch = tvm.te.create_schedule(grad_to_input.op)
    sch[padding].compute_inline()
    c, h, w = sch[grad_to_input].op.axis
    sch[grad_to_input].bind(c, tvm.te.thread_axis("blockIdx.z"))
    sch[grad_to_input].bind(h, tvm.te.thread_axis("blockIdx.y"))
    sch[grad_to_input].bind(w, tvm.te.thread_axis("blockIdx.x"))
    
    func = tvm.build(sch, [grad_to_output, input, weight, grad_to_input], "opencl")
    print(func.imported_modules[0].get_source())


def grad_to_weight(C, H, W, K, R, S):
    P = H - R + 1
    Q = W - S + 1
    grad_to_output = tvm.te.placeholder([K, P, Q], name="grad_to_output")
    input = tvm.te.placeholder([C, H, W], name="input")
    rp = tvm.te.reduce_axis([0, P], name="rp")
    rq = tvm.te.reduce_axis([0, Q], name="rq")

    grad_to_weight = tvm.te.compute(
        [K, C, R, S],
        lambda k, c, r, s:
            tvm.te.sum(grad_to_output[k, rp, rq] * input[c, r + rp, s + rq], axis=[rp, rq]),
            name="grad_to_weight"
    )

    sch = tvm.te.create_schedule(grad_to_weight.op)
    k, c, r, s = sch[grad_to_weight].op.axis
    sch[grad_to_weight].bind(k, tvm.te.thread_axis("blockIdx.y"))
    sch[grad_to_weight].bind(c, tvm.te.thread_axis("blockIdx.x"))
    
    func = tvm.build(sch, [grad_to_output, input, grad_to_weight], "opencl")
    print(func.imported_modules[0].get_source())


def grad_to_img_pooling(C, H, W, K, R, S):
    P = H - R + 1
    Q = W - S + 1
    grad_to_output = tvm.te.placeholder([K, P, Q], name="grad_to_output")
    input = tvm.te.placeholder([C, H, W], name="input")
    weight = tvm.te.placeholder([K], name="weight")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")
    padding = tvm.te.compute(
        [C, H + R - 1, W + S - 1],
        lambda c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= R - 1, h < P + R - 1, w >= S - 1, w < W + S - 1),
                grad_to_output[c, h - R + 1, w - S + 1],
                0.0
            )
    )
    grad_to_input = tvm.te.compute(
        [C, H, W],
        lambda c, h, w:
            tvm.te.sum(padding[c, h + rr, w + rs] * weight[c] * (1 - input[c, h, w] * input[c, h, w]), axis=[rr, rs]),
            name="grad_to_input"
    )

    sch = tvm.te.create_schedule(grad_to_input.op)
    sch[padding].compute_inline()
    c, h, w = sch[grad_to_input].op.axis
    sch[grad_to_input].bind(c, tvm.te.thread_axis("blockIdx.z"))
    sch[grad_to_input].bind(h, tvm.te.thread_axis("blockIdx.y"))
    sch[grad_to_input].bind(w, tvm.te.thread_axis("blockIdx.x"))
    
    func = tvm.build(sch, [grad_to_output, input, weight, grad_to_input], "opencl")
    print(func.imported_modules[0].get_source())


def grad_to_weight_pooling(C, H, W, K, R, S):
    P = H - R + 1
    Q = W - S + 1
    grad_to_output = tvm.te.placeholder([K, P, Q], name="grad_to_output")
    input = tvm.te.placeholder([C, H, W], name="input")
    rp = tvm.te.reduce_axis([0, P], name="rp")
    rq = tvm.te.reduce_axis([0, Q], name="rq")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    grad_to_weight = tvm.te.compute(
        [K],
        lambda k:
            tvm.te.sum(grad_to_output[k, rp, rq] * input[k, rr + rp, rs + rq], axis=[rr, rs, rp, rq]),
            name="grad_to_weight"
    )

    sch = tvm.te.create_schedule(grad_to_weight.op)
    k, = sch[grad_to_weight].op.axis
    sch[grad_to_weight].bind(k, tvm.te.thread_axis("blockIdx.x"))
    
    func = tvm.build(sch, [grad_to_output, input, grad_to_weight], "opencl")
    print(func.imported_modules[0].get_source())


LeNetShapes = [
    # C, H, W, K, R, S
    (1, 32, 32, 6, 5, 5),
    (6, 28, 28, 6, 2, 2),
    (6, 14, 14, 16, 5, 5),
    (16, 10, 10, 16, 2, 2),
    (16, 5, 5, 120, 5, 5)
]

if __name__ == "__main__":
    # for shape in LeNetShapes[::2]:
        # grad_to_img(*shape)
        # grad_to_weight(*shape)
    for shape in LeNetShapes[1::2]:
        grad_to_img_pooling(*shape)
        # grad_to_weight_pooling(*shape)