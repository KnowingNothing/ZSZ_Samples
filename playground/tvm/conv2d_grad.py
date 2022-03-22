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
    # sch[grad_to_input].bind(c, tvm.te.thread_axis("blockIdx.z"))
    # sch[grad_to_input].bind(h, tvm.te.thread_axis("blockIdx.y"))
    # sch[grad_to_input].bind(w, tvm.te.thread_axis("blockIdx.x"))

    def tile_axes(sch, op, axis, factors):
        ret = []
        for f in reversed(factors[1:]):
            axis, inner = sch[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    c0, c1, c2, c3 = tile_axes(sch, grad_to_input, c, [-1, 1, 2, 1])
    h0, h1, h2, h3 = tile_axes(sch, grad_to_input, h, [-1, 1, 2, 1])
    w0, w1, w2, w3 = tile_axes(sch, grad_to_input, w, [-1, 1, 2, 1])
    sch[grad_to_input].reorder(c0, h0, w0, c1, h1, w1, c2, h2, w2, c3, h3, w3)
    sch[grad_to_input].bind(c0, bz)
    sch[grad_to_input].bind(h0, by)
    sch[grad_to_input].bind(w0, bx)
    sch[grad_to_input].bind(c2, tz)
    sch[grad_to_input].bind(h2, ty)
    sch[grad_to_input].bind(w2, tx)

    
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
    # sch[grad_to_weight].bind(k, tvm.te.thread_axis("blockIdx.y"))
    # sch[grad_to_weight].bind(c, tvm.te.thread_axis("blockIdx.x"))
    def tile_axes(sch, op, axis, factors):
        ret = []
        for f in reversed(factors[1:]):
            axis, inner = sch[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    k0, k1, k2, k3 = tile_axes(sch, grad_to_weight, k, [-1, 1, 2, 1])
    c0, c1, c2, c3 = tile_axes(sch, grad_to_weight, c, [-1, 1, 2, 1])
    sch[grad_to_weight].reorder(k0, c0, k1, c1, k2, c2, k3, c3)
    sch[grad_to_weight].bind(k0, by)
    sch[grad_to_weight].bind(c0, bx)

    sch[grad_to_weight].bind(k2, ty)
    sch[grad_to_weight].bind(c2, tx)

    
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
    # sch[grad_to_input].bind(c, tvm.te.thread_axis("blockIdx.z"))
    # sch[grad_to_input].bind(h, tvm.te.thread_axis("blockIdx.y"))
    # sch[grad_to_input].bind(w, tvm.te.thread_axis("blockIdx.x"))
    def tile_axes(sch, op, axis, factors):
        ret = []
        for f in reversed(factors[1:]):
            axis, inner = sch[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))

    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    c0, c1, c2, c3 = tile_axes(sch, grad_to_input, c, [-1, 1, 2, 1])
    h0, h1, h2, h3 = tile_axes(sch, grad_to_input, h, [-1, 1, 2, 1])
    w0, w1, w2, w3 = tile_axes(sch, grad_to_input, w, [-1, 1, 2, 1])
    sch[grad_to_input].reorder(c0, h0, w0, c1, h1, w1, c2, h2, w2, c3, h3, w3)
    sch[grad_to_input].bind(c0, bz)
    sch[grad_to_input].bind(h0, by)
    sch[grad_to_input].bind(w0, bx)
    sch[grad_to_input].bind(c2, tz)
    sch[grad_to_input].bind(h2, ty)
    sch[grad_to_input].bind(w2, tx)
    
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
        # grad_to_img_pooling(*shape)
        grad_to_weight_pooling(*shape)