import tvm
import numpy as np


def avgpool(C, H, W, K, R, S, stride=2):
    A = tvm.te.placeholder([C, H, W], dtype="float32", name="data")
    B = tvm.te.placeholder([K], dtype="float32", name="weight")
    bias = tvm.te.placeholder([K], dtype="float32", name="bias")

    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")
    Output = tvm.te.compute(
        [K, (H - R + 1)//stride, (W - S + 1)//stride],
        lambda k, p, q:
            tvm.te.sum(
                A[k, p*stride+rr, q*stride+rs] * B[k]/(R*S) + bias[k]/(R*S),
                axis=[rr, rs]
            ),
        name="output"
    )

    return [A, B, bias, Output]

class Config:
    def __init__(self):
        self.tile_p = [-1, 1, 1, 1]
        self.tile_q = [-1, 1, 1, 1]
        self.tile_k = [-1, 1, 1, 1]
        self.tile_r = [-1, 1, 1]
        self.tile_s = [-1, 1, 1]

def schedule_avgpool_local_mem(A, B, bias, Output, config):
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

    sch = tvm.te.create_schedule(Output.op)
    LL = sch.cache_write(Output, "local")
    # AA = sch.cache_read(A, "shared", [LL])
    # BB = sch.cache_read(B, "shared", [LL])

    k, p, q = sch[Output].op.axis
    k0, k1, k2, k3 = tile_axes(sch, Output, k, config.tile_k)
    p0, p1, p2, p3 = tile_axes(sch, Output, p, config.tile_p)
    q0, q1, q2, q3 = tile_axes(sch, Output, q, config.tile_q)

    sch[Output].reorder(k0, p0, q0, k1, p1, q1, k2, p2, q2, k3, p3, q3)
    sch[Output].bind(k0, bz)
    sch[Output].bind(p0, by)
    sch[Output].bind(q0, bx)

    sch[Output].bind(k2, tz)
    sch[Output].bind(p2, ty)
    sch[Output].bind(q2, tx)

    sch[LL].compute_at(sch[Output], q2)
    k, p, q = sch[LL].op.axis
    r, s = sch[LL].op.reduce_axis
    r0, r1, r2 = tile_axes(sch, LL, r, config.tile_r)
    s0, s1, s2 = tile_axes(sch, LL, s, config.tile_s)
    sch[LL].reorder(r0, s0, r1, s1, k, p, q, r2, s2)

    # sch[AA].compute_at(sch[LL], s0)
    # sch[BB].compute_at(sch[LL], s0)

    # for SS in [AA, BB]:
    #     axes = sch[SS].op.axis
    #     fused = sch[SS].fuse(*axes)
    #     fused, ttz, tty, ttx = tile_axes(
    #         sch, SS, fused,
    #         [-1, config.tile_k[2], config.tile_p[2], config.tile_q[2]])
    #     sch[SS].bind(ttz, tz)
    #     sch[SS].bind(tty, ty)
    #     sch[SS].bind(ttx, tx)
    
    return sch


def schedule_avgpool_vload(A, B, bias, Output, config):
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

    sch = tvm.te.create_schedule(Output.op)
    LL = sch.cache_write(Output, "local")
    AA = sch.cache_read(A, "shared", [LL])
    BB = sch.cache_read(B, "shared", [LL])

    k, p, q = sch[Output].op.axis
    k0, k1, k2, k3 = tile_axes(sch, Output, k, config.tile_k)
    p0, p1, p2, p3 = tile_axes(sch, Output, p, config.tile_p)
    q0, q1, q2, q3 = tile_axes(sch, Output, q, config.tile_q)

    sch[Output].reorder(k0, p0, q0, k1, p1, q1, k2, p2, q2, k3, p3, q3)
    sch[Output].bind(k0, bz)
    sch[Output].bind(p0, by)
    sch[Output].bind(q0, bx)

    sch[Output].bind(k2, tz)
    sch[Output].bind(p2, ty)
    sch[Output].bind(q2, tx)

    sch[LL].compute_at(sch[Output], q2)
    k, p, q = sch[LL].op.axis
    r, s = sch[LL].op.reduce_axis
    r0, r1, r2 = tile_axes(sch, LL, r, config.tile_r)
    s0, s1, s2 = tile_axes(sch, LL, s, config.tile_s)
    sch[LL].reorder(r0, s0, r1, s1, k, p, q, r2, s2)

    sch[AA].compute_at(sch[LL], s0)
    sch[BB].compute_at(sch[LL], s0)

    for SS in [AA, BB]:
        if SS == BB:
            vv = 2
        else:
            vv = 1
        axes = sch[SS].op.axis
        fused = sch[SS].fuse(*axes)
        fused, ttz, tty, ttx, vec = tile_axes(
            sch, SS, fused,
            [-1, config.tile_k[2], config.tile_p[2], config.tile_q[2], vv])
        sch[SS].bind(ttz, tz)
        sch[SS].bind(tty, ty)
        sch[SS].bind(ttx, tx)
        sch[SS].vectorize(vec)
    
    return sch


FUNCS = {
    "local_mem": schedule_avgpool_local_mem,
    "vload": schedule_avgpool_vload
}


def run(C, H, W, K, R, S, stride, config, option="local_mem"):
    A, B, bias, Output = avgpool(C, H, W, K, R, S)
    sch = FUNCS[option](A, B, bias, Output, config)
    A_np = np.random.uniform(-1, 1, [C, H, W]).astype("float32")
    B_np = np.random.uniform(-1, 1, [K]).astype("float32")
    bias_np = np.random.uniform(-1, 1, [K]).astype("float32")
    Output_np = np.random.uniform(-1, 1, [K, (H - R + 1)//stride, (W - R + 1)//stride]).astype("float32")

    ctx = tvm.context("opencl")
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    bias_tvm = tvm.nd.array(bias_np, ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    func = tvm.build(sch, [A, B, bias, Output], "opencl")
    with open(f"lenet_avgpool_{C}_{H}_{W}_{K}_{R}_{S}_{stride}_" + option + ".cl", "w") as fout:
        print(func.imported_modules[0].get_source(), file=fout)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    cost = evaluator(A_tvm, B_tvm, bias_tvm, Output_tvm).mean * 1e3
    return cost


lenet_shapes = [
    (1, 6, 28, 28, 6, 1, 2, 2, 1, 2, 0, 1, 1),
    (1, 16, 10, 10, 16, 1, 2, 2, 1, 2, 0, 1, 1)
]

config1 = Config()
config1.tile_k = [6, 1, 1, 1]
config1.tile_p = [7, 1, 2, 1]
config1.tile_q = [7, 1, 2, 1]
config1.tile_r = [1, 2, 1]
config1.tile_s = [1, 1, 2]

config2 = Config()
config2.tile_k = [16, 1, 1, 1]
config2.tile_p = [5, 1, 1, 1]
config2.tile_q = [5, 1, 1, 1]
config2.tile_r = [1, 2, 1]
config2.tile_s = [1, 1, 2]

# config1 = Config()
# config1.tile_k = [6, 1, 1, 1]
# config1.tile_p = [7, 1, 1, 2]
# config1.tile_q = [7, 1, 2, 1]
# config1.tile_r = [1, 2, 1]
# config1.tile_s = [1, 1, 2]

# config2 = Config()
# config2.tile_k = [8, 1, 1, 2]
# config2.tile_p = [5, 1, 1, 1]
# config2.tile_q = [5, 1, 1, 1]
# config2.tile_r = [1, 2, 1]
# config2.tile_s = [1, 1, 2]

lenet_configs = [
    config1, config2
]

if __name__ == "__main__":
    for shape, config in zip(lenet_shapes, lenet_configs):
        N, C, H, W, K, _, R, S, _, stride, padding, dilation, group = shape
        cost = run(C, H, W, K, R, S, stride, config, option="local_mem")
        print(f"shape: N={N},C={C},H={H},W={W},K={K},R={R},S={S},stride={stride}")
        print("local_mem", cost, "ms")