import tvm

dtype = "int8"

def MTTKRP(M, N, K, L):
    A = tvm.placeholder([M, K, L], name="A", dtype=dtype)
    B = tvm.placeholder([N, K], name="B", dtype=dtype)
    C = tvm.placeholder([N, L], name="C", dtype=dtype)

    k = tvm.reduce_axis([0, K], name="k")
    l = tvm.reduce_axis([0, L], name="L")
    Out = tvm.compute([M, N], lambda m, n: tvm.sum(A[m, k, l] * B[n, k] * C[n, l], axis=[k, l]), name="Out")

    return [A, B, C], [Out]



if __name__ == "__main__":
    [A, B, C], [Out] = MTTKRP(512, 16, 32, 32)
    sch = tvm.create_schedule(Out.op)
    print(tvm.lower(sch, [A, B, C, Out], simple_mode=True))
