import tvm

dtype = "int8"

#  n-mode TTM
def TTM(Ms, N, n):
    K = Ms[n]
    A = tvm.placeholder(Ms, name="A", dtype=dtype)
    B = tvm.placeholder([N, K], name="B", dtype=dtype)

    k = tvm.reduce_axis([0, K], name="k")
    Out = tvm.compute([*Ms[:n], N, *Ms[n+1:]], lambda *ids: tvm.sum(A(*[*ids[:n], k, *ids[n+1:]]) * B[ids[n], k], axis=[k]), name="Out")

    return [A, B], [Out]



if __name__ == "__main__":
    [A, B], [Out] = TTM([512, 16, 64], 32, 1)
    sch = tvm.create_schedule(Out.op)
    print(tvm.lower(sch, [A, B, Out], simple_mode=True))
