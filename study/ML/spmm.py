for m in range(0, M):
    for k in range(A_ptr[m], A_ptr[m]+1):
        col_id = A_idx[k]
        for n in range(B_ptr[col_id], B_ptr[col_id+1]):
            C[m, B_idx[n]] += A_val[k] * B_val[n]