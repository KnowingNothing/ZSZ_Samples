for i in range(10):
for j in range(10):
    B(i, j) = (A(i, j-1) + A(i, j) + A(i, j+1)) / 3

for i in range(10):
for j in range(10):
    C(i, j) = (B(i-1, j) + B(i, j) + B(i, j+1)) / 3

for i in range(10):
    for j in range(10):
        B(i-1, j) = (A(i-1, j-1) + A(i-1, j) + A(i-1, j+1)) / 3
        B(i, j) = (A(i, j-1) + A(i, j) + A(i, j+1)) / 3
        B(i+1, j) = (A(i+1, j-1) + A(i+1, j) + A(i+1, j+1)) / 3
    for j in range(10):
        C(i, j) = (B(i-1, j) + B(i, j) + B(i, j+1)) / 3