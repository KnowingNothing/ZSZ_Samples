#include <iostream>

#define CUTLASS_HOST_DEVICE

template <typename value_t>
CUTLASS_HOST_DEVICE value_t clz(value_t x) {
    for (int i = 31; i >= 0; --i) {
        if ((1 << i) & x)
            return 31 - i;
    }
    return 32;
}

template <typename value_t>
CUTLASS_HOST_DEVICE value_t find_log2(value_t x) {
    int a = int(31 - clz(x));
    a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
    return a;
}

/**
 * Find divisor, using find_log2
 */
CUTLASS_HOST_DEVICE
void find_divisor(unsigned int& mul, unsigned int& shr, unsigned int denom) {
    if (denom == 1) {
        mul = 0;
        shr = 0;
    } else {
        unsigned int p = 31 + find_log2(denom);
        unsigned m =
                unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

        mul = m;
        shr = p - 32;
        printf("p=%u, mul=%u, shr=%u\n", p, mul, shr);
        // shr = p;
    }
}

/**
 * Find quotient and remainder using device-side intrinsics
 */
CUTLASS_HOST_DEVICE
void fast_divmod(int& quo, int& rem, int src, int div, unsigned int mul,
                 unsigned int shr) {
#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if div != 1, else simply copy the source.
    quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
    quo = int((div != 1) ? (int((int64_t)(src * mul) >> 32)) >> shr : src);
#endif

    // The remainder.
    rem = src - (quo * div);
}

// For long int input
CUTLASS_HOST_DEVICE
void fast_divmod(int& quo, int64_t& rem, int64_t src, int div, unsigned int mul,
                 unsigned int shr) {
#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if div != 1, else simply copy the source.
    quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
    quo = int((div != 1) ? (int((int64_t)(src * mul) >> 32)) >> shr : src);
#endif
    // The remainder.
    rem = src - (quo * div);
    printf("quo=%d, rem=%lld\n", quo, rem);
}


void kernel(int64_t src, int div) {
  int q;
  int64_t rem;
  unsigned int mul, shr;
  int idx = 0;
  if (idx == 0) {
    find_divisor(mul, shr, div);
    fast_divmod(q, rem, src, div, mul, shr);
  }
}


int main() {
  int div;
  int64_t src;
  src = 5;
  div = 3;
  kernel(src, div);
}