#define ARM_COMPUTE_CL

#include "arm_compute/runtime/CL/CLFunctions.h"
#include "utils/Utils.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <unistd.h>


#include "measure_acl.h"
#include "measure_method.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

using namespace measure;
using namespace arm_compute;
using namespace utils;

class GEMMMeasuree : public Measuree {
public:
  GEMMMeasuree(size_t M, size_t N, size_t K, float alpha = 1.0f,
               float beta = 0.0f, std::string dtype = "float32",
               bool bias = false) {
    if (dtype == "float32") {
      src0_.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
      src1_.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
      src2_.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

      init_sgemm_output(dst_, src0_, src1_, DataType::F32);
    } else if (dtype == "float16") {
      src0_.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F16));
      src1_.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F16));
      src2_.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F16));

      init_sgemm_output(dst_, src0_, src1_, DataType::F16);
    } else {
      std::cerr << "Don't know data type: " << dtype << ".";
      abort();
    }

    sgemm.configure(&src0_, &src1_, bias ? &src2_ : nullptr, &dst_, alpha,
                    beta);

    src0_.allocator()->allocate();
    src1_.allocator()->allocate();
    dst_.allocator()->allocate();

    if (bias) {
      src2_.allocator()->allocate();
      fill_random_tensor(src2_, -1.f, 1.f);
    }
    fill_random_tensor(src0_, -1.f, 1.f);
    fill_random_tensor(src1_, -1.f, 1.f);
  }

  void run() { sgemm.run(); }

private:
  CLTensor src0_, src1_, src2_, dst_;
  CLGEMM sgemm;
};

void test(size_t M, size_t N, size_t K, float alpha = 1.0f, float beta = 0.0f,
          std::string dtype = "float32", bool bias = false, int trials = 10) {
  auto Mr = std::make_shared<Measurer>();
  CLTuner tuner;
  auto sync = std::make_shared<ACLStreamSynchronizer>(&tuner);
  sync->init();
  auto gemm = std::make_shared<GEMMMeasuree>(M, N, K, alpha, beta, dtype, bias);
  double cost = Mr->measure(gemm, sync, trials);
  std::cout << "Time cost of GEMM with shape " << M << "x" << N << "x" << K
            << " dtype= " << dtype << " is: " << cost / 1e3 << " ms(" << trials
            << "runs).\n";
  sleep(20);
}

int main() {
  for (int s = 4; s <= 10; ++s) {
    test((size_t)std::pow(2, s), (size_t)std::pow(2, s),
         (size_t)std::pow(2, s));
  }
  return 0;
}