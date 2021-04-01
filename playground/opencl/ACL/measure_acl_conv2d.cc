#define ARM_COMPUTE_CL

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <unistd.h>


#include "measure_acl.h"
#include "measure_method.h"

#include "arm_compute/graph.h"
#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/Utils.h"
#endif /* ARM_COMPUTE_CL */
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

using namespace measure;
using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;
using namespace utils;

class ConvolutionMeasuree : public Measuree {
public:
  ConvolutionMeasuree(size_t N, size_t C, size_t H, size_t W, size_t K,
                      size_t R, size_t S, size_t stride = 1, size_t padding = 0,
                      std::string dtype = "float32") {
    PadStrideInfo conv_info(stride, stride, padding, padding);
    Format format;
    if (dtype == "float32") {
        format = Format::F32;
    } else if (dtype == "float16") {
        format = Format::F16;
    } else if (dtype == "int8") {
        format = Format::U8;
    }
    src0_.allocator()->init(TensorInfo(TensorShape(W, H, C, N), format));
    src1_.allocator()->init(TensorInfo(TensorShape(S, R, C, K), format));
    size_t w_out = (W - S + 2 * padding) / stride + 1;
    size_t h_out = (H - R + 2 * padding) / stride + 1;
    dst_.allocator()->init(TensorInfo(TensorShape(w_out, h_out, K, N), format));

    conv2d.configure(&src0_, &src1_, nullptr, &dst_, conv_info);

    src0_.allocator()->allocate();
    src1_.allocator()->allocate();
    dst_.allocator()->allocate();

    fill_random_tensor(src0_, -1.f, 1.f);
    fill_random_tensor(src1_, -1.f, 1.f);
  }

  void run() { conv2d.run(); }

private:
  CLTensor src0_, src1_, dst_;
  CLConvolutionLayer conv2d;
};

void test(size_t N, size_t C, size_t H, size_t W, size_t K, size_t R, size_t S,
          size_t stride = 1, size_t padding = 0, std::string dtype = "float32",
          int trials = 10) {
  auto Mr = std::make_shared<Measurer>();
  CLTuner tuner;
  auto sync = std::make_shared<ACLStreamSynchronizer>(&tuner);
  sync->init();
  auto conv2d = std::make_shared<ConvolutionMeasuree>(N, C, H, W, K, R, S, stride,
                                                    padding, dtype);
  double cost = Mr->measure(conv2d, sync, trials);
  std::cout << "Time cost of Convolution with shape (" << N << "x" << C << "x"
            << H << "x" << W << ") * (" << K << "x" << C << "x" << R << "x" << S
            << ") "
            << "stride=" << stride << " padding=" << padding << " "
            << "dtype= " << dtype << " is: " << cost / 1e3 << " ms (" << trials
            << " runs).\n";
  sleep(20);
}

size_t arg_list[][9] = {
    // batch, in_channel, height, width, out_channel, kernel_h, kernel_w,
    // stride, padding
    {1, 32, 112, 112, 32, 3, 3, 1, 1}, {1, 16, 112, 112, 96, 3, 3, 2, 1},
    {1, 24, 56, 56, 144, 3, 3, 2, 1},  {1, 32, 28, 28, 192, 3, 3, 2, 1},
    {1, 64, 14, 14, 384, 3, 3, 1, 1},  {1, 96, 14, 14, 576, 3, 3, 2, 1},
    {1, 160, 7, 7, 960, 3, 3, 1, 1},
};

int main() {
  size_t length = sizeof(arg_list) / sizeof(arg_list[0]);
  for (int i = 0; i < length; ++i) {
    size_t N, C, H, W, K, R, S, stride, padding;
    N = arg_list[i][0];
    C = arg_list[i][1];
    H = arg_list[i][2];
    W = arg_list[i][3];
    K = arg_list[i][4];
    R = arg_list[i][5];
    S = arg_list[i][6];
    stride = arg_list[i][7];
    padding = arg_list[i][8];
    test(N, C, H, W, K, R, S, stride, padding, "int8", 10);
  }
  return 0;
}