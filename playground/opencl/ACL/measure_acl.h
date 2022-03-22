#ifndef MEASURE_ACL_H_
#define MEASURE_ACL_H_

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"

#include "measure_method.h"

namespace measure {

using namespace arm_compute;

class ACLStreamSynchronizer : public StreamSynchronizer {
 public:
  ACLStreamSynchronizer(CLTuner* tuner) : tuner_(tuner) {}

  void init() {
    CLScheduler::get().default_init(tuner_);
  }

  void sync() {
    CLScheduler::get().sync();
  }

 private:
  CLTuner* tuner_;
};


} // measure

#endif  // MEASURE_ACL_H_