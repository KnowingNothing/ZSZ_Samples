#ifndef MEASURE_METHOD_H_
#define MEASURE_METHOD_H_


#include <chrono>
#include <memory>
#include <iostream>

namespace measure {

class Measuree {
 public:
  virtual ~Measuree() {}
  virtual void run()=0;
};


class StreamSynchronizer {
 public:
  virtual ~StreamSynchronizer() {}
  virtual void init()=0;
  virtual void sync()=0;
};


class Measurer {
 public:
  double measure(std::shared_ptr<Measuree> measuree, std::shared_ptr<StreamSynchronizer> stream_sync, int n_times) {
    if (n_times <= 0) {
      std::cerr << "n_times should > 0! get " << n_times << "\n";
      abort();
    }
    measuree->run();
    stream_sync->sync();

    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_times; i++) {
        measuree->run();
    }
    stream_sync->sync();
    auto tend = std::chrono::high_resolution_clock::now();

    double cost = std::chrono::duration_cast<std::chrono::microseconds>(tend - tbegin).count();
    return cost / n_times;
  }
};

}  // measure


#endif  // MEASURE_METHOD_H_