#include <iostream>
#include <mutex>
#include <thread>
#include <deque>
#include <unistd.h>


class Queue {
 public:
  std::deque<int> q;
  std::mutex mutex;

  void push(int value) {
    std::unique_lock<std::mutex> lock(mutex);
    q.push_back(value);
  }

  int pop() {
    std::unique_lock<std::mutex> lock(mutex);
    int ret = q.front();
    q.pop_front();
    return ret;
  }

  bool has_value() {
    std::unique_lock<std::mutex> lock(mutex);
    return !q.empty();
  }
};


int main() {
  Queue q;

  std::thread consumer([&q](){
    int i = 0;
    for (;;) {
      usleep(100 * 1000);
      if (q.has_value()) {
        i += 1;
        std::cout << q.pop() << "\n";
      }
      if (i == 10) {
        return;
      }
    }
  });

  std::thread producer([&q](){
    for (int i = 0; i < 10; ++i) {
      q.push(i);
      usleep(100 * 1000);
    }
  });
  producer.join();
  consumer.join();
  return  0;
}