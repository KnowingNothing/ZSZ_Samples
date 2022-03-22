#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <csignal>


void signalHandler(int signum) {
  std::cout << "Signal received " << signum << "\n";
}


class ThreadPool {
public:
    ThreadPool(size_t);

    template<typename FType, typename... Args>
    auto push_front(FType&& f, Args&&... args) ->std::future<decltype(f(args...))>;

    template<typename FType, typename... Args>
    auto push_back(FType&& f, Args&&... args) ->std::future<decltype(f(args...))>;

    size_t num_workers();

    ~ThreadPool();
private:

    std::vector< std::thread > workers;
    std::deque< std::function<void()> > tasks;
    
    std::mutex deque_mutex;
    std::condition_variable condition;
    bool stop;
};


inline ThreadPool::ThreadPool(size_t threads=std::thread::hardware_concurrency()) : stop(false) {
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this] {
                for(;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->deque_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop_front();
                    }

                    task();
                }
            }
        );
}


template<typename FType, typename... Args>
auto ThreadPool::push_front(FType&& f, Args&&... args) ->std::future<decltype(f(args...))> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_front on stopped ThreadPool");

        tasks.emplace_front([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}


template<typename FType, typename... Args>
auto ThreadPool::push_back(FType&& f, Args&&... args) ->std::future<decltype(f(args...))> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_back on stopped ThreadPool");

        tasks.emplace_back([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}


size_t ThreadPool::num_workers() {
  return workers.size();
}


inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(deque_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}


class Func1 {
 public:
  size_t operator() () {
    // throw std::runtime_error("an error");
    abort();
    return 0U;
  }
};


class Func2 {
 public:
  size_t operator() () {
    return 1U;
  }
};


int main() {
  signal(SIGABRT, signalHandler);
  ThreadPool pool;
  std::cout << "initial workers: " << pool.num_workers() << "\n";

  Func1 func1;
  Func2 func2;
  for (int i = 0; i < 100; ++i) {
    std::future<size_t> future;
    if (i % 2 == 0) {
      future = pool.push_back(func1);
    } else {
      future = pool.push_back(func2);
    }
    try {
      size_t result = future.get();
      std::cout << "get result: " << result << "\n";
    } catch(...) {
      std::cout << "catch exception " << "\n";
    }
    std::cout << "workers: " << pool.num_workers() << "\n";
  }

  return 0;
}