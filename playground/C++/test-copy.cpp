#include <iostream>

template<typename T, size_t N>
struct Array {
  T data[N];

  Array() {}
  Array(const Array& x) {
    for (size_t i = 0; i < N; ++i) {
      data[i] = x.data[i];
    }
  }

  void print() {
    for (size_t i = 0; i < N; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << "\n";
  }
};

int main() {
  Array<int, 5> a;
  for (int i = 0; i < 5; ++i) {
    a.data[i] = i;
  }
  Array<int, 5> b = a;
  b.print();
  Array<int, 5> c;
  c.data[0] = 100;
  c.print();
  c = a;
  c.print();
  c.data[0] = 999;
  a.print();
  c.print();
  return 0;
}