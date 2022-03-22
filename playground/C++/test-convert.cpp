#include <iostream>

class A {
 public:
  A(int v) : v_(v) {}
 private:
  int v_;
};


void test(const A& arg) {
  std::cout << "Success!" << std::endl;
  return;
}


int main() {

  test(0);
  return 0;
}