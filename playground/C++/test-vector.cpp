#include <iostream>
#include <vector>

int main() {
  std::vector<int> v1;
  v1.push_back(3);
  v1.push_back(4);

  std::vector<int> v2 = v1;
  std::vector<int> v3;
  v3 = v1;
  std::vector<int> v4;
  v4.push_back(5);
  v4 = v1;
  
  v2.push_back(101);
  v4.push_back(102);
  v3.push_back(103);

  std::cout << "v1:\n";
  for (auto v : v1) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  std::cout << "v2:\n";
  for (auto v : v2) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  std::cout << "v3:\n";
  for (auto v : v3) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  std::cout << "v4:\n";
  for (auto v : v4) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  std::vector<std::vector<int>> v11;
  v11.push_back({1, 2, 3});
  v11.push_back({4, 5});
  std::vector<std::vector<int>> v12;
  v12.push_back({6, 7, 8});
  v12 = v11;
  
  v12.push_back({9, 10});

  std::cout << "v11:\n";
  for (auto l : v11) {
    for (auto v : l) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }

  std::cout << "v12:\n";
  for (auto l : v12) {
    for (auto v : l) {
      std::cout << v << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
