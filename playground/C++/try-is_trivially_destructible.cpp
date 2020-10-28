#include <iostream>
#include <type_traits>


class A
{
    int *a = nullptr;
    A(){ a = new int[8]; }
};

class B
{
    ~B() {}
};

int main()
{
    std::cout << std::boolalpha; 
    std::cout << std::is_trivially_destructible<A>::value << std::endl;
    std::cout << std::is_trivially_destructible<B>::value << std::endl;
    return 0;
}
