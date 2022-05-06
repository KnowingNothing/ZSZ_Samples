#include <iostream>
#include <immintrin.h>
#include <bitset>

using namespace std;


void print_vector(__m256 &vec) {
  for (int i = 0; i < 7; ++i) {
    cout << vec[i] << ", ";
  }
  cout << vec[7] << endl;
}


int main(){

  char imm = _MM_SHUFFLE(0,1,2,3);
  bitset<8> x(imm);
  cout << x << endl;

  __m256 r0 = {11.0f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f};
  __m256 r1 = {21.0f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f};
  __m256 t0, t1, t3;
  t0 = _mm256_unpacklo_ps(r0, r1);
  print_vector(t0);
  t1 = _mm256_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 0, 1, 0));
  print_vector(t1);
  t3 = _mm256_permute2f128_ps(r0, r1, 0b00100000);
  print_vector(t3);
  return 0;
}