#include <iostream>
#include <random>
#include <string>

#include <sycl/sycl.hpp>

void hello_sycl_example() {
  std::cout << "HELLO SYCL EXAMPLE:\n";

  std::string secret{"Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf/"
                     "!J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"};

  sycl::queue q;
  {
    sycl::buffer m_secret{secret};
    q.submit([size = secret.size(), &m_secret](sycl::handler &h) {
      sycl::accessor s{m_secret, h};
      h.parallel_for<class HelloSYCL>(sycl::range{size},
                                      [s](sycl::id<1> i) { s[i] -= 1; });
    });
  }

  std::cout << secret << '\n';
}

void map_example() {
  std::cout << "MAP EXAMPLE:\n";

  std::random_device rd;
  std::minstd_rand randomEngine{rd()};
  std::uniform_real_distribution<> dist{0.0, 1024.0};
  std::vector<double> input_data(8);
  std::generate(input_data.begin(), input_data.end(),
                [&randomEngine, &dist]() { return dist(randomEngine); });
  std::vector<double> output_data(input_data.size());

  sycl::queue q;
  {
    sycl::buffer m_in_data{input_data};
    sycl::buffer m_out_data{output_data};
    q.submit(
        [size = input_data.size(), &m_in_data, &m_out_data](sycl::handler &h) {
          sycl::accessor d_in{m_in_data, h, sycl::read_only};
          sycl::accessor d_out{m_out_data, h, sycl::write_only};
          h.parallel_for<class MapExample>(
              sycl::range{size},
              [d_in, d_out](sycl::id<1> i) { d_out[i] = sycl::sqrt(d_in[i]); });
        });
  }

  for (std::size_t i = 0; i < input_data.size(); i++) {
    std::cout << "sqrt(" << input_data[i] << ") = " << output_data[i] << '\n';
  }
}

std::unique_ptr<float[]> generate_random_matrix(size_t rows, size_t cols) {
  auto matrix{std::make_unique<float[]>(rows * cols)};
  std::random_device rd;
  std::minstd_rand randomEngine{rd()};
  std::uniform_real_distribution<> dist{0.0, 1024.0};
  std::generate(matrix.get(), matrix.get() + rows * cols,
                [&randomEngine, &dist]() { return dist(randomEngine); });
  return matrix;
}

std::unique_ptr<float[]> generate_zero_matrix(size_t rows, size_t cols) {
  auto matrix{std::make_unique<float[]>(rows * cols)};
  std::fill(matrix.get(), matrix.get() + rows * cols, 0);
  return matrix;
}

std::unique_ptr<float[]> generate_identity_matrix(size_t size) {
  auto matrix{std::make_unique<float[]>(size * size)};
  std::fill(matrix.get(), matrix.get() + size * size, 0);
  for (size_t i = 0; i < size; ++i) {
    matrix[i * size + i] = 1;
  }
  return matrix;
}

void print_matrix(const float *matrix, size_t rows, size_t cols) {
  std::cout << "[";
  for (size_t i = 0; i < rows; ++i) {
    std::cout << "[";
    for (size_t j = 0; j < cols; ++j) {
      std::cout << matrix[i * cols + j];
      if (j < cols - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]";

    if (i < rows - 1) {
      std::cout << ",\n";
    }
  }
  std::cout << "]\n";
}

void matrix_multiplication(size_t M, size_t N, size_t P, const float *A,
                           const float *B, float *C) {
  sycl::queue q;
  {
    sycl::buffer m_A{A, sycl::range<2>{M, N}};
    sycl::buffer m_B{B, sycl::range<2>{N, P}};
    sycl::buffer m_C{C, sycl::range<2>{M, P}};
    q.submit([M, N, P, &m_A, &m_B, &m_C](sycl::handler &h) {
      sycl::accessor d_A{m_A, h, sycl::read_only};
      sycl::accessor d_B{m_B, h, sycl::read_only};
      sycl::accessor d_C{m_C, h, sycl::write_only};
      h.parallel_for<class MatrixMultiplication>(
          sycl::range<2>{M, P}, [=](sycl::id<2> idx) {
            auto i{idx.get(0)};
            auto j{idx.get(1)};

            float sum = 0;
            for (size_t k = 0; k < N; k++) {
              sum += d_A[i][k] * d_B[k][j];
            }
            d_C[idx] = sum;
          });
    });
  }

  q.wait();
}

void matrix_multiplication_example_random() {
  std::cout << "MATRIX MULTIPLICATION RANDOM EXAMPLE:\n";

  constexpr size_t M{4};
  constexpr size_t N{4};
  constexpr size_t P{4};

  auto A{generate_random_matrix(M, N)};
  auto B{generate_random_matrix(N, P)};
  auto C{generate_zero_matrix(M, P)};

  matrix_multiplication(M, N, P, A.get(), B.get(), C.get());
  std::cout << "A:\n";
  print_matrix(A.get(), M, P);
  std::cout << "B:\n";
  print_matrix(B.get(), N, P);
  std::cout << "C:\n";
  print_matrix(C.get(), M, P);
}

void matrix_multiplication_example_identity() {
  std::cout << "MATRIX MULTIPLICATION IDENTITY EXAMPLE:\n";

  constexpr size_t M{4};
  constexpr size_t N{4};
  constexpr size_t P{N};

  auto A{generate_random_matrix(M, N)};
  auto B = generate_identity_matrix(N);
  auto C{generate_zero_matrix(M, P)};
  matrix_multiplication(M, N, P, A.get(), B.get(), C.get());
  std::cout << "A:\n";
  print_matrix(A.get(), M, P);
  std::cout << "B:\n";
  print_matrix(B.get(), N, P);
  std::cout << "C:\n";
  print_matrix(C.get(), M, P);
}

int main(int argc, char **argv) {
  hello_sycl_example();

  map_example();

  matrix_multiplication_example_identity();

  matrix_multiplication_example_random();

  return 0;
}
