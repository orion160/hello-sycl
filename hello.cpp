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

int main(int argc, char **argv) {
  hello_sycl_example();

  map_example();

  return 0;
}
