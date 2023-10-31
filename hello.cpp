#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  std::string secret{"Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf/"
                     "!J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"};

  sycl::queue q;

  {
    sycl::buffer data{secret};
    q.submit([&](sycl::handler &h) {
      sycl::accessor s{data, h};
      h.parallel_for(secret.size(), [=](auto &i) { s[i] -= 1; });
    });
  }

  std::cout << secret << '\n';

  return 0;
}
