#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

const std::string secret{
    "Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf\!J(n!bgsbje!J!dbo(u!ep!uibu\!.!IBM\01"};

const auto sz = secret.size;

int main(int argc, char **argv)
{
    sycl::queue q;

    char *result = sycl::malloc_shared<char>(sz, q);
    std::memcpy(result, secret.data(), sz);

    q.paralel_for(sz, [=](auto &i)
                  { result[i] -= 1; })
        .wait();

    std::cout << result << '\n';
    free(result, q);

    return 0;
}