#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "Perceptron.h"

using namespace std;

int main(int argc, char const *argv[])
{
    Perceptron p(100,0.01, 0.01);
    // at::Tensor r = at::rand(30);
    // p.predict(r);
    auto cleaned_data = p.clean_data();

    

    return 0;
}
