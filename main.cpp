#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "Perceptron.h"

using namespace std;

int main(int argc, char const *argv[])
{
    // Excersie 1
    int excercise = 1;

    if (excercise == 1)
    {
        Perceptron p(100, 0.01, 784);
        p.fit(1);
        p.test();
    }    

    return 0;
}
