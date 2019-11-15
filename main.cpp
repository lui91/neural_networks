#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "Perceptron.h"

using namespace std;

int main(int argc, char const *argv[])
{
    // Excersie 1
    int excercise = 2;

    Perceptron p(100, 0.01, 784);
    p.fit(0);
    // p.test();
       

    if (excercise == 2)
    {
        Perceptron p2(100, 0.01, 3);
        p2.fit_second_level(p);
        p2.test_second_layer(p);
    }    

    return 0;
}
