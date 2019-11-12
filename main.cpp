#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "Perceptron.h"

using namespace std;

int main(int argc, char const *argv[])
{
    Perceptron p(100,0.01, 0.01);
    // torch::Tensor r = torch::rand(784);
    // p.predict(r);
    torch::Tensor train_imgs;
    torch::Tensor train_labels;
    torch::Tensor test_imgs;
    torch::Tensor test_labels;
    
    p.clean_data(train_imgs, train_labels, test_imgs, test_labels);
    std::cout << "train_imgs: " << train_imgs.size(0)  << std::endl;
    std::cout << "train_labels: " << train_labels.size(0)  << std::endl;
    std::cout << "test_imgs: " << test_imgs.size(0)  << std::endl;
    std::cout << "test_labels: " << test_labels.size(0)  << std::endl;

    return 0;
}
