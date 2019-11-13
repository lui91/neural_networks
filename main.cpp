#include <iostream>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "Perceptron.h"

using namespace std;

int main(int argc, char const *argv[])
{
    int clean_data = 0;
    int train =0;

    Perceptron p(100,0.01, 0.01);
    torch::Tensor train_imgs;
    torch::Tensor train_labels;
    torch::Tensor test_imgs;
    torch::Tensor test_labels;

    if (clean_data)
    {
        p.clean_data(train_imgs, train_labels, test_imgs, test_labels);
        std::cout << "train_imgs: " << train_imgs.size(0) << "|" <<  train_imgs.size(1) << std::endl;
        std::cout << "train_labels: " << train_labels.size(0)  << std::endl;
        std::cout << "test_imgs: " << test_imgs.size(0) << "|" <<  test_imgs.size(1) << std::endl;
        std::cout << "test_labels: " << test_labels.size(0)  << std::endl;  
    }else
    {
        torch::load(train_imgs, "train_imgs.pt");
        torch::load(train_labels, "train_labels.pt");
        torch::load(test_imgs, "test_imgs.pt");
        torch::load(test_labels, "test_labels.pt");
        
    }
    
    // torch::print(cout, train_imgs[0].reshape(-1), 99);
    // std::cout << "train_imgs: " << train_imgs[0].size(0) << "|" <<  train_imgs[0].size(1) << std::endl;
    if (train)
    {
        p.fit(train_imgs, train_labels);
    }else
    {
        torch::Tensor w;
        torch::load(w, "weights.pk");
        p.setW(w);
    }
    
    p.test(test_imgs, test_labels);

    return 0;
}
