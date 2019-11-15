#include <ATen/ATen.h>
#include <torch/torch.h>

class Perceptron
{
private:
    torch::Tensor w;
    float treshold;
    float learning_rate;
    float bias;
    
    
public:
    Perceptron(float treshold, float learning_rate, int neurons);
    void fit(int train);
    int predict(torch::Tensor data);        
    void clean_data(torch::Tensor& train_imgs_pointer, torch::Tensor &train_labels_pointer
    , torch::Tensor &test_imgs_pointer, torch::Tensor &test_labels_pointer);
    void test();
    void test_second_layer(Perceptron p);
    int ReLU(float x);
    float Sig(float x);
    float Tanh(float x);
    void fit_second_level(Perceptron p);

    //member functions
    at::Tensor getW();
    void setW(at::Tensor w_value);
    float getTreshold();
    void setTreshold(float treshold_value);
    float getLearning_Rate();
    void setLearning_Rate(float learning_rate_value);
    float getBias();
    void setBias(float bias_value);

};