#include <ATen/ATen.h>
#include <torch/torch.h>

class Perceptron
{
private:
    at::Tensor w;
    float treshold;
    float learning_rate;
    float bias;
    int relu_derivative(float x);
    
public:
    Perceptron(float treshold, float learning_rate, float bias);
    void fit();
    int predict(at::Tensor data);        
    void clean_data(torch::Tensor& train_imgs_pointer, torch::Tensor &train_labels_pointer
    , torch::Tensor &test_imgs_pointer, torch::Tensor &test_labels_pointer);

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