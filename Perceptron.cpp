#include "Perceptron.h"

using namespace at;
using namespace std;


struct Data{
    Tensor labels;
    Tensor imgs;
};

Perceptron::Perceptron(float treshold, float learning_rate, float bias){
    setTreshold(treshold);
    setLearning_Rate(learning_rate);
    setBias(bias);
    Tensor r = rand(30);
    setW(r);
}

void Perceptron::fit(float data, int labels, int iterations){
    for (size_t i = 0; i < getTreshold(); i++)
    {
        
    }
    

}

int Perceptron::predict(Tensor data){
    Tensor prediction = dot(data, Perceptron::getW());
    int relu = relu_derivative(prediction.item().toFloat());
    return relu;
}

int Perceptron::relu_derivative(float x){
    if (x <= 0) return 0;
    else if (x > 0) return 1;
}

torch::data::Example<>  Perceptron::clean_data(){
    Tensor data = torch::rand({28,28});
    Tensor labels;

    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(5).workers(8));

    for (torch::data::Example<> &batch : *data_loader)
    {
        
        // std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i)
        {
            // batch.target[i] = NULL;
            // cout << batch.target[i] << " ";
            
            labels = torch::cat({data, batch.data[i][0]}, 0);            
            // cout << batch.data[i] << " ";
            cout << labels << endl;
        }
        
        std::cout << "size: " << std::endl;
        
    }
    

}

//Member methods definition
Tensor Perceptron::getW(){
    return w;
}

void Perceptron::setW(Tensor w_value){
    w = w_value;
}

float Perceptron::getTreshold(){
    return treshold;
}

void Perceptron::setTreshold(float treshold_value){
    treshold = treshold_value;
}

float Perceptron::getLearning_Rate(){
    return learning_rate;
}

void Perceptron::setLearning_Rate(float learning_rate_value){
    learning_rate = learning_rate_value;
}

float Perceptron::getBias(){
    return bias;
}

void Perceptron::setBias(float bias_value){
    bias = bias_value;
}