#include "Perceptron.h"

using namespace at;
using namespace std;

Perceptron::Perceptron(float treshold, float learning_rate, float bias){
    setTreshold(treshold);
    setLearning_Rate(learning_rate);
    setBias(bias);
    torch::Tensor r = rand(784);
    setW(r);
}

void Perceptron::fit(){
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(5).workers(8));

   for (torch::data::Example<> &batch : *data_loader)
    {
        for (int64_t i = 0; i < batch.data.size(0); ++i)
        {
            int label = batch.target[i].item<int64_t>(); 
            if(label == 0 || label == 1) {
                Tensor input = batch.data[i][0].flatten();
                int prediction = predict(input);
                Tensor update_weights =+ getLearning_Rate() * (label - prediction) * input;
                setW(update_weights);
                int update_bias =+ getLearning_Rate() * (label - prediction);
                setBias(update_bias);
                cout << "label |" << label << " prediction | " << prediction << endl;
            }
        }
    }
}

int Perceptron::predict(Tensor data){
    cout << "sample: " << data.sizes() << endl;
    cout << "weights: " << getW().transpose(0,-1).sizes() << endl;
    // torch::print(cout, data, 99);
    Tensor prediction = dot(data, getW().transpose(0,-1));
    int relu = relu_derivative(prediction.item().toFloat());
    return relu;
}

int Perceptron::relu_derivative(float x){
    if (x <= 0) return 0;
    else if (x > 0) return 1;
}

torch::data::Example<>  Perceptron::clean_data(){
    Tensor data = torch::rand({784});
    // Tensor labels;

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
            // if(batch.target[i].item() > 0) {

                // labels = torch::cat({data, batch.data[i][0].flatten()}, 0);   
                
                  
                int bar = batch.target[i].item<int64_t>(); 
                cout << bar << " |";
                // cout << labels << endl;
            // }
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