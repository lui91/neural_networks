#include "Perceptron.h"

using namespace at;
using namespace std;

Perceptron::Perceptron(float treshold, float learning_rate, float bias){
    setTreshold(treshold);
    setLearning_Rate(learning_rate);
    setBias(bias);
    torch::Tensor r = torch::rand(784);
    setW(r);
}

void Perceptron::fit(torch::Tensor &train_imgs_pointer, torch::Tensor &train_labels_pointer){

    for (size_t i = 0; i < 9000; i++)
    {
        int label = train_labels_pointer[i].item<int64_t>();
        Tensor input = train_imgs_pointer[i];
        int prediction = predict(input);
        Tensor update_weights = +getLearning_Rate() * (label - prediction) * input;
        setW(update_weights);
        int update_bias = +getLearning_Rate() * (label - prediction);
        setBias(update_bias);
        cout << "label |" << label << " prediction | " << prediction << endl;
    }
    torch::save(getW(), "weights.pk");
    cout << "Perceptron fitted" << endl;
}

void Perceptron::test(torch::Tensor test_imgs_pointer, torch::Tensor test_labels_pointer){
    int size = test_labels_pointer.size(0);
    int correct_predictions = 0;
    for (size_t i = 0; i < size; i++)
    {
       int prediction = predict(test_imgs_pointer[i]);

       if (prediction == test_labels_pointer[i].item<int64_t>()){
           correct_predictions =+ correct_predictions + 1;
       }
    }
    int precision = size / correct_predictions;
    cout << "precision: " << precision << endl;

}

int Perceptron::predict(Tensor data){
    auto prediction = dot(data, getW());
    int relu = relu_derivative(prediction.item().toFloat());
    return relu;
}

int Perceptron::relu_derivative(float x){
    if (x <= 0) return 0;
    else if (x > 0) return 1;
}

void Perceptron::clean_data(torch::Tensor &train_imgs_pointer, torch::Tensor &train_labels_pointer,
 torch::Tensor &test_imgs_pointer, torch::Tensor &test_labels_pointer){

    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(5).workers(8));

    // 5923
    // torch::Tensor zero_labels ;
    // 6742
    // torch::Tensor one_labels;

    torch::Tensor train_imgs = torch::rand({1, 784});
    std::vector <int64_t> train_vector;
    torch::Tensor test_imgs =  torch::rand({1, 784});
    std::vector <int64_t> test_vector;
    int cont = 0;
    for (torch::data::Example<> &batch : *data_loader)
    {
        for (int64_t i = 0; i < batch.data.size(0); ++i)
        {
            int label = batch.target[i].item<int64_t>(); 
            Tensor input = batch.data[i].reshape(-1).unsqueeze(0);
            if((label == 0 || label == 1) && cont < 9000) {
                train_imgs = torch::cat({train_imgs, input}, 0);
                train_vector.push_back(label);
                cont =+ cont + 1;
            }

            if((label == 0 || label == 1) && cont >= 9000) {
                test_imgs = torch::cat({test_imgs, input}, 0);
                test_vector.push_back(label);
                cont =+ cont + 1;
            }

        } 
    }

    torch::Tensor train_labels = torch::tensor(train_vector);
    torch::Tensor test_labels = torch::tensor(test_vector);
    train_imgs_pointer = train_imgs;
    train_labels_pointer = train_labels;
    test_imgs_pointer = test_imgs;
    test_labels_pointer = test_labels;
    torch::save(train_imgs, "train_imgs.pt");
    torch::save(train_labels, "train_labels.pt");
    torch::save(test_imgs, "test_imgs.pt");
    torch::save(test_labels, "test_labels.pt");
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