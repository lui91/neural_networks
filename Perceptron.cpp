#include "Perceptron.h"

Perceptron::Perceptron(float treshold, float learning_rate, float bias){
    setTreshold(treshold);
    setLearning_Rate(learning_rate);
    setBias(bias);
}

void fit(float data, int labels, int iterations);

double predict(vec data){
    double prediction = dot(data, Perceptron::getW());
}

int relu_derivative(float x){
    if (x <= 0) return 0;
    else if (x > 0) return 1;
}




//Member methods definition
vec Perceptron::getW(){
    return w;
}

void Perceptron::setW(vec w_value){
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