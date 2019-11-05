#include <armadillo>

using namespace arma;

class Perceptron
{
private:
    static vec w;
    float treshold;
    float learning_rate;
    float bias;
    int relu_derivative(float x);

public:
    Perceptron(float treshold, float learning_rate, float bias);
    void fit(float data, int labels, int iterations);
    double predict(vec data);
    static vec getW();
    void setW(vec w_value);
    float getTreshold();
    void setTreshold(float treshold_value);
    float getLearning_Rate();
    void setLearning_Rate(float learning_rate_value);
    float getBias();
    void setBias(float bias_value);

};