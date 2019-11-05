#include <iostream>
#include "Perceptron.h"

using namespace std;

int main(int argc, char const *argv[])
{
    Perceptron p(100.f, 0.01f, 0.01f);
    cout << p.getTreshold() << endl;
    return 0;
}
