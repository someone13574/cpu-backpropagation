#include "backpropagation.hpp"

int main()
{
    Backpropagation::Training_Example training_example_1;
    training_example_1.m_inputs = Matrix<double>(2, 1);
    training_example_1.m_expected_outputs = Matrix<double>(1, 1);
    Backpropagation::Training_Example training_example_2;
    training_example_2.m_inputs = Matrix<double>(2, 1);
    training_example_2.m_expected_outputs = Matrix<double>(1, 1);
    Backpropagation::Training_Example training_example_3;
    training_example_3.m_inputs = Matrix<double>(2, 1);
    training_example_3.m_expected_outputs = Matrix<double>(1, 1);
    Backpropagation::Training_Example training_example_4;
    training_example_4.m_inputs = Matrix<double>(2, 1);
    training_example_4.m_expected_outputs = Matrix<double>(1, 1);

    training_example_1.m_inputs(0, 0) = 0;
    training_example_1.m_inputs(1, 0) = 0;
    training_example_1.m_expected_outputs(0, 0) = 0;
    training_example_2.m_inputs(0, 0) = 0;
    training_example_2.m_inputs(1, 0) = 1;
    training_example_2.m_expected_outputs(0, 0) = 1;
    training_example_3.m_inputs(0, 0) = 1;
    training_example_3.m_inputs(1, 0) = 0;
    training_example_3.m_expected_outputs(0, 0) = 1;
    training_example_4.m_inputs(0, 0) = 1;
    training_example_4.m_inputs(1, 0) = 1;
    training_example_4.m_expected_outputs(0, 0) = 0;
    Matrix<Backpropagation::Training_Example> data(8, 1);
    data(0, 0) = training_example_1;
    data(1, 0) = training_example_2;
    data(2, 0) = training_example_3;
    data(3, 0) = training_example_4;
    data(4, 0) = training_example_1;
    data(5, 0) = training_example_2;
    data(6, 0) = training_example_3;
    data(7, 0) = training_example_4;

    Matrix<unsigned int> topology(3, 1);
    topology(0, 0) = 2;
    topology(1, 0) = 4;
    topology(2, 0) = 1;
    Network network(topology);

    Backpropagation backpropagation(network, data, 3.0, 4, 1000, 4);

    return 0;
}
