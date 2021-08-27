#include "Backpropagation.hpp"

int main()
{
    Backpropagation::TrainingExample trainingExample1;
    trainingExample1.m_inputs = Matrix<double>(2, 1);
    trainingExample1.m_expectedOutputs = Matrix<double>(1, 1);
    Backpropagation::TrainingExample trainingExample2;
    trainingExample2.m_inputs = Matrix<double>(2, 1);
    trainingExample2.m_expectedOutputs = Matrix<double>(1, 1);
    Backpropagation::TrainingExample trainingExample3;
    trainingExample3.m_inputs = Matrix<double>(2, 1);
    trainingExample3.m_expectedOutputs = Matrix<double>(1, 1);
    Backpropagation::TrainingExample trainingExample4;
    trainingExample4.m_inputs = Matrix<double>(2, 1);
    trainingExample4.m_expectedOutputs = Matrix<double>(1, 1);

    trainingExample1.m_inputs(0, 0) = 0;
    trainingExample1.m_inputs(1, 0) = 0;
    trainingExample1.m_expectedOutputs(0, 0) = 0;
    trainingExample2.m_inputs(0, 0) = 0;
    trainingExample2.m_inputs(1, 0) = 1;
    trainingExample2.m_expectedOutputs(0, 0) = 1;
    trainingExample3.m_inputs(0, 0) = 1;
    trainingExample3.m_inputs(1, 0) = 0;
    trainingExample3.m_expectedOutputs(0, 0) = 1;
    trainingExample4.m_inputs(0, 0) = 1;
    trainingExample4.m_inputs(1, 0) = 1;
    trainingExample4.m_expectedOutputs(0, 0) = 0;
    Matrix<Backpropagation::TrainingExample> data(8, 1);
    data(0, 0) = trainingExample1;
    data(1, 0) = trainingExample2;
    data(2, 0) = trainingExample3;
    data(3, 0) = trainingExample4;
    data(4, 0) = trainingExample1;
    data(5, 0) = trainingExample2;
    data(6, 0) = trainingExample3;
    data(7, 0) = trainingExample4;

    Matrix<unsigned int> topology(3, 1);
    topology(0, 0) = 2;
    topology(1, 0) = 4;
    topology(2, 0) = 1;
    Network network(topology);

    Backpropagation backprop(network, data, 1, 4, 100, 4);
    Network::SaveToFile("/home/owen/neuralnet", network);

    return 0;
}

//No changes: Epoch 100 out of 100. Test cost is 0.197815
