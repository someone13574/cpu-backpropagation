#ifndef BACKPROPAGATION_HEADER
#define BACKPROPAGATION_HEADER

#include "Network.hpp"

class Backpropagation
{
public:
    struct TrainingExample
    {
        Matrix<double> m_inputs;
        Matrix<double> m_expectedOutputs;
    };
private:
    double m_learningRate;
    unsigned int m_batchSize;
    unsigned int m_numEpochs;

    Network& m_network;

    unsigned int m_numTraining;
    unsigned int m_numTesting;

    Matrix<TrainingExample> m_trainingExamples;
    Matrix<TrainingExample> m_testingExamples;

    Matrix<Matrix<double> > m_nabla_b;
    Matrix<Matrix<double> > m_nabla_w;
    Matrix<Matrix<double> > m_delta_nabla_b;
    Matrix<Matrix<double> > m_delta_nabla_w;

    Matrix<Matrix<double> > m_desiredActivation;
private:
    void ValidateParameters(Matrix<TrainingExample>& p_data);
    void SplitTrainingAndTestingData(Matrix<TrainingExample>& p_data);
    void ComputeEpoch(unsigned int epochNum);
    void ShuffleTrainingData();
    void ComputeBatch(unsigned int batchStartIndex);
public:
    Backpropagation(Network& p_network, Matrix<TrainingExample>& p_data, double learningRate, unsigned int batchSize, unsigned int numEpochs, unsigned int numTestingExamples);
};

#endif
