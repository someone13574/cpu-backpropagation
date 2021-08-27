#ifndef BACKPROPAGATION_SOURCE
#define BACKPROPAGATION_SOURCE

#include "Backpropagation.hpp"

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

Backpropagation::Backpropagation(Network& p_network, Matrix<TrainingExample>& p_data, double learningRate, unsigned int batchSize, unsigned int numEpochs, unsigned int numTestingExamples) : m_network(p_network), m_trainingExamples(p_data.GetRowCount() - numTestingExamples, 1), m_testingExamples(numTestingExamples, 1), m_learningRate(learningRate), m_batchSize(batchSize), m_numEpochs(numEpochs), m_numTraining(p_data.GetRowCount() - numTestingExamples), m_numTesting(numTestingExamples), m_nabla_b(p_network.GetLayerCount(), 1), m_nabla_w(p_network.GetLayerCount(), 1), m_delta_nabla_b(p_network.GetLayerCount(), 1), m_delta_nabla_w(p_network.GetLayerCount(), 1), m_desiredActivation(p_network.GetLayerCount(), 1)
{
    for (unsigned int ii = 0; ii < m_network.GetLayerCount(); ii++)
    {
        m_desiredActivation(ii, 0) = Matrix<double>(m_network.GetLayerSize(ii), 1);
        m_nabla_b(ii, 0) = Matrix<double>(m_network.GetLayerSize(ii), 1);
        m_delta_nabla_b(ii, 0) = Matrix<double>(m_network.GetLayerSize(ii), 1);;

        if (ii != 0)
        {
            m_nabla_w(ii, 0) = Matrix<double>(m_network.GetLayerSize(ii), m_network.GetLayerSize(ii - 1));
            m_delta_nabla_w(ii, 0) = Matrix<double>(m_network.GetLayerSize(ii), m_network.GetLayerSize(ii - 1));
        }
        else
        {
            m_nabla_w(ii, 0) = Matrix<double>(1, 1);
            m_delta_nabla_w(ii, 0) = Matrix<double>(1, 1);
        }
    }

    ValidateParameters(p_data);
    SplitTrainingAndTestingData(p_data);

    for (unsigned int ii = 0; ii < m_numEpochs; ii++)
    {
        ComputeEpoch(ii);
    }
}

void Backpropagation::ValidateParameters(Matrix<TrainingExample>& p_data)
{
    if (p_data.GetRowCount() != m_numTraining + m_numTesting)
    {
        std::cerr << "Parameter validation failed: The amount of data allocated to testing excides the amount of data provided!";
        exit(-1);
    }

    unsigned int numInputs = m_network.GetLayerSize(0);
    unsigned int numOutputs = m_network.GetLayerSize(m_network.GetLayerCount() - 1);

    for (unsigned int ii = 0; ii < p_data.GetRowCount(); ii++)
    {
        if (p_data(ii, 0).m_inputs.GetRowCount() != numInputs)
        {
            std::cerr << "Parameter validation failed: Training example " << ii << "had a different number of inputs than the provided neural network!";
            exit(-1);
        }
        if (p_data(ii, 0).m_expectedOutputs.GetRowCount() != numOutputs)
        {
            std::cerr << "Parameter validation failed: Training example " << ii << "had a different number of outputs than the provided neural network!";
            exit(-1);
        }
    }
}

void Backpropagation::SplitTrainingAndTestingData(Matrix<TrainingExample>& p_data)
{
    // srand(time(NULL));
    // for (unsigned int ii = p_data.GetRowCount() - 1; ii > 0; ii--)
    // {
    //     unsigned int random = rand() % (ii + 1);

    //     TrainingExample temp = p_data(random, 0);
    //     p_data(random, 0) = p_data(ii, 0);
    //     p_data(ii, 0) = temp;
    // }

    for (unsigned int ii = 0; ii < m_numTraining; ii++)
    {
        m_trainingExamples(ii, 0) = p_data(ii, 0);
    }
    for (unsigned int ii = m_numTraining; ii < m_numTraining + m_numTesting; ii++)
    {
        m_testingExamples(ii - m_numTraining, 0) = p_data(ii, 0);
    }
}

void Backpropagation::ComputeEpoch(unsigned int epochNum)
{
    //ShuffleTrainingData();
    for (unsigned int ii = 0; ii < m_numTraining; ii += m_batchSize)
    {
        ComputeBatch(ii);
    }
    double costSum = 0;
    for (unsigned int ii = 0; ii < m_numTesting; ii++)
    {
        TrainingExample& currentTestingExample = m_testingExamples(ii, 0);

        m_network.SetInputs(currentTestingExample.m_inputs);
        m_network.SetExpectedOutputs(currentTestingExample.m_expectedOutputs);

        m_network.FeedForward();

        costSum += m_network.ComputeMse();
    }
    std::cout << "Epoch " << epochNum + 1 << " out of " << m_numEpochs << ". Test cost is " << costSum / m_numTesting << ".\n";
}

void Backpropagation::ShuffleTrainingData()
{
    srand(time(NULL));
    for (unsigned int ii = m_trainingExamples.GetRowCount() - 1; ii > 0; ii--)
    {
        unsigned int random = rand() % (ii + 1);

        TrainingExample temp = m_trainingExamples(random, 0);
        m_trainingExamples(random, 0) = m_trainingExamples(ii, 0);
        m_trainingExamples(ii, 0) = temp;
    }
}

void Backpropagation::ComputeBatch(unsigned int batchStartIndex)
{
    for (unsigned int ii = 0; ii < m_network.GetLayerCount(); ii++)
    {
        for (unsigned int jj = 0; jj < m_nabla_b(ii, 0).GetRowCount(); jj++)
        {
            m_nabla_b(ii, 0)(jj, 0) = 0;
        }
        for (unsigned int jj = 0; jj < m_nabla_w(ii, 0).GetRowCount(); jj++)
        {
            for (unsigned int kk = 0; kk < m_nabla_w(ii, 0).GetColumnCount(); kk++)
            {
                m_nabla_w(ii, 0)(jj, kk) = 0;
            }
        }
    }

    unsigned int batchEndIndex = batchStartIndex + m_batchSize;
    for (unsigned int ii = batchStartIndex; ii < batchEndIndex; ii++)
    {
        for (unsigned int jj = 0; jj < m_network.GetLayerCount(); jj++)
        {
            for (unsigned int kk = 0; kk < m_network.GetLayerSize(jj); kk++)
            {
                m_desiredActivation(jj, 0)(kk, 0) = 0;
            }
        }

        TrainingExample& currentTrainingExample = m_trainingExamples(ii, 0);
        m_network.SetInputs(currentTrainingExample.m_inputs);
        m_network.SetExpectedOutputs(currentTrainingExample.m_expectedOutputs);
        m_desiredActivation(m_network.GetLayerCount() - 1, 0) = currentTrainingExample.m_expectedOutputs;

        m_network.FeedForward();

        for (unsigned int jj = m_network.GetLayerCount() - 1; jj > 0; jj--)
        {
            Matrix<double> costDerivative = (m_network.m_activations(jj, 0) - m_desiredActivation(jj, 0)) * 2;
            Matrix<double> zDerivative(m_network.GetLayerSize(jj), 1);
            for (unsigned int kk = 0; kk < zDerivative.GetRowCount(); kk++)
            {
                zDerivative(kk, 0) = m_network.m_activations(jj, 0)(kk, 0) * (1 - m_network.m_activations(jj, 0)(kk, 0));
            }

            Matrix<double> delta = costDerivative.HadamardProduct(zDerivative);

            m_delta_nabla_b(jj, 0) = delta;
            m_delta_nabla_w(jj, 0) = delta * m_network.m_activations(jj - 1 , 0).Transpose();

            for (unsigned int kk = 0; kk < m_network.GetLayerSize(jj); kk++)
            {
                m_desiredActivation(jj - 1, 0) = m_desiredActivation(jj - 1, 0) - (delta * m_network.m_weights(jj, 0)(kk)).Transpose();
            }
        }

        for (unsigned int jj = 0; jj < m_network.GetLayerCount(); jj++)
        {
            m_nabla_b(jj, 0) = m_nabla_b(jj, 0) - m_delta_nabla_b(jj, 0);
            m_nabla_w(jj, 0) = m_nabla_w(jj, 0) - m_delta_nabla_w(jj, 0);
        }
    }

    for (unsigned int ii = 0; ii < m_network.GetLayerCount(); ii++)
    {
        m_network.m_bias(ii, 0) = m_network.m_bias(ii, 0) + (m_nabla_b(ii, 0) * m_learningRate);
        m_network.m_weights(ii, 0) = m_network.m_weights(ii, 0) + (m_nabla_w(ii, 0) * m_learningRate);
    }
}

#endif
