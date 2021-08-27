#ifndef NETWORK_SOURCE
#define NETWORK_SOURCE

#include "Network.hpp"

#include <math.h>
#include <random>
#include <time.h>

#include <fstream>

Network::Network(Matrix<unsigned int> networkTopology) : m_networkTopology(networkTopology), m_numLayers(networkTopology.GetRowCount()), m_zs(m_numLayers, 1), m_activations(m_numLayers, 1), m_bias(m_numLayers, 1), m_weights(m_numLayers, 1), m_expectedOutputs(m_networkTopology(m_numLayers - 1, 0), 1)
{
    for (unsigned int ii = 0; ii < m_numLayers; ii++)
    {
        m_zs(ii, 0) = Matrix<double>(m_networkTopology(ii, 0), 1);
        m_activations(ii, 0) = Matrix<double>(m_networkTopology(ii, 0), 1);
        m_bias(ii, 0) = Matrix<double>(m_networkTopology(ii, 0), 1);

        for (unsigned int jj = 0; jj < m_networkTopology(ii, 0); jj++)
        {
            m_bias(ii, 0)(jj, 0) = 0;
        }

        std::default_random_engine defaultEngine(3);
        std::normal_distribution<double> standardNormalDistribution(0.0, 1.0);

        if (ii == 0)
        {
            m_weights(0, 0) = Matrix<double>(1, 1);
            m_weights(0, 0)(0, 0) = 1;
        }
        else
        {
            m_weights(ii, 0) = Matrix<double>(m_networkTopology(ii, 0), m_networkTopology(ii - 1, 0));
            for (unsigned int jj = 0; jj < m_weights(ii, 0).GetRowCount(); jj++)
            {
                for (unsigned int kk = 0; kk < m_weights(ii, 0).GetColumnCount(); kk++)
                {
                    m_weights(ii, 0)(jj, kk) = standardNormalDistribution(defaultEngine) * std::sqrt(1.0 / (double)m_networkTopology(ii - 1, 0));
                }
            }
        }
    }
}

void Network::FeedForward()
{
    for (unsigned int ii = 1; ii < m_numLayers; ii++) 
    {
        m_zs(ii, 0) = (m_weights(ii, 0) * m_activations(ii - 1, 0)) + m_bias(ii, 0);

        for (unsigned int jj = 0; jj < GetLayerSize(ii); jj++)
        {
            m_activations(ii, 0)(jj, 0) = 1.0 / (1.0 + std::exp(-m_zs(ii, 0)(jj, 0)));
        }
    }
}

double Network::ComputeMse()
{
    Matrix<double> sqauredError = m_expectedOutputs - m_activations(GetLayerCount() - 1, 0);
    sqauredError = sqauredError * sqauredError;

    double total = 0;
    for (unsigned int ii = 0; ii < GetLayerSize(GetLayerCount() - 1); ii++)
    {
        total += sqauredError(ii, 0);
    }
    
    return total / GetLayerSize(GetLayerCount() - 1);
}

void Network::SetInputs(Matrix<double>& inputs)
{
    m_activations(0, 0) = inputs;
}

void Network::SetExpectedOutputs(Matrix<double>& expectedOutputs)
{
    m_expectedOutputs = expectedOutputs;
}

unsigned int Network::GetLayerCount()
{
    return m_networkTopology.GetRowCount();
}

unsigned int Network::GetLayerSize(unsigned int layerIndex)
{
    return m_networkTopology(layerIndex, 0);
}

void Network::SaveToFile(const char* path, const Network network)
{
    std::ofstream file(path);

    boost::archive::text_oarchive save(file);
    save << network;
}

#endif
