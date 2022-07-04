#include "network.hpp"

#include <cmath>
#include <ctime>
#include <fstream>
#include <random>


Network::Network(Matrix<unsigned int> network_topology)
    : m_num_layers(network_topology.Get_Row_Count())
    , m_network_topology(network_topology)
    , m_expected_outputs(network_topology(m_num_layers - 1, 0), 1)
    , m_bias(m_num_layers, 1)
    , m_weights(m_num_layers, 1)
    , m_zs(m_num_layers, 1)
    , m_activations(m_num_layers, 1)
{
    for (unsigned int ii = 0; ii < m_num_layers; ii++)
    {
        m_zs(ii, 0) = Matrix<double>(m_network_topology(ii, 0), 1);
        m_activations(ii, 0) = Matrix<double>(m_network_topology(ii, 0), 1);
        m_bias(ii, 0) = Matrix<double>(m_network_topology(ii, 0), 1);

        for (unsigned int jj = 0; jj < m_network_topology(ii, 0); jj++)
        {
            m_bias(ii, 0)(jj, 0) = 0;
        }

        std::default_random_engine default_random_engine(3);
        std::normal_distribution<double> standard_normal_distribution(0.0, 1.0);

        if (ii == 0)
        {
            m_weights(0, 0) = Matrix<double>(1, 1);
            m_weights(0, 0)(0, 0) = 1;
        }
        else
        {
            m_weights(ii, 0) = Matrix<double>(m_network_topology(ii, 0), m_network_topology(ii - 1, 0));
            for (unsigned int jj = 0; jj < m_weights(ii, 0).Get_Row_Count(); jj++)
            {
                for (unsigned int kk = 0; kk < m_weights(ii, 0).Get_Column_Count(); kk++)
                {
                    m_weights(ii, 0)(jj, kk) = standard_normal_distribution(default_random_engine) * std::sqrt(1.0 / (double)m_network_topology(ii - 1, 0));
                }
            }
        }
    }
}

void Network::Feed_Forward()
{
    for (unsigned int ii = 1; ii < m_num_layers; ii++)
    {
        m_zs(ii, 0) = (m_weights(ii, 0) * m_activations(ii - 1, 0)) + m_bias(ii, 0);

        for (unsigned int jj = 0; jj < Get_Layer_Size(ii); jj++)
        {
            m_activations(ii, 0)(jj, 0) = 1.0 / (1.0 + std::exp(-m_zs(ii, 0)(jj, 0)));
        }
    }
}

double Network::Compute_Mse()
{
    Matrix<double> sqaured_error = m_expected_outputs - m_activations(Get_Layer_Count() - 1, 0);
    sqaured_error = sqaured_error * sqaured_error;

    double total = 0;
    for (unsigned int ii = 0; ii < Get_Layer_Size(Get_Layer_Count() - 1); ii++)
    {
        total += sqaured_error(ii, 0);
    }

    return total / Get_Layer_Size(Get_Layer_Count() - 1);
}

void Network::Set_Inputs(Matrix<double>& inputs)
{
    m_activations(0, 0) = inputs;
}

void Network::Set_Expected_Outputs(Matrix<double>& expected_outputs)
{
    m_expected_outputs = expected_outputs;
}

unsigned int Network::Get_Layer_Count()
{
    return m_network_topology.Get_Row_Count();
}

unsigned int Network::Get_Layer_Size(unsigned int layer_index)
{
    return m_network_topology(layer_index, 0);
}
