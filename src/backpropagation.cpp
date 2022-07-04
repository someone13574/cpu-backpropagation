#include "backpropagation.hpp"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>


Backpropagation::Backpropagation(Network& network_ptr, Matrix<Training_Example>& data_ptr, double learning_rate, unsigned int batch_size, unsigned int num_epochs, unsigned int num_testing_examples)
    : m_learning_rate(learning_rate)
    , m_batch_size(batch_size)
    , m_num_epochs(num_epochs)
    , m_num_training(data_ptr.Get_Row_Count() - num_testing_examples)
    , m_num_testing(num_testing_examples)
    , m_training_examples(data_ptr.Get_Row_Count() - num_testing_examples, 1)
    , m_testing_examples(num_testing_examples, 1)
    , m_nabla_b(network_ptr.Get_Layer_Count(), 1)
    , m_nabla_w(network_ptr.Get_Layer_Count(), 1)
    , m_delta_nabla_b(network_ptr.Get_Layer_Count(), 1)
    , m_delta_nabla_w(network_ptr.Get_Layer_Count(), 1)
    , m_desired_activation(network_ptr.Get_Layer_Count(), 1)
    , m_network(network_ptr)
{
    for (unsigned int ii = 0; ii < m_network.Get_Layer_Count(); ii++)
    {
        m_desired_activation(ii, 0) = Matrix<double>(m_network.Get_Layer_Size(ii), 1);
        m_nabla_b(ii, 0) = Matrix<double>(m_network.Get_Layer_Size(ii), 1);
        m_delta_nabla_b(ii, 0) = Matrix<double>(m_network.Get_Layer_Size(ii), 1);
        ;

        if (ii != 0)
        {
            m_nabla_w(ii, 0) = Matrix<double>(m_network.Get_Layer_Size(ii), m_network.Get_Layer_Size(ii - 1));
            m_delta_nabla_w(ii, 0) = Matrix<double>(m_network.Get_Layer_Size(ii), m_network.Get_Layer_Size(ii - 1));
        }
        else
        {
            m_nabla_w(ii, 0) = Matrix<double>(1, 1);
            m_delta_nabla_w(ii, 0) = Matrix<double>(1, 1);
        }
    }

    Validate_Parameters(data_ptr);
    Split_Training_Data(data_ptr);

    for (unsigned int ii = 0; ii < m_num_epochs; ii++)
    {
        Compute_Epoch(ii);
    }
}

void Backpropagation::Validate_Parameters(Matrix<Training_Example>& data_ptr)
{
    if (data_ptr.Get_Row_Count() != m_num_training + m_num_testing)
    {
        std::cerr << "Parameter validation failed: The amount of data allocated to testing excides the amount of data provided!";
        exit(EXIT_FAILURE);
    }

    unsigned int num_inputs = m_network.Get_Layer_Size(0);
    unsigned int num_outputs = m_network.Get_Layer_Size(m_network.Get_Layer_Count() - 1);

    for (unsigned int ii = 0; ii < data_ptr.Get_Row_Count(); ii++)
    {
        if (data_ptr(ii, 0).m_inputs.Get_Row_Count() != num_inputs)
        {
            std::cerr << "Parameter validation failed: Training example " << ii << "had a different number of inputs than the provided neural network!";
            exit(EXIT_FAILURE);
        }
        if (data_ptr(ii, 0).m_expected_outputs.Get_Row_Count() != num_outputs)
        {
            std::cerr << "Parameter validation failed: Training example " << ii << "had a different number of outputs than the provided neural network!";
            exit(EXIT_FAILURE);
        }
    }
}

void Backpropagation::Split_Training_Data(Matrix<Training_Example>& data_ptr)
{
    for (unsigned int ii = 0; ii < m_num_training; ii++)
    {
        m_training_examples(ii, 0) = data_ptr(ii, 0);
    }
    for (unsigned int ii = m_num_training; ii < m_num_training + m_num_testing; ii++)
    {
        m_testing_examples(ii - m_num_training, 0) = data_ptr(ii, 0);
    }
}

void Backpropagation::Compute_Epoch(unsigned int epoch_num)
{
    // ShuffleTrainingData();
    for (unsigned int ii = 0; ii < m_num_training; ii += m_batch_size)
    {
        Compute_Batch(ii);
    }
    double costSum = 0;
    for (unsigned int ii = 0; ii < m_num_testing; ii++)
    {
        Training_Example& current_testing_example_ptr = m_testing_examples(ii, 0);

        m_network.Set_Inputs(current_testing_example_ptr.m_inputs);
        m_network.Set_Expected_Outputs(current_testing_example_ptr.m_expected_outputs);

        m_network.Feed_Forward();

        costSum += m_network.Compute_Mse();
    }
    std::cout << "Epoch " << epoch_num + 1 << " out of " << m_num_epochs << ". Test cost is " << costSum / m_num_testing << ".\n";
}

void Backpropagation::Shuffle_Training_Data()
{
    srand(time(NULL));
    for (unsigned int ii = m_training_examples.Get_Row_Count() - 1; ii > 0; ii--)
    {
        unsigned int random = rand() % (ii + 1);

        Training_Example temp = m_training_examples(random, 0);
        m_training_examples(random, 0) = m_training_examples(ii, 0);
        m_training_examples(ii, 0) = temp;
    }
}

void Backpropagation::Compute_Batch(unsigned int batch_start_index)
{
    for (unsigned int ii = 0; ii < m_network.Get_Layer_Count(); ii++)
    {
        for (unsigned int jj = 0; jj < m_nabla_b(ii, 0).Get_Row_Count(); jj++)
        {
            m_nabla_b(ii, 0)(jj, 0) = 0;
        }
        for (unsigned int jj = 0; jj < m_nabla_w(ii, 0).Get_Row_Count(); jj++)
        {
            for (unsigned int kk = 0; kk < m_nabla_w(ii, 0).Get_Column_Count(); kk++)
            {
                m_nabla_w(ii, 0)(jj, kk) = 0;
            }
        }
    }

    unsigned int batch_end_index = batch_start_index + m_batch_size;
    for (unsigned int ii = batch_start_index; ii < batch_end_index; ii++)
    {
        for (unsigned int jj = 0; jj < m_network.Get_Layer_Count(); jj++)
        {
            for (unsigned int kk = 0; kk < m_network.Get_Layer_Size(jj); kk++)
            {
                m_desired_activation(jj, 0)(kk, 0) = 0;
            }
        }

        Training_Example& current_training_example = m_training_examples(ii, 0);
        m_network.Set_Inputs(current_training_example.m_inputs);
        m_network.Set_Expected_Outputs(current_training_example.m_expected_outputs);
        m_desired_activation(m_network.Get_Layer_Count() - 1, 0) = current_training_example.m_expected_outputs;

        m_network.Feed_Forward();

        for (unsigned int jj = m_network.Get_Layer_Count() - 1; jj > 0; jj--)
        {
            Matrix<double> costDerivative = (m_network.m_activations(jj, 0) - m_desired_activation(jj, 0)) * 2;
            Matrix<double> zDerivative(m_network.Get_Layer_Size(jj), 1);
            for (unsigned int kk = 0; kk < zDerivative.Get_Row_Count(); kk++)
            {
                zDerivative(kk, 0) = m_network.m_activations(jj, 0)(kk, 0) * (1 - m_network.m_activations(jj, 0)(kk, 0));
            }

            Matrix<double> delta = costDerivative.Hadamard_Product(zDerivative);

            m_delta_nabla_b(jj, 0) = delta;
            m_delta_nabla_w(jj, 0) = delta * m_network.m_activations(jj - 1, 0).Transpose();

            for (unsigned int kk = 0; kk < m_network.Get_Layer_Size(jj); kk++)
            {
                m_desired_activation(jj - 1, 0) = m_desired_activation(jj - 1, 0) - (delta * m_network.m_weights(jj, 0)(kk)).Transpose();
            }
        }

        for (unsigned int jj = 0; jj < m_network.Get_Layer_Count(); jj++)
        {
            m_nabla_b(jj, 0) = m_nabla_b(jj, 0) - m_delta_nabla_b(jj, 0);
            m_nabla_w(jj, 0) = m_nabla_w(jj, 0) - m_delta_nabla_w(jj, 0);
        }
    }

    for (unsigned int ii = 0; ii < m_network.Get_Layer_Count(); ii++)
    {
        m_network.m_bias(ii, 0) = m_network.m_bias(ii, 0) + (m_nabla_b(ii, 0) * m_learning_rate);
        m_network.m_weights(ii, 0) = m_network.m_weights(ii, 0) + (m_nabla_w(ii, 0) * m_learning_rate);
    }
}
