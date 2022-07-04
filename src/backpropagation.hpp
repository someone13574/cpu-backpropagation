#pragma once

#include "network.hpp"


class Backpropagation
{
public:
    struct Training_Example
    {
        Matrix<double> m_inputs;
        Matrix<double> m_expected_outputs;
    };

private:
    double m_learning_rate;
    unsigned int m_batch_size;
    unsigned int m_num_epochs;

    unsigned int m_num_training;
    unsigned int m_num_testing;
    Matrix<Training_Example> m_training_examples;
    Matrix<Training_Example> m_testing_examples;

    Matrix<Matrix<double>> m_nabla_b;
    Matrix<Matrix<double>> m_nabla_w;
    Matrix<Matrix<double>> m_delta_nabla_b;
    Matrix<Matrix<double>> m_delta_nabla_w;
    Matrix<Matrix<double>> m_desired_activation;

    Network& m_network;

private:
    void Validate_Parameters(Matrix<Training_Example>& data_ptr);
    void Split_Training_Data(Matrix<Training_Example>& data_ptr);
    void Compute_Epoch(unsigned int epoch_num);
    void Shuffle_Training_Data();
    void Compute_Batch(unsigned int batch_start_index);

public:
    Backpropagation(Network& network_ptr, Matrix<Training_Example>& data_ptr, double learning_rate, unsigned int batch_size, unsigned int num_epochs, unsigned int num_testing_examples);
};
