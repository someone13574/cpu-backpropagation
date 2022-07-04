#pragma once

#include "matrix.hpp"


class Network
{
    friend class Backpropagation;

private:
    unsigned int m_num_layers;
    Matrix<unsigned int> m_network_topology;

    Matrix<double> m_expected_outputs;
    Matrix<Matrix<double>> m_bias;
    Matrix<Matrix<double>> m_weights;
    Matrix<Matrix<double>> m_zs;
    Matrix<Matrix<double>> m_activations;

public:
    Network(Matrix<unsigned int> network_topology);

public:
    void Feed_Forward();
    double Compute_Mse();

public:
    void Set_Inputs(Matrix<double>& inputs);
    void Set_Expected_Outputs(Matrix<double>& expected_uputs);

    unsigned int Get_Layer_Count();
    unsigned int Get_Layer_Size(unsigned int layer_index);
};