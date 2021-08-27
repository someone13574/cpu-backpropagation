#ifndef NETWORK_HEADER
#define NETWORK_HEADER

#include "Matrix.hpp"

class Network
{
    friend class Backpropagation;
    friend class boost::serialization::access;
private:
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_numLayers;
        ar & m_networkTopology;
        ar & m_bias;
        ar & m_weights;
        ar & m_zs;
        ar & m_activations;
        ar & m_expectedOutputs;
    }
private:
    unsigned int m_numLayers;
    Matrix<unsigned int> m_networkTopology;
    
    Matrix<Matrix<double> > m_bias;
    Matrix<Matrix<double> > m_weights;
    Matrix<Matrix<double> > m_zs;
    Matrix<Matrix<double> > m_activations;

    Matrix<double> m_expectedOutputs;
public:
    Network(Matrix<unsigned int> networkTopology);

    void FeedForward();
    double ComputeMse();

    void SetInputs(Matrix<double>& inputs);
    void SetExpectedOutputs(Matrix<double>& expectedOuputs);

    unsigned int GetLayerCount();
    unsigned int GetLayerSize(unsigned int layerIndex);

    static void SaveToFile(const char* path, const Network network);
};

#endif
