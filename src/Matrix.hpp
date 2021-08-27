#ifndef MATRIX_HEADER
#define MATRIX_HEADER

#include <memory>
#include <type_traits>
#include <iostream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

template <typename T, typename = void>
class Matrix
{
    friend class boost::serialization::access;
private:
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                ar & m_data[ii][jj];
            }
        }
        ar & m_isAllocated;
        ar & m_numRows;
        ar & m_numCols;
    }
private:
    T **m_data;
    bool m_isAllocated = false;

    unsigned int m_numRows;
    unsigned int m_numCols;
public:
    Matrix() = default;
    Matrix(unsigned int numRows, unsigned int numCols = 1) : m_numRows(numRows), m_numCols(numCols)
    {
        m_data = new T *[m_numRows];
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            m_data[ii] = new T[m_numCols];
        }
        m_isAllocated = true;
    };
    Matrix(const Matrix<T>& obj)
    {
        m_numRows = obj.GetRowCount();
        m_numCols = obj.GetColumnCount();

        if (m_isAllocated)
        {
            for (unsigned int ii = 0; ii < m_numRows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T *[m_numRows];
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            m_data[ii] = new T[m_numCols];
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
    };
    ~Matrix()
    {
        if (m_isAllocated)
        {
            for (unsigned int ii = 0; ii < m_numRows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }
    };

    T& operator()(const unsigned int& row, const unsigned int& column)
    {
        if (row >= m_numRows || column >= m_numCols)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    const T& operator()(const unsigned int& row, const unsigned int& column) const
    {
        if (row >= m_numRows || column >= m_numCols)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    Matrix<T> operator()(const unsigned int& row)
    {
        Matrix<T> result(1, m_numCols);
        for (unsigned int ii = 0; ii < m_numCols; ii++)
        {
            result(0, ii) = m_data[row][ii];
        }

        return result;
    }

    void operator=(const Matrix<T>& obj)
    {
        m_numRows = obj.GetRowCount();
        m_numCols = obj.GetColumnCount();

        if (m_isAllocated)
        {
            for (unsigned int ii = 0; ii < m_numRows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T *[m_numRows];
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            m_data[ii] = new T[m_numCols];
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
    }

    const unsigned int& GetRowCount() const
    {
        return m_numRows;
    };
    const unsigned int& GetColumnCount() const
    {
        return m_numCols;
    };

    Matrix<T> Transpose()
    {
        Matrix result(m_numCols, m_numRows);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(jj, ii) = m_data[ii][jj];
            }
        }

        return result;
    };
};

template <typename T>
class Matrix <T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    friend class boost::serialization::access;
private:
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                ar & m_data[ii][jj];
            }
        }
        ar & m_isAllocated;
        ar & m_numRows;
        ar & m_numCols;
    }
private:
    T **m_data;
    bool m_isAllocated = false;

    unsigned int m_numRows;
    unsigned int m_numCols;
public:
    Matrix() = default;
    Matrix(unsigned int numRows, unsigned int numCols = 1) : m_numRows(numRows), m_numCols(numCols)
    {
        m_data = new T *[m_numRows];
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            m_data[ii] = new T[m_numCols];
        }
        m_isAllocated = true;
    };
    Matrix(const Matrix<T>& obj)
    {
        m_numRows = obj.GetRowCount();
        m_numCols = obj.GetColumnCount();

        if (m_isAllocated)
        {
            for (unsigned int ii = 0; ii < m_numRows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T *[m_numRows];
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            m_data[ii] = new T[m_numCols];
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
        m_isAllocated = true;
    };
    ~Matrix()
    {
        if (m_isAllocated)
        {
            for (unsigned int ii = 0; ii < m_numRows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }
    };

    T& operator()(const unsigned int& row, const unsigned int& column)
    {
        if (row >= m_numRows || column >= m_numCols)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    const T& operator()(const unsigned int& row, const unsigned int& column) const
    {
        if (row >= m_numRows || column >= m_numCols)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    Matrix<T> operator()(const unsigned int& row)
    {
        Matrix<T> result(1, m_numCols);
        for (unsigned int ii = 0; ii < m_numCols; ii++)
        {
            result(0, ii) = m_data[row][ii];
        }

        return result;
    }

    void operator=(const Matrix<T>& obj)
    {
        m_numRows = obj.GetRowCount();
        m_numCols = obj.GetColumnCount();

        if (m_isAllocated)
        {
            for (unsigned int ii = 0; ii < m_numRows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T *[m_numRows];
        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            m_data[ii] = new T[m_numCols];
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
        m_isAllocated = true;
    }

    Matrix<T> operator+(const Matrix<T>& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] + obj(ii, jj);
            }
        }

        return result;
    };
    Matrix<T>& operator+=(const Matrix<T>& obj)
    {
        for (unsigned int ii = 0; ii < obj.GetRowCount(); ii++)
        {
            for (unsigned int jj = 0; jj < obj.GetColumnCount(); jj++)
            {
                m_data[ii][jj] += obj(ii, jj);
            }
        }

        return *this;
    };
    Matrix<T> operator-(const Matrix<T>& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] - obj(ii, jj);
            }
        }

        return result;
    };
    Matrix<T>& operator-=(const Matrix<T>& obj)
    {
        for (unsigned int ii = 0; ii < obj.GetRowCount(); ii++)
        {
            for (unsigned int jj = 0; jj < obj.GetColumnCount(); jj++)
            {
                m_data[ii][jj] -= obj(ii, jj);
            }
        }

        return *this;
    };
    Matrix<T> operator*(const Matrix<T>& obj)
    {
        Matrix<T> result(m_numRows, obj.GetColumnCount());

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < obj.GetColumnCount(); jj++)
            {
                for (unsigned int kk = 0; kk < m_numCols; kk++)
                {
                    result(ii, jj) += m_data[ii][kk] * obj(kk, jj);
                }
            }
        }

        return result;
    };
    Matrix<T>& operator*=(const Matrix<T>& obj)
    {
        Matrix result = (*this) * obj;
        (*this) = result;
        return *this;
    };

    Matrix<T> operator+(const T& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] + obj;
            }
        }

        return result;
    };
    Matrix<T> operator-(const T& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] - obj;
            }
        }

        return result;
    };
    Matrix<T> operator*(const T& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] * obj;
            }
        }

        return result;
    };
    Matrix<T> operator/(const T& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] / obj;
            }
        }

        return result;
    };

    Matrix<T> HadamardProduct(const Matrix<T>& obj)
    {
        Matrix<T> result(m_numRows, m_numCols);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(ii, jj) = m_data[ii][jj] * obj(ii, jj);
            }
        }

        return result;
    };

    const unsigned int& GetRowCount() const
    {
        return m_numRows;
    };
    const unsigned int& GetColumnCount() const
    {
        return m_numCols;
    };

    Matrix<T> Transpose()
    {
        Matrix result(m_numCols, m_numRows);

        for (unsigned int ii = 0; ii < m_numRows; ii++)
        {
            for (unsigned int jj = 0; jj < m_numCols; jj++)
            {
                result(jj, ii) = m_data[ii][jj];
            }
        }

        return result;
    };
};

#endif
