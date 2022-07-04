#pragma once

#include <iostream>
#include <memory>
#include <type_traits>


template <typename T, typename = void>
class Matrix
{
private:
    T** m_data;
    bool m_is_allocated = false;

    unsigned int m_num_rows;
    unsigned int m_num_columns;

public:
    Matrix() = default;
    Matrix(unsigned int num_rows, unsigned int num_columns = 1) : m_num_rows(num_rows), m_num_columns(num_columns)
    {
        m_data = new T*[m_num_rows];
        for (unsigned int i = 0; i < m_num_rows; i++)
        {
            m_data[i] = new T[m_num_columns];
        }
        m_is_allocated = true;
    };
    Matrix(const Matrix<T>& obj)
    {
        m_num_rows = obj.GetRowCount();
        m_num_columns = obj.Get_Column_Count();

        if (m_is_allocated)
        {
            for (unsigned int ii = 0; ii < m_num_rows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T*[m_num_rows];
        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            m_data[ii] = new T[m_num_columns];
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
    };
    ~Matrix()
    {
        if (m_is_allocated)
        {
            for (unsigned int ii = 0; ii < m_num_rows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }
    };

    T& operator()(const unsigned int& row, const unsigned int& column)
    {
        if (row >= m_num_rows || column >= m_num_columns)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    const T& operator()(const unsigned int& row, const unsigned int& column) const
    {
        if (row >= m_num_rows || column >= m_num_columns)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    Matrix<T> operator()(const unsigned int& row)
    {
        Matrix<T> result(1, m_num_columns);
        for (unsigned int ii = 0; ii < m_num_columns; ii++)
        {
            result(0, ii) = m_data[row][ii];
        }

        return result;
    }

    void operator=(const Matrix<T>& obj)
    {
        m_num_rows = obj.GetRowCount();
        m_num_columns = obj.Get_Column_Count();

        if (m_is_allocated)
        {
            for (unsigned int ii = 0; ii < m_num_rows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T*[m_num_rows];
        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            m_data[ii] = new T[m_num_columns];
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
    }

    const unsigned int& Get_Row_Count() const
    {
        return m_num_rows;
    };
    const unsigned int& Get_Column_Count() const
    {
        return m_num_columns;
    };

    Matrix<T> Transpose()
    {
        Matrix result(m_num_columns, m_num_rows);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(jj, ii) = m_data[ii][jj];
            }
        }

        return result;
    };
};

template <typename T>
class Matrix<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
private:
    T** m_data;
    bool m_is_allocated = false;

    unsigned int m_num_rows;
    unsigned int m_num_columns;

public:
    Matrix() = default;
    Matrix(unsigned int num_rows, unsigned int num_columns = 1) : m_num_rows(num_rows), m_num_columns(num_columns)
    {
        m_data = new T*[m_num_rows];
        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            m_data[ii] = new T[m_num_columns];
        }
        m_is_allocated = true;
    };
    Matrix(const Matrix<T>& obj)
    {
        m_num_rows = obj.Get_Row_Count();
        m_num_columns = obj.Get_Column_Count();

        if (m_is_allocated)
        {
            for (unsigned int ii = 0; ii < m_num_rows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T*[m_num_rows];
        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            m_data[ii] = new T[m_num_columns];
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
        m_is_allocated = true;
    };
    ~Matrix()
    {
        if (m_is_allocated)
        {
            for (unsigned int ii = 0; ii < m_num_rows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }
    };

    T& operator()(const unsigned int& row, const unsigned int& column)
    {
        if (row >= m_num_rows || column >= m_num_columns)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    const T& operator()(const unsigned int& row, const unsigned int& column) const
    {
        if (row >= m_num_rows || column >= m_num_columns)
        {
            std::cout << "Index out of bounds!";
            exit(-1);
        }
        return m_data[row][column];
    };
    Matrix<T> operator()(const unsigned int& row)
    {
        Matrix<T> result(1, m_num_columns);
        for (unsigned int ii = 0; ii < m_num_columns; ii++)
        {
            result(0, ii) = m_data[row][ii];
        }

        return result;
    }

    void operator=(const Matrix<T>& obj)
    {
        m_num_rows = obj.Get_Row_Count();
        m_num_columns = obj.Get_Column_Count();

        if (m_is_allocated)
        {
            for (unsigned int ii = 0; ii < m_num_rows; ii++)
            {
                delete[] m_data[ii];
            }
            delete[] m_data;
        }

        m_data = new T*[m_num_rows];
        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            m_data[ii] = new T[m_num_columns];
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                m_data[ii][jj] = obj(ii, jj);
            }
        }
        m_is_allocated = true;
    }

    Matrix<T> operator+(const Matrix<T>& obj)
    {
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
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
            for (unsigned int jj = 0; jj < obj.Get_Column_Count(); jj++)
            {
                m_data[ii][jj] += obj(ii, jj);
            }
        }

        return *this;
    };
    Matrix<T> operator-(const Matrix<T>& obj)
    {
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
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
            for (unsigned int jj = 0; jj < obj.Get_Column_Count(); jj++)
            {
                m_data[ii][jj] -= obj(ii, jj);
            }
        }

        return *this;
    };
    Matrix<T> operator*(const Matrix<T>& obj)
    {
        Matrix<T> result(m_num_rows, obj.Get_Column_Count());

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < obj.Get_Column_Count(); jj++)
            {
                for (unsigned int kk = 0; kk < m_num_columns; kk++)
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
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(ii, jj) = m_data[ii][jj] + obj;
            }
        }

        return result;
    };
    Matrix<T> operator-(const T& obj)
    {
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(ii, jj) = m_data[ii][jj] - obj;
            }
        }

        return result;
    };
    Matrix<T> operator*(const T& obj)
    {
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(ii, jj) = m_data[ii][jj] * obj;
            }
        }

        return result;
    };
    Matrix<T> operator/(const T& obj)
    {
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(ii, jj) = m_data[ii][jj] / obj;
            }
        }

        return result;
    };

    Matrix<T> Hadamard_Product(const Matrix<T>& obj)
    {
        Matrix<T> result(m_num_rows, m_num_columns);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(ii, jj) = m_data[ii][jj] * obj(ii, jj);
            }
        }

        return result;
    };

    const unsigned int& Get_Row_Count() const
    {
        return m_num_rows;
    };
    const unsigned int& Get_Column_Count() const
    {
        return m_num_columns;
    };

    Matrix<T> Transpose()
    {
        Matrix result(m_num_columns, m_num_rows);

        for (unsigned int ii = 0; ii < m_num_rows; ii++)
        {
            for (unsigned int jj = 0; jj < m_num_columns; jj++)
            {
                result(jj, ii) = m_data[ii][jj];
            }
        }

        return result;
    };
};
