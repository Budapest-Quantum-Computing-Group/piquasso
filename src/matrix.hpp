/*
 * Copyright 2021-2026 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <cstring>

#ifdef DEBUG
#include <iostream>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif


/**
 * The class for storing vectors.
 *
 * @param size The number of elements.
 * @param data The optional input data. If not provided, new data is allocated.
 */
template <typename TScalar>
class Vector
{
public:
    size_t length;
    TScalar *data;
    bool owner;       // True if data is owned by the instance, otherwise false.
    size_t *refcount; // Number of references

    HOST_DEVICE Vector(size_t length) : length(length)
    {
        data = new TScalar[length];
        owner = true;
        refcount = new size_t;
        (*refcount) = 1;
    }

    HOST_DEVICE Vector(size_t length, TScalar *data)
        : length(length), data(data)
    {
        owner = false;
        refcount = new size_t;
        (*refcount) = 1;
    }

    HOST_DEVICE Vector(const Vector &other)
        : length(other.length), data(other.data), owner(other.owner),
            refcount(other.refcount)
    {
        (*refcount)++;
    }

    HOST_DEVICE Vector()
        : length(0), data(nullptr), owner(false), refcount(nullptr) {}

    HOST_DEVICE void operator=(const Vector &other)
    {
        length = other.length;
        data = other.data;
        owner = other.owner;
        refcount = other.refcount;

        (*refcount)++;
    }

    HOST_DEVICE Vector copy()
    {
        Vector vector_copy(length);

        memcpy(vector_copy.data, data, length * sizeof(TScalar));

        return vector_copy;
    }

    HOST_DEVICE size_t size()
    {
        return length;
    }

    HOST_DEVICE TScalar sum()
    {
        TScalar result = TScalar{};

        for (std::size_t i = 0; i < length; i++) {
            result += data[i];
        }
        return result;
    }

    HOST_DEVICE ~Vector()
    {
        bool call_delete = ((*refcount) == 1);

        if (call_delete)
            delete refcount;
        else
            (*refcount)--;

        if (call_delete && owner)
            delete[] data;
    }

#ifdef DEBUG
    HOST_DEVICE void print()
    {
        std::cout << "The stored vector:\n";
        for (size_t idx = 0; idx < length; idx++)
            std::cout << " " << data[idx];

        std::cout << "\n------------------------\n";
    }
#endif

    HOST_DEVICE TScalar &operator[](size_t idx) const
    {
        return data[idx];
    }
};

/**
 * The class for storing matrices.
 *
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @param data The optional input data. If not provided, new data is allocated.
 */
template <typename TScalar>
class Matrix
{
public:
    size_t rows;
    size_t cols;
    size_t stride; // Column stride.
    TScalar *data;
    bool owner;       // True if data is owned by the instance, otherwise false.
    size_t *refcount; // Number of references

    HOST_DEVICE Matrix(size_t rows, size_t cols)
        : rows(rows), cols(cols), stride(cols)
    {
        data = new TScalar[rows * cols];
        owner = true;
        refcount = new size_t;
        (*refcount) = 1;
    }

    HOST_DEVICE Matrix(size_t rows, size_t cols, TScalar *data)
        : rows(rows), cols(cols), stride(cols), data(data)
    {
        owner = false;
        refcount = new size_t;
        (*refcount) = 1;
    }

    HOST_DEVICE Matrix(const Matrix &matrix)
        : rows(matrix.rows), cols(matrix.cols), stride(matrix.stride),
          data(matrix.data), owner(matrix.owner), refcount(matrix.refcount)
    {
        (*refcount)++;
    }

    HOST_DEVICE Matrix()
        : rows(0), cols(0), stride(0), data(nullptr), owner(false),
          refcount(nullptr) {}

    HOST_DEVICE void operator=(const Matrix &matrix)
    {
        rows = matrix.rows;
        cols = matrix.cols;
        stride = matrix.stride;
        data = matrix.data;
        owner = matrix.owner;
        refcount = matrix.refcount;

        (*refcount)++;
    }

    HOST_DEVICE size_t size()
    {
        return rows * cols;
    }

    HOST_DEVICE Matrix copy()
    {
        Matrix matrix_copy(rows, cols);

        memcpy(matrix_copy.data, data, size() * sizeof(TScalar));

        return matrix_copy;
    }

    HOST_DEVICE ~Matrix()
    {
        bool call_delete = ((*refcount) == 1);

        if (call_delete)
            delete refcount;
        else
            (*refcount)--;

        if (call_delete && owner)
            delete[] data;
    }

#ifdef DEBUG
    HOST_DEVICE void print()
    {
        std::cout << std::endl
                  << "The stored matrix:" << std::endl;
        for (size_t row_idx = 0; row_idx < rows; row_idx++)
        {
            for (size_t col_idx = 0; col_idx < cols; col_idx++)
            {
                size_t element_idx = row_idx * stride + col_idx;
                std::cout << " " << data[element_idx];
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }
#endif

    HOST_DEVICE TScalar &operator[](size_t idx) const
    {
        return data[idx];
    }

    HOST_DEVICE TScalar &operator()(size_t row, size_t col)
    {
        return data[row * stride + col];
    }
};

#endif
