/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        12/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_ARRAY_H
#define GMU_ARRAY_H

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <cstring>
#include <iostream>
#include "Helpers.h"

namespace hmm
{

/**
 * 1D array.
 * @tparam T
 */
template <typename T>
class Array1D
{
public:
    Array1D(std::uint32_t n) :
            mCount { n }
    {
        mData = new T[mCount];
        this->initializeData(0, mCount - 1);
    }

    Array1D(const Array1D<T> &o) :
        mCount { o.mCount },
        mFreeMemory { o.mFreeMemory }
    {
        mData = new T[mCount];
        std::memcpy(mData, o.mData, sizeof(T) * mCount);
    }

    Array1D(Array1D<T> && o) :
            mCount { o.mCount },
            mFreeMemory { o.mFreeMemory }
    {
        mData = o.mData;
        o.mFreeMemory = false;
    }

    virtual ~Array1D()
    {
        if (mFreeMemory)
            free(mData);
    }

    /**
     * Return a pointer to a raw data.
     * It's up to a caller to delete this data.
     * @return
     */
    T *getData()
    {
        mFreeMemory = false;
        return mData;
    }

    /**
     * Return a reference to the element at position (x,y) in the
     * 2D array.
     * @return Reference to the element at the specified position.
     */
    T &at(std::uint32_t n)
    {
        if (n >= mCount) {
            throw std::out_of_range("Array::at(n): n = " + std::to_string(n)
                                    + " >= size = " + std::to_string(mCount));
        }

        return mData[n];
    }

    /**
     * Return the number of elements in the container.
     * @return
     */
    std::uint32_t size() const
    {
        return mCount;
    }

protected:
    T *mData = nullptr;
    std::uint32_t mCount;
    bool mFreeMemory = true;

    void initializeData(std::uint32_t firstElement, std::uint32_t lastElement)
    {
        for (std::uint32_t i = firstElement; i <= lastElement; i++) {
            mData[i] = -std::numeric_limits<T>::infinity();
        }
    }
};

/**
 * 2D Array
 * @tparam T
 */
template <typename T>
class Array2D : public Array1D<T>
{
public:

    Array2D(std::uint32_t numRows, std::uint32_t numCols) :
            Array1D<T>( numRows * numCols),
            mNumRows { numRows },
            mNumCols { numCols }
    {
    }

    Array2D(const Array2D<T> &o) :
            Array1D<T>(o),
            mNumRows { o.mNumRows },
            mNumCols { o.mNumCols }
    {
    }

    Array2D(Array2D<T> &&o) :
            Array1D<T>(o),
            mNumRows { o.mNumRows },
            mNumCols { o.mNumCols}
    {

    }

    virtual ~Array2D() { }

    /**
     * Return a reference to the element at position (x,y) in the
     * 2D array.
     * @return Reference to the element at the specified position.
     */
    T &at(std::uint32_t x, std::uint32_t y)
    {
        std::uint32_t n = index1D(x, y, mNumCols);

        return Array1D<T>::at(n);
    }

    std::uint32_t getNumRows() const
    {
        return mNumRows;
    }

    std::uint32_t getNumCols() const
    {
        return mNumCols;
    }

private:
    std::uint32_t mNumRows;
    std::uint32_t mNumCols;
};



} // namespace hmm


#endif //GMU_ARRAY_H
