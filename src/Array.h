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
#include <limits>
#include <cstring>
#include "Helpers.h"

namespace hmm
{

template<typename T>
class ArrayInterface
{
public:

    ArrayInterface()
    {
    }

    virtual ~ArrayInterface()
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

protected:
    T *mData = nullptr;
    std::size_t mNumAllocElements = 0;
    bool mFreeMemory = true;

    /*
     * Allocation step
     */
    static const std::size_t ALLOC_STEP = 100;

protected:

    void initializeData(std::size_t firstElement, std::size_t lastElement)
    {
        for (std::size_t i = firstElement; i <= lastElement; i++) {
            mData[i] = -std::numeric_limits<T>::infinity();
        }
    }
};

/**
 * Dynamically allocated 2D array.
 * @tparam T
 */
template <typename T>
class Array2D : public ArrayInterface<T>
{
public:

    Array2D(std::size_t numRows = 0, std::size_t numCols = 0) :
            mNumRows { numRows },
            mNumCols { numCols }
    {
        this->mNumAllocElements = mNumRows * mNumCols + this->ALLOC_STEP;
        this->mData = (T*) malloc(sizeof(T) * this->mNumAllocElements);
        this->initializeData(0, this->mNumAllocElements - 1);
    }


    Array2D(const Array2D<T> &array) :
        mNumRows { array.mNumRows },
        mNumCols { array.mNumCols }
    {
        this->mNumAllocElements = array.mNumAllocElements;
        this->mData = (T *) malloc(sizeof(T) * this->mNumAllocElements);

        std::memcpy(this->mData, array.mData, sizeof(T) * this->mNumAllocElements);
    }


    virtual ~Array2D() { }

    /**
     * Return a reference to the element at position (x,y) in the
     * 2D array.
     * @return Reference to the element at the specified position.
     */
    T &at(std::size_t x, std::size_t y)
    {
        if (x >= mNumRows || y >= mNumCols)
            reallocate(x + 1, y + 1);

        std::size_t i = index1D(x, y, mNumCols);
        return this->mData[i];
    }

    size_t getNumRows() const
    {
        return mNumRows;
    }

    size_t getNumCols() const
    {
        return mNumCols;
    }

    size_t getNumElements() const
    {
        return mNumRows * mNumCols;
    }

private:

    void reallocate(std::size_t numRows, std::size_t numCols)
    {
        mNumRows = numRows;
        mNumCols = numCols;

        std::size_t numElements = mNumRows * mNumCols;
        if (numElements < this->mNumAllocElements)
            return;

        std::size_t newNumElements = numElements + this->ALLOC_STEP;
        std::size_t prevNumElements = this->mNumAllocElements;

        this->mNumAllocElements = newNumElements;
        this->mData = (T *) std::realloc(this->mData, sizeof(T) * this->mNumAllocElements);

        this->initializeData(prevNumElements, newNumElements - 1);
    }

private:
    std::size_t mNumRows;
    std::size_t mNumCols;
};


/**
 * Dynamically allocated 2D array.
 * @tparam T
 */
template <typename T>
class Array1D : public ArrayInterface<T>
{
public:
    Array1D(std::size_t n = 0) :
            mNumElements { n }
    {
        this->mNumAllocElements = mNumElements + this->ALLOC_STEP;
        this->mData = (T*) malloc(sizeof(T) * this->mNumAllocElements);

        this->initializeData(0, this->mNumAllocElements - 1);
    }

    Array1D(const Array1D<T> &array) :
        mNumElements { array.mNumElements }
    {
        this->mNumAllocElements = array.mNumAllocElements;
        this->mData = (T *) malloc(sizeof(T) * this->mNumAllocElements);

        std::memcpy(this->mData, array.mData, sizeof(T) * this->mNumAllocElements);
    }

    ~Array1D() { }

    /**
     * Return a reference to the element at position (x,y) in the
     * 2D array.
     * @return Reference to the element at the specified position.
     */
    T &at(std::size_t n)
    {
        if (n >= mNumElements)
            reallocate(n + 1);

        return this->mData[n];
    }

    size_t getNumElements() const
    {
        return mNumElements;
    }

    void push(const T &value)
    {
        this->at(mNumElements) = value;
    }


private:

    /**
     * Allocate memory for a new element at position (x,y)
     */
    void reallocate(std::size_t numElements)
    {
        mNumElements = numElements;
        if (numElements < this->mNumAllocElements)
            return;

        std::size_t newNumElements = numElements + this->ALLOC_STEP;
        std::size_t prevNumElements = this->mNumAllocElements;

        this->mNumAllocElements = newNumElements;
        this->mData = (T*) std::realloc(this->mData, sizeof(T) * this->mNumAllocElements);

        this->initializeData(prevNumElements, newNumElements - 1);
    }

private:
    std::size_t mNumElements;
};

} // namespace hmm


#endif //GMU_ARRAY_H
