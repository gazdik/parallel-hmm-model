/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#include <fstream>
#include <iostream>
#include <set>
#include <cmath>
#include <sstream>
#include <vector>
#include <random>
#include "HiddenMarkovModel.h"

using namespace std;

namespace hmm
{

HiddenMarkovModel::HiddenMarkovModel(const std::string &fileTransitions,
                                     const std::string &fileEmissions)
{
    loadSymbols(fileTransitions, fileEmissions);

    mLogA = new Array2D<float>((uint32_t) mStateToIndexMap.size(),
                        (uint32_t) mStateToIndexMap.size());
    mLogB = new Array2D<float>((uint32_t) mStateToIndexMap.size(),
                        (uint32_t) mOutputToIndexMap.size());
    mLogPi = new Array1D<float>((uint32_t) mStateToIndexMap.size());

    loadModel(fileTransitions, fileEmissions);
}

HiddenMarkovModel::HiddenMarkovModel(size_t numStates, size_t numSymbols)
{
    mLogA = new Array2D<float>((uint32_t) numStates, (uint32_t) numStates);
    mLogB = new Array2D<float>((uint32_t) numStates, (uint32_t) numSymbols);
    mLogPi = new Array1D<float>((uint32_t) numStates);

    randomInit();
}


uint32_t HiddenMarkovModel::getNumStates() const
{
    return mLogA->getNumRows();
}

uint32_t HiddenMarkovModel::getNumSymbols() const
{
    return mLogB->getNumCols();
}

void HiddenMarkovModel::loadModel(
        const std::string &fileTransitions,
        const std::string &fileEmissions)
{
    ifstream ifs(fileTransitions);

    float probability;
    string fromState;
    string toState;

    // Read transition probabilities from the file
    while (ifs.good())
    {
        ifs >> fromState;
        ifs >> toState;
        ifs >> probability;
        probability = convertToLog(probability);

        if (probability != probability) {
            cerr << "Probability is equal to NAN" << endl;
        }

        // Set an initial state probability
        if (fromState == "#") {
            mLogPi->at(mStateToIndexMap[toState]) = probability;
        }
        // Set a transition probability between states
        else {
            mLogA->at(mStateToIndexMap[fromState], mStateToIndexMap[toState]) = probability;
        }
    }

    ifs.close();
    ifs.open(fileEmissions);
    string state;
    string output;

    // Read emission probabilities from the file
    while (ifs.good())
    {
        ifs >> state;
        ifs >> output;
        ifs >> probability;
        probability = convertToLog(probability);
        if (probability != probability) {
            cerr << "Probability is equal to NAN" << endl;
        }

        mLogB->at(mStateToIndexMap[state], mOutputToIndexMap[output]) = probability;
    }

//    debugPrint();
}

void HiddenMarkovModel::debugPrint()
{
    // Print transition matrix
    printf("Transition probability matrix A:\n");
    for (uint32_t x = 0; x < mLogA->getNumRows(); x++) {
        for (uint32_t y = 0; y < mLogA->getNumCols(); y++) {
            printf("%+6.2f ", mLogA->at(x, y));
        }
        printf("\n");
    }

    printf("Emission probability matrix B:\n");
    for (uint32_t x = 0; x < mLogB->getNumRows(); x++) {
        for (uint32_t y = 0; y < mLogB->getNumCols(); y++) {
            printf("%+6.2f ", mLogB->at(x, y));
        }
        printf("\n");
    }

    printf("Initial probability vector PI:\n");
    for (uint32_t i = 0; i < mLogPi->size(); i++) {
        printf("%+6.2f ", mLogPi->at(i));
    }
    printf("\n");
}

std::vector<uint32_t>
HiddenMarkovModel::translateObservation(const std::string &strObservation)
{
    vector<unsigned> observation;

    stringstream ss(strObservation);
    string word;

    while (ss >> word) {
        observation.push_back(mOutputToIndexMap[word]);
    }

    return observation;
}

void HiddenMarkovModel::loadSymbols(const std::string &fileTransitions,
                                    const std::string &fileEmissions)
{
    ifstream ifs(fileTransitions);
    uint32_t iState = 0;
    float probability;
    string fromState;
    string toState;

    // Load states
    while (ifs.good()) {
        ifs >> fromState;
        ifs >> toState;
        ifs >> probability;

        if (fromState != "#") {
            if (mStateToIndexMap.count(fromState) == 0) {
                mStateToIndexMap[fromState] = iState;
                mIndexToStateMap[iState++] = fromState;
            }
        }

        if (mStateToIndexMap.count(toState) == 0) {
            mStateToIndexMap[toState] = iState;
            mIndexToStateMap[iState++] = toState;
        }
    }

    ifs.close();
    ifs.open(fileEmissions);
    string output;
    uint32_t iOutput = 0;

    // Load output symbols
    while (ifs.good()) {
        ifs >> fromState;
        ifs >> output;
        ifs >> probability;

        if (mStateToIndexMap.count(fromState) == 0) {
            mStateToIndexMap[fromState] = iState;
            mIndexToStateMap[iState++] = fromState;
        }

        if (mOutputToIndexMap.count(output) == 0) {
            mOutputToIndexMap[output] = iOutput;
            mIndexToOutputMap[iOutput++] = output;
        }
    }
}

std::string
HiddenMarkovModel::translateStateSequence(const std::vector<uint32_t> &sequence)
{
    string strSequence;
    for (auto sIndex: sequence) {
        strSequence += mIndexToStateMap[sIndex] + " ";
    }

    return strSequence;
}

void HiddenMarkovModel::randomInit()
{
    cout << "Initialiazing HMM..." << endl;

    default_random_engine generator(56464646565566);
    uniform_real_distribution<float> distribution;

    float p;
    float sum = mLogPi->size() * 0.5f;
    for (uint32_t i = 0; i < mLogPi->size(); i++) {
        p = distribution(generator);
        mLogPi->at(i) = log(p / sum);
    }

    sum = mLogA->getNumCols() * 0.5f;
    for (uint32_t i = 0; i < mLogA->getNumRows(); i++) {
        for (uint32_t j = 0; j < mLogA->getNumCols(); j++) {
            p = distribution(generator);
            mLogA->at(i, j) = log(p/sum);
        }
    }

    sum = mLogB->getNumCols() * 0.5f;
    for (uint32_t i = 0; i < mLogB->getNumRows(); i++) {
        for (uint32_t j = 0; j < mLogB->getNumCols(); j++) {
            p = distribution(generator);
            mLogB->at(i, j) = log(p / sum);
        }
    }

    cout << "HMM has been initialized" << endl;
}

}
