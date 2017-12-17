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
#include "HiddenMarkovModel.h"

using namespace std;

namespace hmm
{

HiddenMarkovModel::HiddenMarkovModel(const std::string &fileTransitions,
                                     const std::string &fileEmissions)
{
    loadFromFile(fileTransitions, fileEmissions);
}

size_t HiddenMarkovModel::getNumStates() const
{
    return mLogA.getNumRows();
}

size_t HiddenMarkovModel::getNumSymbols() const
{
    return mLogB.getNumCols();
}

void HiddenMarkovModel::loadFromFile(
        const std::string &fileTransitions,
        const std::string &fileEmissions)
{
    ifstream its(fileTransitions);

    float probability;

    size_t iState = 0;
    string fromState;
    string toState;

    // Read transition probabilities from the file
    while (its.good())
    {
        its >> fromState;
        if (fromState != "#") {
            if (mStateToIndexMap.count(fromState) == 0) {
                mIndexToStateMap[iState] = fromState;
                mStateToIndexMap[fromState] = iState++;
            }
        }
        its >> toState;
        if (mStateToIndexMap.count(toState) == 0) {
            mIndexToStateMap[iState] = toState;
            mStateToIndexMap[toState] = iState++;
        }
        its >> probability;
        probability = convertToLog(probability);

        if (fromState == "#") {
            // Set an initial state probability
            mLogPi.at(mStateToIndexMap[toState]) = probability;
        }
        else {
            // Set an transition probability between states
            mLogA.at(mStateToIndexMap[fromState], mStateToIndexMap[toState]) = probability;
        }

        // TODO delete
//        std::cout << "From : " << mStateToIndexMap[fromState]
//                  << ", To: " << mStateToIndexMap[toState]
//                  << ", Probability: " << mLogA.at(mStateToIndexMap[fromState], mStateToIndexMap[toState]) << std::endl;
    }

    ifstream ies(fileEmissions);
    string state;
    string output;
    size_t iOutput = 0;

    // Read emission probabilities from the file
    while (ies.good())
    {
        ies >> state;
        if (mStateToIndexMap.count(state) == 0) {
            mIndexToStateMap[iState] = state;
            mStateToIndexMap[state] = iState++;
        }
        ies >> output;
        if (mOutputToIndexMap.count(output) == 0) {
            mIndexToOutputMap[iOutput] = output;
            mOutputToIndexMap[output] = iOutput++;
        }
        ies >> probability;
        probability = convertToLog(probability);

        mLogB.at(mStateToIndexMap[state], mOutputToIndexMap[output]) = probability;

        // TODO delete
//        std::cout << "State : " << mStateToIndexMap[state]
//                  << ", Output: " << mOutputToIndexMap[output]
//                  << ", Probability: "
//                  << mLogB.at(mStateToIndexMap[state], mOutputToIndexMap[output])
//                  << std::endl;
    }

    debugPrint();
}

void HiddenMarkovModel::debugPrint()
{
    // Print transition matrix
    printf("Transition probability matrix A:\n");
    for (size_t x = 0; x < mLogA.getNumRows(); x++) {
        for (size_t y = 0; y < mLogA.getNumCols(); y++) {
            printf("%+6.2f ", mLogA.at(x, y));
        }
        printf("\n");
    }

    printf("Emission probability matrix B:\n");
    for (size_t x = 0; x < mLogB.getNumRows(); x++) {
        for (size_t y = 0; y < mLogB.getNumCols(); y++) {
            printf("%+6.2f ", mLogB.at(x, y));
        }
        printf("\n");
    }

    printf("Initial probability vector PI:\n");
    for (size_t i = 0; i < mLogPi.getNumElements(); i++) {
        printf("%+6.2f ", mLogPi.at(i));
    }
    printf("\n");
}

}
