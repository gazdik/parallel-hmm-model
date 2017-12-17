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

    map<string, size_t> mapStateIndex;
    map<string, size_t> mapOutputIndex;
    float probability;

    size_t iState = 0;
    string fromState;
    string toState;

    // Read transition probabilities from the file
    while (its.good())
    {
        its >> fromState;
        if (mapStateIndex.count(fromState) == 0) {
            mStatesMapping[iState] = fromState;
            mapStateIndex[fromState] = iState++;
        }
        its >> toState;
        if (mapStateIndex.count(toState) == 0) {
            mStatesMapping[iState] = toState;
            mapStateIndex[toState] = iState++;
        }
        its >> probability;
        probability = log(probability);

        if (fromState == "#") {
            // Set an initial state probability
            mLogPi.at(mapStateIndex[toState]) = probability;
        }
        else {
            // Set an transition probability between states
            mLogA.at(mapStateIndex[fromState], mapStateIndex[toState]) = probability;
        }


        // TODO delete
        std::cout << "From : " << mapStateIndex[fromState]
                  << ", To: " << mapStateIndex[toState]
                  << ", Probability: " << mLogA.at(mapStateIndex[fromState], mapStateIndex[toState]) << std::endl;
    }

    ifstream ies(fileEmissions);
    string state;
    string output;
    size_t iOutput = 0;

    // Read emission probabilities from the file
    while (ies.good())
    {
        ies >> state;
        if (mapStateIndex.count(state) == 0) {
            mStatesMapping[iState] = state;
            mapStateIndex[state] = iState++;
        }
        ies >> output;
        if (mapOutputIndex.count(output) == 0) {
            mVocabularyMapping[iOutput] = output;
            mapOutputIndex[output] = iOutput++;
        }
        ies >> probability;
        probability = log(probability);

        mLogB.at(mapStateIndex[state], mapOutputIndex[output]) = probability;

        // TODO delete
        std::cout << "State : " << mapStateIndex[state]
                  << ", Output: " << mapOutputIndex[output]
                  << ", Probability: "
                  << mLogB.at(mapStateIndex[state], mapOutputIndex[output])
                  << std::endl;
    }

}

}
