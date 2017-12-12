/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#include "HiddenMarkovModel.h"

unsigned int HiddenMarkovModel::getNumStates() const
{
    return mNumStates;
}

unsigned int HiddenMarkovModel::getNumSymbols() const
{
    return mNumSymbols;
}

float *HiddenMarkovModel::getLogA() const
{
    return mLogA;
}

float *HiddenMarkovModel::getLogB() const
{
    return mLogB;
}

float *HiddenMarkovModel::getLogPi() const
{
    return mLogPi;
}

HiddenMarkovModel::HiddenMarkovModel(const std::string &fileTransitions,
                                     const std::string &fileEmissions)
{

}

HiddenMarkovModel::HiddenMarkovModel(unsigned numStates, unsigned numSymbols)
{

}

HiddenMarkovModel::HiddenMarkovModel(float *A, float *B, float *pi)
{

}
