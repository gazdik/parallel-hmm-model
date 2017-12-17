/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "ForwardAlgorithm.h"
#include <cmath>
#include <assert.h>
#include <iostream>

using namespace std;

namespace hmm
{

ForwardAlgorithm::ForwardAlgorithm(hmm::HiddenMarkovModel &hmm)
        : HMMAlgorithm(hmm)
{}

ForwardAlgorithmCPU::ForwardAlgorithmCPU(hmm::HiddenMarkovModel &hmm)
        : ForwardAlgorithm(hmm)
{}

float ForwardAlgorithmCPU::evaluate(Array1D<unsigned int> &observation)
{
    float alpha = forward(observation);
    float beta = backward(observation);

    assert(alpha + beta < 1e-5);
    return alpha;
}

Array2D<float> ForwardAlgorithmCPU::getAlpha(Array1D<unsigned int> &observation)
{
    int observationCount = observation.getNumElements();
    int stateCount = mHmm.getNumStates();
    Array2D<float> mat(stateCount, observationCount);

    // Initialiation
    // for each state
    for (int i = 0; i < stateCount; i++) {
        mat.at(i, 0) = mHmm.mLogPi.at(i) + mHmm.mLogB.at(i, observation.at(0));
    }

    float temp;
    // for each observation
    for (int t = 1; t < observationCount; t++) {
        // for each HMM state
        for (int j = 0; j < mHmm.getNumStates(); j++) {
            for (int i = 0; i < mHmm.getNumStates(); i++) {
                temp = mat.at(i, t - 1) + mHmm.mLogA.at(i, j) +
                       mHmm.mLogB.at(j, observation.at(t));
                mat.at(j, t) = logAdd(mat.at(j, t), temp);
            }
        }
    }

    return mat;
}


float ForwardAlgorithmCPU::forward(Array1D<unsigned int> &observation)
{
    Array2D<float> alpha = getAlpha(observation);
    int observationCount = observation.getNumElements();
    int stateCount = mHmm.getNumStates();

    float logLikelyhood = -INFINITY;
    for (int i = 0; i < stateCount; i++) {
        logLikelyhood = logAdd(logLikelyhood, alpha.at(i, observationCount - 1));
    }

    return logLikelyhood;
}

Array2D<float> ForwardAlgorithmCPU::getBeta(Array1D<unsigned int> &observation)
{
    int observationCount = observation.getNumElements();
    int stateCount = mHmm.getNumStates();
    Array2D<float> mat(stateCount, observationCount);

    // Initialiation
    // for each state
    for (int i = 0; i < stateCount; i++) {
        // 0 is log(1)
        mat.at(i, observationCount - 1) = 0;
    }

    float temp;
    // for each observation
    for (int t = observationCount - 2; t >= 0; t--) {
        // for each HMM state
        for (int i = 0; i < stateCount; i++) {
            for (int j = 0; j < stateCount; j++) {
                temp = mat.at(j, t + 1) + mHmm.mLogA.at(i, j) +
                       mHmm.mLogB.at(j, observation.at(t + 1));
                mat.at(i, t) = logAdd(mat.at(i, t), temp);
            }
        }
    }

    return mat;

}

float ForwardAlgorithmCPU::backward(Array1D<unsigned int> &observation)
{
    Array2D<float> beta = getBeta(observation);
    int stateCount = mHmm.getNumStates();

    float logLikelyhood = -INFINITY;
    for (int i = 0; i < stateCount; i++) {
        logLikelyhood = logAdd(logLikelyhood, beta.at(i, 0) +
                                              mHmm.mLogB.at(i,
                                                            observation.at(0)) +
                                              mHmm.mLogPi.at(i));
    }

    return logLikelyhood;
}

float ForwardAlgorithmCPU::logAdd(float x, float y)
{
    float result = 0;
    if (x == -INFINITY && y == -INFINITY) {
        // just return -INFINITY
        return x;
    }
    if (x >= y) {
        result = x + log1p(exp(y - x));
    } else {
        result = y + log1p(exp(x - y));
    }

    return result;
}

ForwardAlgorithmGPU::ForwardAlgorithmGPU(hmm::HiddenMarkovModel &hmm)
        : ForwardAlgorithm(hmm)
{}

float ForwardAlgorithmGPU::evaluate(Array1D<unsigned> &observation)
{
    return ForwardAlgorithm::evaluate(observation);
}

} // namespace hmm
