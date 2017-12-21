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

float ForwardAlgorithm::evaluate(const std::string &strObservation)
{
    auto observation = mHmm.translateObservation(strObservation);
    return evaluate(observation);
}

ForwardAlgorithmCPU::ForwardAlgorithmCPU(hmm::HiddenMarkovModel &hmm)
        : ForwardAlgorithm(hmm)
{}

float ForwardAlgorithmCPU::evaluate(vector<uint32_t> &observation)
{
    float alpha = forward(observation);
//    float beta = backward(observation);

//    if (!(alpha + beta < 1e-5)) {
//        cerr << "ForwardAlgorithm: alpha and beta are different." << endl
//             << "alpha: " << alpha << endl
//             << "beta:  " << beta << endl;
//    }

    return alpha;
}

Array2D<float> ForwardAlgorithmCPU::getAlpha(vector<uint32_t> &observation)
{
    uint32_t observationCount = (uint32_t) observation.size();
    uint32_t stateCount = mHmm.getNumStates();
    Array2D<float> mat(stateCount, observationCount);

    // Initialiation
    // for each state
    for (int i = 0; i < stateCount; i++) {
        mat.at(i, 0) = mHmm.mLogPi->at(i) + mHmm.mLogB->at(i, observation.at(0));
    }

    float temp;
    // for each observation
    for (uint32_t t = 1; t < observationCount; t++) {
        // for each HMM state
        for (uint32_t j = 0; j < mHmm.getNumStates(); j++) {
            for (uint32_t i = 0; i < mHmm.getNumStates(); i++) {
                temp = mat.at(i, t - 1) + mHmm.mLogA->at(i, j) +
                       mHmm.mLogB->at(j, observation.at(t));
                mat.at(j, t) = logAdd(mat.at(j, t), temp);
            }
        }
    }

    return mat;
}


float ForwardAlgorithmCPU::forward(vector<uint32_t> &observation)
{
    Array2D<float> alpha = getAlpha(observation);
    uint32_t observationCount = (uint32_t) observation.size();
    uint32_t stateCount = mHmm.getNumStates();

    float logLikelyhood = -INFINITY;
    for (int i = 0; i < stateCount; i++) {
        logLikelyhood = logAdd(logLikelyhood, alpha.at(i, observationCount - 1));
    }

    return logLikelyhood;
}

Array2D<float> ForwardAlgorithmCPU::getBeta(vector<uint32_t> &observation)
{
    uint32_t observationCount = (uint32_t) observation.size();
    uint32_t stateCount = mHmm.getNumStates();
    Array2D<float> mat(stateCount, observationCount);

    // Initialiation
    // for each state
    for (uint32_t i = 0; i < stateCount; i++) {
        // 0 is log(1)
        mat.at(i, observationCount - 1) = 0;
    }

    float temp;
    // for each observation
    for (int t = observationCount - 2; t >= 0; t--) {
        // for each HMM state
        for (uint32_t i = 0; i < stateCount; i++) {
            for (uint32_t j = 0; j < stateCount; j++) {
                temp = mat.at(j, t + 1) + mHmm.mLogA->at(i, j) +
                       mHmm.mLogB->at(j, observation.at(t + 1));
                mat.at(i, t) = logAdd(mat.at(i, t), temp);
            }
        }
    }

    return mat;

}

float ForwardAlgorithmCPU::backward(vector<uint32_t> &observation)
{
    Array2D<float> beta = getBeta(observation);
    int stateCount = mHmm.getNumStates();

    float logLikelyhood = -INFINITY;
    for (int i = 0; i < stateCount; i++) {
        logLikelyhood = logAdd(logLikelyhood, beta.at(i, 0) +
                                              mHmm.mLogB->at(i,
                                                            observation.at(0)) +
                                              mHmm.mLogPi->at(i));
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

} // namespace hmm
