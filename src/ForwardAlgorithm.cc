/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "ForwardAlgorithm.h"

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
    int observationCount = observation.getNumElements();
    int stateCount = mHmm.getNumStates();
    Array2D<float> mat(stateCount, observationCount);

    // Initialiation
    for(int i = 0; i < stateCount; i++) {
        //mat.at(i, 0) = mHmm.mLogPi.at(i)+ mHmm.mLogB.at(observation.at())
    }

    // for each observation
    for(int t = 1; t < observationCount; t++) {
        // for each HMM state
        for (int i = 0; i < mHmm.getNumStates();i++) {

        }
    }

    return ForwardAlgorithm::evaluate(observation);
}

ForwardAlgorithmGPU::ForwardAlgorithmGPU(hmm::HiddenMarkovModel &hmm)
        : ForwardAlgorithm(hmm)
{}

float ForwardAlgorithmGPU::evaluate(Array1D<unsigned> &observation)
{
    return ForwardAlgorithm::evaluate(observation);
}

} // namespace hmm
