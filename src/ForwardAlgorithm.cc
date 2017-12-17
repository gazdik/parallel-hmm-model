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

float ForwardAlgorithmCPU::evaluate(Array2D<unsigned> &observation)
{
    return ForwardAlgorithm::evaluate(observation);
}

ForwardAlgorithmGPU::ForwardAlgorithmGPU(hmm::HiddenMarkovModel &hmm)
        : ForwardAlgorithm(hmm)
{}

float ForwardAlgorithmGPU::evaluate(Array2D<unsigned> &observation)
{
    return ForwardAlgorithm::evaluate(observation);
}

} // namespace hmm
