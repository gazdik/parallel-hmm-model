/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "ViterbiAlgorithm.h"

namespace hmm
{

ViterbiAlgorithm::ViterbiAlgorithm(hmm::HiddenMarkovModel &hmm)
        : HMMAlgorithm(hmm)
{}

ViterbiAlgorithmCPU::ViterbiAlgorithmCPU(HiddenMarkovModel &hmm)
        : ViterbiAlgorithm(hmm)
{}


ViterbiAlgorithmGPU::ViterbiAlgorithmGPU(HiddenMarkovModel &hmm)
        : ViterbiAlgorithm(hmm)
{}


} // namespace hmm
