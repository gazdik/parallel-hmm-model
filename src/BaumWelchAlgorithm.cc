/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "BaumWelchAlgorithm.h"

namespace hmm
{

BaumWelchAlgorithm::BaumWelchAlgorithm(hmm::HiddenMarkovModel &hmm)
        : HMMAlgorithm(hmm)
{}

BaumWelchAlgorithmCPU::BaumWelchAlgorithmCPU(HiddenMarkovModel &hmm)
        : BaumWelchAlgorithm(hmm)
{}

BaumWelchAlgorithmGPU::BaumWelchAlgorithmGPU(HiddenMarkovModel &hmm)
        : BaumWelchAlgorithm(hmm)
{}

} // namespace hmm
