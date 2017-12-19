/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

#include "BaumWelchAlgorithmGPU.h"

namespace hmm
{

BaumWelchAlgorithmGPU::BaumWelchAlgorithmGPU(HiddenMarkovModel &hmm,
                                             cl::Context &context)
        : BaumWelchAlgorithm(hmm), GPUImplementation(context)
{}

} // namespace hmm