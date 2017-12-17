/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_VITERBIALGORITHM_H
#define GMU_VITERBIALGORITHM_H

#include "HMMAlgorithm.h"

namespace hmm
{

/**
 * Base class for CPU and GPU implementation of Viterbi Algorithm
 */
class ViterbiAlgorithm : public HMMAlgorithm
{
public:
    ViterbiAlgorithm(HiddenMarkovModel &hmm);

};

/**
 * CPU implementation of Viterbi Algorithm
 */
class ViterbiAlgorithmCPU : public ViterbiAlgorithm
{
public:
    ViterbiAlgorithmCPU(HiddenMarkovModel &hmm);

};

/**
 * GPU implementation of Viterbi Algorithm
 */
class ViterbiAlgorithmGPU : public ViterbiAlgorithm
{
public:
    ViterbiAlgorithmGPU(HiddenMarkovModel &hmm);

};

} // namespace hmm


#endif //GMU_VITERBIALGORITHM_H
