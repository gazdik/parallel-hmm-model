/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_BAUMWELCHALGORITHM_H
#define GMU_BAUMWELCHALGORITHM_H

#include "HMMAlgorithm.h"

namespace hmm
{

/**
 * Base class for CPU and GPU implementation of Baum-Welch Algorithm
 */
class BaumWelchAlgorithm : public HMMAlgorithm
{
public:
    BaumWelchAlgorithm(HiddenMarkovModel &hmm);

};

/**
 * CPU implementation of Baum-Welch Algorithm
 */
class BaumWelchAlgorithmCPU : public BaumWelchAlgorithm
{
public:
    BaumWelchAlgorithmCPU(HiddenMarkovModel &hmm);

};

} // namespace hmm


#endif //GMU_BAUMWELCHALGORITHM_H
