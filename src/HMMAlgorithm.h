/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_HMMALGORITHM_H
#define GMU_HMMALGORITHM_H

#include "HiddenMarkovModel.h"

namespace hmm
{

class HMMAlgorithm
{
public:
    HMMAlgorithm(HiddenMarkovModel &hmm) :
            mHmm { hmm }
    {
    }

protected:
    HiddenMarkovModel &mHmm;
};

}


#endif //GMU_HMMALGORITHM_H
