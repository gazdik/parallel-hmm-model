/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_VITERBIALGORITHMGPU_H
#define GMU_VITERBIALGORITHMGPU_H

#include "ViterbiAlgorithm.h"
#include "GPUImplementation.h"

namespace hmm
{

class ViterbiAlgorithmGPU : public ViterbiAlgorithm,
                            public GPUImplementation
{
public:
    ViterbiAlgorithmGPU(HiddenMarkovModel &hmm, cl::Context &context);

    using ViterbiAlgorithm::evaluate;
};

} // namespace hmm


#endif //GMU_VITERBIALGORITHMGPU_H
