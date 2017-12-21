/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_FORWARDALGORITHMGPU_H
#define GMU_FORWARDALGORITHMGPU_H

#include "ForwardAlgorithm.h"
#include "GPUImplementation.h"

namespace hmm
{

/**
 * GPU implementation of Forward Algorithm
 */
class ForwardAlgorithmGPU : public ForwardAlgorithm,
                            public GPUImplementation
{
public:
    ForwardAlgorithmGPU(HiddenMarkovModel &hmm, cl::Context &context,
                        std::vector<cl::Device> &devices);

public:

    using ForwardAlgorithm::evaluate;
};

} // namespace hmm

#endif //GMU_FORWARDALGORITHMGPU_H
