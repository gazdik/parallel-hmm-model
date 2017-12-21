/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_BAUMWELCHALGORITHMGPU_H
#define GMU_BAUMWELCHALGORITHMGPU_H

#include "BaumWelchAlgorithm.h"
#include "GPUImplementation.h"

namespace hmm
{

/**
 * GPU implementation of Baum-Welch Algorithm
 */
class BaumWelchAlgorithmGPU : public BaumWelchAlgorithm,
                              public GPUImplementation
{
public:
    BaumWelchAlgorithmGPU(HiddenMarkovModel &hmm, cl::Context &context,
                          std::vector<cl::Device> &devices);

public:

};

} // namespace hmm


#endif //GMU_BAUMWELCHALGORITHMGPU_H
