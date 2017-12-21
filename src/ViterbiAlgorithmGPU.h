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
    using ViterbiAlgorithm::evaluate;

    ViterbiAlgorithmGPU(HiddenMarkovModel &hmm, uint32_t maxObservationLength,
                        cl::Context &context, std::vector<cl::Device> &devices);

    virtual std::vector<std::uint32_t> evaluate(
            std::vector<std::uint32_t> &observation);

private:

    void createBuffers();
    void initKernels();

private:
    cl::Kernel mKernelRecursionStep;
    cl::Kernel mKernelInit;
    cl::Kernel mKernelTermination;
    cl::Kernel mKernelViterbiPath;

    cl::Buffer mLogABuffer;
    cl::Buffer mLogBBuffer;
    cl::Buffer mLogPiBuffer;
    cl::Buffer mViterbiBuffer;
    cl::Buffer mBacktraceBuffer;
    cl::Buffer mMaxStateBuffer;
    cl::Buffer mPathBuffer;
    uint32_t mMaxObservationLength;
    size_t mInitLocalWorkSize = 1;
    size_t mInitGlobalWorkSize = 1;
    size_t mRecursionLocalWorkSize = 1;
    size_t mRecursionGlobalWorkSize = 1;
//    cl::NDRange mInitLocalNDRange;
//    cl::NDRange mInitGlobalNDRange;
//    cl::NDRange mReductionLocalNDRange;
//    cl::NDRange mReductionGlobalNDRange;
    size_t mMaxWorkGroupSize;
};

} // namespace hmm


#endif //GMU_VITERBIALGORITHMGPU_H
