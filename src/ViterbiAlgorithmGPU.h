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

    virtual void printStatistics() override;

private:

    void createBuffers();
    void initKernels();

private:
    cl::Kernel mKernelRecursionStep;
    cl::Kernel mKernelInit;
    cl::Kernel mKernelTermination;
    cl::Kernel mKernelBacktrace;

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
    size_t mMaxWorkGroupSize;

    // Profiling variables
    cl::UserEvent mEventLogACopy;
    cl::UserEvent mEventLogBCopy;
    cl::UserEvent mEventLogPiCopy;
    cl::UserEvent mEventPathCopy;
    cl::UserEvent mEventKernelInitialization;
    std::vector<cl::UserEvent> mEventKernelRecursion;
    cl::UserEvent mEventKernelTermination;
    cl::UserEvent mEventKernelBacktrace;
    double mStartTime = 0.0f;
    double mEndTime = 0.0f;

    static std::vector<std::string> SOURCE_FILES;
};

} // namespace hmm


#endif //GMU_VITERBIALGORITHMGPU_H
