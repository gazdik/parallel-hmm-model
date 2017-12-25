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
    ForwardAlgorithmGPU(HiddenMarkovModel &hmm, uint32_t maxObservationLength,
                        cl::Context &context,
                        std::vector<cl::Device> &devices);

    using ForwardAlgorithm::evaluate;


    virtual float evaluate(
            std::vector<std::uint32_t> &observation);

    virtual void printStatistics() override;

protected:

private:

    void createBuffers();
    void initKernels();

private:
    cl::Kernel mKernelRecursionStep;
    cl::Kernel mKernelInitAlpha;
    cl::Kernel mKernelTermination;

    cl::Buffer mLogABuffer;
    cl::Buffer mLogBBuffer;
    cl::Buffer mLogPiBuffer;
    cl::Buffer mAlphaBuffer;
    cl::Buffer mLikelihoodBuffer;
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
    cl::UserEvent mEventKernelInitialization;
    std::vector<cl::UserEvent> mEventKernelRecursion;
    cl::UserEvent mEventKernelTermination;
    cl::UserEvent mEventResultCopy;
    double mStartTime = 0.0;
    double mEndTime = 0.0;
};

} // namespace hmm

#endif //GMU_FORWARDALGORITHMGPU_H
