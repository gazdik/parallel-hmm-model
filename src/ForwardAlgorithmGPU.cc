/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

#include "ForwardAlgorithmGPU.h"
#include "oclHelpers.h"

using namespace std;

namespace hmm
{

vector<string> sourceFilesForward
        {
                "ForwardAlgorithmGPU.cl"
        };

void ForwardAlgorithmGPU::initKernels()
{
    cl_int err;

    /**
     * Create kernels
     */
    mKernelInitAlpha = cl::Kernel(mProgram, "forward_init", &err);
    clCheckError(err, "clCreateKernel forward_init");
    mKernelRecursionStep = cl::Kernel(mProgram, "forward_step", &err);
    clCheckError(err, "clCreateKernel forward_step");
    mKernelTermination = cl::Kernel(mProgram, "forward_termination", &err);
    clCheckError(err, "clCreateKernel forward_termination");

    /*
     * Calc local and global worksizes
     */
    mInitLocalWorkSize = 1;
    mInitGlobalWorkSize = mHmm.getNumStates();

    mMaxWorkGroupSize = mKernelRecursionStep.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(
            mDevices.front());
    size_t optimalLocalSize =
            (size_t) (int) pow(2.0f, (int) (log((float) mHmm.getNumStates()) /
                                            log(2.0f))); // the largest 2^n number smaller than nState
    mRecursionLocalWorkSize = (optimalLocalSize > mMaxWorkGroupSize)
                              ? mMaxWorkGroupSize : optimalLocalSize;
    mRecursionGlobalWorkSize = mRecursionLocalWorkSize * mHmm.getNumStates();

    /*
     * Set static kernel arguments
     */
    mKernelInitAlpha.setArg(0, mLogABuffer);
    mKernelInitAlpha.setArg(1, mLogBBuffer);
    mKernelInitAlpha.setArg(2, mLogPiBuffer);
    mKernelInitAlpha.setArg(3, mAlphaBuffer);
    mKernelInitAlpha.setArg(4, mHmm.getNumStates());
    mKernelInitAlpha.setArg(5, mHmm.getNumSymbols());

    err |= mKernelRecursionStep.setArg(0, mLogABuffer);
    err |= mKernelRecursionStep.setArg(1, mLogBBuffer);
    err |= mKernelRecursionStep.setArg(2, mLogPiBuffer);
    err |= mKernelRecursionStep.setArg(3, mAlphaBuffer);
    err |= mKernelRecursionStep.setArg(4, cl::Local(
            sizeof(float) * mRecursionLocalWorkSize));
    err |= mKernelRecursionStep.setArg(5, mHmm.getNumStates());
    err |= mKernelRecursionStep.setArg(6, mHmm.getNumSymbols());
    clCheckError(err, "clKernelSetArg mKernelRecursionStep");

    mKernelTermination.setArg(0, mAlphaBuffer);
    mKernelTermination.setArg(1, mLikelihoodBuffer);
    mKernelTermination.setArg(2, mRecursionLocalWorkSize * sizeof(float),
                              nullptr);
    mKernelTermination.setArg(3, mHmm.getNumStates());

}

void ForwardAlgorithmGPU::createBuffers()
{
    // Create buffers
    cl_int err;
    mLogABuffer = cl::Buffer(mContext,
                             CL_MEM_READ_ONLY,
                             mHmm.mLogA->size() * sizeof(float),
                             nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mLogABuffer");
    mLogBBuffer = cl::Buffer(mContext,
                             CL_MEM_READ_ONLY,
                             mHmm.mLogB->size() * sizeof(float),
                             nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mLogBBuffer");
    mLogPiBuffer = cl::Buffer(mContext,
                              CL_MEM_READ_ONLY,
                              mHmm.mLogPi->size() * sizeof(float),
                              nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mLogPiBuffer");
    mAlphaBuffer = cl::Buffer(mContext,
                                CL_MEM_READ_WRITE,
                                mHmm.getNumStates() * mMaxObservationLength *
                                sizeof(float),
                                nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mAlphaBuffer");
    mLikelihoodBuffer = cl::Buffer(mContext,
                                 CL_MEM_WRITE_ONLY,
                                 sizeof(float),
                                 nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mLikelihoodBuffer");

    // Copy data on the GPU device
    mEventLogACopy = cl::UserEvent(mContext);
    mCmdQueue.enqueueWriteBuffer(mLogABuffer, CL_FALSE, 0,
                                 mHmm.mLogA->size() * sizeof(float),
                                 mHmm.mLogA->getData(),
                                 nullptr,
                                 &mEventLogACopy
    );
    mEventLogBCopy = cl::UserEvent(mContext);
    mCmdQueue.enqueueWriteBuffer(mLogBBuffer, CL_FALSE, 0,
                                 mHmm.mLogB->size() * sizeof(float),
                                 mHmm.mLogB->getData(),
                                 nullptr,
                                 &mEventLogBCopy
    );
    mEventLogPiCopy = cl::UserEvent(mContext);
    mCmdQueue.enqueueWriteBuffer(mLogPiBuffer, CL_TRUE, 0,
                                 mHmm.mLogPi->size() * sizeof(float),
                                 mHmm.mLogPi->getData(),
                                 nullptr,
                                 &mEventLogPiCopy
    );
}

ForwardAlgorithmGPU::ForwardAlgorithmGPU(HiddenMarkovModel &hmm,
                                         uint32_t maxObservationLength,
                                         cl::Context &context,
                                         std::vector<cl::Device> &devices) :
        ForwardAlgorithm(hmm),
        GPUImplementation(context, devices),
        mMaxObservationLength{maxObservationLength}
{
    compileProgram(sourceFilesForward, "-w -Werror");
    createBuffers();
    initKernels();
}

float
ForwardAlgorithmGPU::evaluate(std::vector<std::uint32_t> &observation)
{
    cl_int err;
    uint32_t obsLength = (uint32_t) observation.size();

    mStartTime = getTime();

    // 1. Initialization
    mKernelInitAlpha.setArg(6, observation.front());
    mEventKernelInitialization = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueNDRangeKernel(mKernelInitAlpha, 0,
                                         cl::NDRange(mInitGlobalWorkSize),
                                         cl::NDRange(mInitLocalWorkSize),
                                         nullptr,
                                         &mEventKernelInitialization
    );
    clCheckError(err, "enqueueNDRangeKernel mKernelInitAlpha");

    // 2. Recursion
    mEventKernelRecursion.clear();
    for (uint32_t t = 1; t < obsLength; t++) {
        mKernelRecursionStep.setArg(7, observation[t]);
        mKernelRecursionStep.setArg(8, t);

        mEventKernelRecursion.push_back(cl::UserEvent(mContext));
        err = mCmdQueue.enqueueNDRangeKernel(mKernelRecursionStep,
                                             0,
                                             cl::NDRange(mRecursionGlobalWorkSize),
                                             cl::NDRange(mRecursionLocalWorkSize),
                                             nullptr,
                                             &mEventKernelRecursion.back()
        );
        clCheckError(err, "enqueueNDRangeKernel mKernelRecursionStep");
    }

    // 3. Termination
    mKernelTermination.setArg(4, obsLength);
    mEventKernelTermination = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueNDRangeKernel(mKernelTermination, 0,
                                         cl::NDRange(mRecursionGlobalWorkSize),
                                         cl::NDRange(mRecursionLocalWorkSize),
                                         nullptr,
                                         &mEventKernelTermination
    );
    clCheckError(err, "enqueueNDRangeKernel mKernelTermination");

    // Get result
    float result = 0;
    mEventResultCopy = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueReadBuffer(mLikelihoodBuffer, CL_TRUE, 0,
                                sizeof(float),
                                &result,
                                nullptr,
                                &mEventResultCopy
    );
    clCheckError(err, "enqueueReadBuffer likelihood");

    mEndTime = getTime();

    return result;
}

void ForwardAlgorithmGPU::printStatistics()
{
    printf("Forward Algorithm GPU\n");
    printf("  Execution time: %.3fms\n", (mEndTime - mStartTime) * 1000);
    printf("  Static buffers copy: %.3fms\n",
           (getEventTime(mEventLogACopy) + getEventTime(mEventLogBCopy)
            + getEventTime(mEventLogPiCopy)) * 1000
    );
    printf("  Initialization step: %.3fms\n", getEventTime(mEventKernelInitialization) * 1000);

    double kernelRecursionTime = 0.0f;
    for (auto &i: mEventKernelRecursion)
        kernelRecursionTime += getEventTime(i);
    printf("  Recursion step: %.3fms\n", kernelRecursionTime * 1000);

    printf("  Termination step: %.3fms\n", getEventTime(mEventKernelTermination) * 1000);
    printf("  Result copy: %.3fms\n", getEventTime(mEventResultCopy) * 1000);
}


} // namespace hmm
