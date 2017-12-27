/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

#include "ViterbiAlgorithmGPU.h"
#include "oclHelpers.h"

using namespace std;

namespace hmm
{

std::vector<std::string> ViterbiAlgorithmGPU::SOURCE_FILES =
        {
                "ViterbiAlgorithmGPU.cl"
        };

void ViterbiAlgorithmGPU::initKernels()
{
    cl_int  err;

    /**
     * Create kernels
     */
    mKernelInit = cl::Kernel(mProgram, "viterbi_init", &err);
    clCheckError(err, "clCreateKernel viterbi_init");
    mKernelRecursionStep = cl::Kernel(mProgram, "viterbi_recursion_step", &err);
    clCheckError(err, "clCreateKernel viterbi_recursion_step");
    mKernelTermination = cl::Kernel(mProgram, "viterbi_termination", &err);
    clCheckError(err, "clCreateKernel viterbi_termination");
    mKernelBacktrace = cl::Kernel(mProgram, "viterbi_path", &err);
    clCheckError(err, "clCreateKernel viterbi_path");

    /*
     * Calc local and global worksizes
     */
    mInitLocalWorkSize = 1;
    mInitGlobalWorkSize = mHmm.getNumStates();

    mMaxWorkGroupSize = mKernelRecursionStep.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(mDevices.front());
    size_t optimalLocalSize =
            (size_t) (int) pow(2.0f, (int) (log((float) mHmm.getNumStates()) /
                                            log(2.0f))); // the largest 2^n number smaller than nState
    mRecursionLocalWorkSize = (optimalLocalSize > mMaxWorkGroupSize) ? mMaxWorkGroupSize : optimalLocalSize;
    mRecursionGlobalWorkSize = mRecursionLocalWorkSize * mHmm.getNumStates();

    /*
     * Set static kernel arguments
     */
    mKernelInit.setArg(0, mLogABuffer);
    mKernelInit.setArg(1, mLogBBuffer);
    mKernelInit.setArg(2, mLogPiBuffer);
    mKernelInit.setArg(3, mViterbiBuffer);
    mKernelInit.setArg(4, mHmm.getNumStates());
    mKernelInit.setArg(5, mHmm.getNumSymbols());

    mKernelRecursionStep.setArg(0, mLogABuffer);
    mKernelRecursionStep.setArg(1, mLogBBuffer);
    mKernelRecursionStep.setArg(2, mLogPiBuffer);
    mKernelRecursionStep.setArg(3, mViterbiBuffer);
    mKernelRecursionStep.setArg(4, mBacktraceBuffer);
    mKernelRecursionStep.setArg(5, cl::Local(sizeof(float) * mRecursionLocalWorkSize));
    mKernelRecursionStep.setArg(6, cl::Local(sizeof(cl_int) * mRecursionLocalWorkSize));
    mKernelRecursionStep.setArg(7, mHmm.getNumStates());
    mKernelRecursionStep.setArg(8, mHmm.getNumSymbols());

    mKernelTermination.setArg(0, mViterbiBuffer);
    mKernelTermination.setArg(1, mMaxStateBuffer);
    mKernelTermination.setArg(2, mRecursionLocalWorkSize * sizeof(float), nullptr);
    mKernelTermination.setArg(3, mRecursionLocalWorkSize * sizeof(cl_int), nullptr);
    mKernelTermination.setArg(4, mHmm.getNumStates());

    mKernelBacktrace.setArg(0, mBacktraceBuffer);
    mKernelBacktrace.setArg(1, mMaxStateBuffer);
    mKernelBacktrace.setArg(2, mHmm.getNumStates());
    mKernelBacktrace.setArg(3, mPathBuffer);
}

void ViterbiAlgorithmGPU::createBuffers()
{
    // Create buffers
    cl_int  err;
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
    mViterbiBuffer = cl::Buffer(mContext,
                                CL_MEM_READ_WRITE,
                                mHmm.getNumStates() * mMaxObservationLength * sizeof(float),
                                nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mViterbiBuffer");
    mBacktraceBuffer = cl::Buffer(mContext,
                                  CL_MEM_READ_ONLY,
                                  mHmm.getNumStates() * (mMaxObservationLength - 1) * sizeof(uint32_t),
                                  nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mBacktraceBuffer");
    mMaxStateBuffer = cl::Buffer(mContext,
                                         CL_MEM_WRITE_ONLY,
                                         sizeof(uint32_t),
                                         nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mMaxStateStateBuffer");
    mPathBuffer = cl::Buffer(mContext,
                             CL_MEM_WRITE_ONLY,
                             mMaxObservationLength * sizeof(uint32_t),
                             nullptr, &err
    );
    clCheckError(err, "clCreateBuffer mPathBuffer");

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

ViterbiAlgorithmGPU::ViterbiAlgorithmGPU(HiddenMarkovModel &hmm,
                                         uint32_t maxObservationLength,
                                         cl::Context &context,
                                         std::vector<cl::Device> &devices) :
    ViterbiAlgorithm(hmm),
    GPUImplementation(context, devices),
    mMaxObservationLength { maxObservationLength }
{
    compileProgram(SOURCE_FILES, "-w -Werror");
    createBuffers();
    initKernels();
}

std::vector<std::uint32_t>
ViterbiAlgorithmGPU::evaluate(std::vector<std::uint32_t> &observation)
{
    uint32_t obsLength = (uint32_t) observation.size();
    cl_int err;

    mStartTime = getTime();

    // 1. Initialization
    mKernelInit.setArg(6, observation.front());
    mEventKernelInitialization = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueNDRangeKernel(mKernelInit, 0,
                                         cl::NDRange(mInitGlobalWorkSize),
                                         cl::NDRange(mInitLocalWorkSize),
                                         nullptr,
                                         &mEventKernelInitialization
    );
    clCheckError(err, "enqueueNDRangeKernel mKernelInitialization");

    // 2. Recursion
    mEventKernelRecursion.clear();
    for (uint32_t t = 1; t < obsLength; t++) {
        mKernelRecursionStep.setArg(9, observation[t]);
        mKernelRecursionStep.setArg(10, t);

        mEventKernelRecursion.push_back(cl::UserEvent(mContext));
        err = mCmdQueue.enqueueNDRangeKernel(mKernelRecursionStep, 0,
                                             cl::NDRange(mRecursionGlobalWorkSize),
                                             cl::NDRange(mRecursionLocalWorkSize),
                                             nullptr,
                                             &mEventKernelRecursion.back()
        );
        clCheckError(err, "enqueueNDRangeKernel mKernelRecursionStep");
    }

    // 3. Termination
    mKernelTermination.setArg(5, obsLength);
    mEventKernelTermination = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueNDRangeKernel(mKernelTermination, 0,
                                         cl::NDRange(mRecursionGlobalWorkSize),
                                         cl::NDRange(mRecursionLocalWorkSize),
                                         nullptr,
                                         &mEventKernelTermination
    );
    clCheckError(err, "enqueueNDRangeKernel mKernelTermination");

    // 4. Backtrace
    mKernelBacktrace.setArg(4, obsLength);
    mEventKernelBacktrace = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueNDRangeKernel(mKernelBacktrace, 0,
                                         cl::NDRange(1),
                                         cl::NDRange(1),
                                         nullptr,
                                         &mEventKernelBacktrace
    );
    clCheckError(err, "enqueueNDRangeKernel mKernelBacktrace");

    // Get result
    uint32_t *path = new uint32_t[obsLength];
    mEventPathCopy = cl::UserEvent(mContext);
    err = mCmdQueue.enqueueReadBuffer(mPathBuffer, CL_TRUE, 0,
                                      obsLength * sizeof(uint32_t),
                                      path,
                                      nullptr,
                                      &mEventPathCopy
    );
    clCheckError(err, "enqueueReadBuffer mPathBuffer");

    mEndTime = getTime();

    vector<uint32_t> ret(path, path + obsLength);
    delete[] path;
    return ret;
}

void ViterbiAlgorithmGPU::printStatistics()
{
    printf("Viterbi Algorithm GPU\n");
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
    printf("  Backtrace step: %.3fms\n", getEventTime(mEventKernelBacktrace) * 1000);
    printf("  Path copy: %.3fms\n", getEventTime(mEventPathCopy) * 1000);
}


} // namespace hmm
