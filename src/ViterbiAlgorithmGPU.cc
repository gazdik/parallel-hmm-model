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

vector<string> sourceFiles
{
        "src/ViterbiAlgorithmGPU.cl"
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
    mKernelViterbiPath = cl::Kernel(mProgram, "viterbi_path", &err);
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

    err |= mKernelRecursionStep.setArg(0, mLogABuffer);
    err |= mKernelRecursionStep.setArg(1, mLogBBuffer);
    err |= mKernelRecursionStep.setArg(2, mLogPiBuffer);
    err |= mKernelRecursionStep.setArg(3, mViterbiBuffer);
    err |= mKernelRecursionStep.setArg(4, mBacktraceBuffer);
    err |= mKernelRecursionStep.setArg(5, cl::Local(sizeof(float) * mRecursionLocalWorkSize));
    err |= mKernelRecursionStep.setArg(6, cl::Local(sizeof(cl_int) * mRecursionLocalWorkSize));
    err |= mKernelRecursionStep.setArg(7, mHmm.getNumStates());
    err |= mKernelRecursionStep.setArg(8, mHmm.getNumSymbols());
    clCheckError(err, "clKernelSetArg mKernelRecursionStep");

    mKernelTermination.setArg(0, mViterbiBuffer);
    mKernelTermination.setArg(1, mMaxStateBuffer);
    mKernelTermination.setArg(2, mRecursionLocalWorkSize * sizeof(float), nullptr);
    mKernelTermination.setArg(3, mRecursionLocalWorkSize * sizeof(cl_int), nullptr);
    mKernelTermination.setArg(4, mHmm.getNumStates());

    mKernelViterbiPath.setArg(0, mBacktraceBuffer);
    mKernelViterbiPath.setArg(1, mMaxStateBuffer);
    mKernelViterbiPath.setArg(2, mHmm.getNumStates());
    mKernelViterbiPath.setArg(3, mPathBuffer);
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
    mCmdQueue.enqueueWriteBuffer(mLogABuffer, CL_FALSE, 0,
                                     mHmm.mLogA->size() * sizeof(float),
                                     mHmm.mLogA->getData()
    );
    mCmdQueue.enqueueWriteBuffer(mLogBBuffer, CL_FALSE, 0,
                                     mHmm.mLogB->size() * sizeof(float),
                                     mHmm.mLogB->getData()
    );
    mCmdQueue.enqueueWriteBuffer(mLogPiBuffer, CL_FALSE, 0,
                                     mHmm.mLogPi->size() * sizeof(float),
                                     mHmm.mLogPi->getData()
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
    compileProgram(sourceFiles, "-w -Werror");
    createBuffers();
    initKernels();

    cout << "Max workgroup size: " << mMaxWorkGroupSize << endl;
    cout << "Number of states: " << mHmm.getNumStates() << endl;
    cout << "Local work size: " << mRecursionLocalWorkSize << endl;
    cout << "Global work size: " << mRecursionGlobalWorkSize << endl;
}

std::vector<std::uint32_t>
ViterbiAlgorithmGPU::evaluate(std::vector<std::uint32_t> &observation)
{
    cl_int err;
    uint32_t obsLength = (uint32_t) observation.size();

    // 1. Initialization
    mKernelInit.setArg(6, observation.front());
    mCmdQueue.enqueueNDRangeKernel(mKernelInit, 0,
                                   cl::NDRange(mInitGlobalWorkSize),
                                   cl::NDRange(mInitLocalWorkSize)
    );

    // 2. Recursion
    for (uint32_t t = 1; t < obsLength; t++) {
        mKernelRecursionStep.setArg(9, observation[t]);
        mKernelRecursionStep.setArg(10, t);

        mCmdQueue.enqueueNDRangeKernel(mKernelRecursionStep, 0,
                                       cl::NDRange(mRecursionGlobalWorkSize),
                                       cl::NDRange(mRecursionLocalWorkSize)
        );
    }

    // 3. Termination
    mKernelTermination.setArg(5, obsLength);
    mCmdQueue.enqueueNDRangeKernel(mKernelTermination, 0,
                                   cl::NDRange(mRecursionGlobalWorkSize),
                                   cl::NDRange(mRecursionLocalWorkSize)
    );

    // 4. Backtrace
    mKernelViterbiPath.setArg(4, obsLength);
    mCmdQueue.enqueueNDRangeKernel(mKernelViterbiPath, 0,
                                   cl::NDRange(1),
                                   cl::NDRange(1)
    );

    // Get result
    uint32_t *path = new uint32_t[obsLength];
    mCmdQueue.enqueueReadBuffer(mPathBuffer, CL_TRUE, 0,
                                obsLength * sizeof(uint32_t),
                                path
    );

    vector<uint32_t> ret(path, path + obsLength);
    delete[] path;
    return ret;
}


} // namespace hmm
