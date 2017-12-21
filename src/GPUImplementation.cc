/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "GPUImplementation.h"
#include "oclHelpers.h"

using namespace std;

namespace hmm
{


void GPUImplementation::compileProgram(const std::vector<std::string> &filenames,
                                       const char *buildOptions)
{
    cl_int err;
    cl::Program::Sources sources;
    for (auto &filename: filenames) {
        char *program_source = readFile(filename.c_str());
        if (program_source == NULL) {
            throw runtime_error("File " + filename + " doesn't exist");
        }
        sources.push_back({program_source, strlen(program_source)});
    }

    mProgram = cl::Program(mContext, sources);

    err = mProgram.build(mDevices, buildOptions);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        cl_int  err2;
        printf("Build log:\n %s\n",
               mProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(mDevices.front(),
                                                           &err2).c_str());
        clCheckError(err2, "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
    }
    clCheckError(err, "clBuildProgram");
}

GPUImplementation::GPUImplementation(cl::Context &context,
                                     std::vector<cl::Device> &devices) :
    mContext { context },
    mDevices { devices }
{
    cl_int err;
    mCmdQueue = cl::CommandQueue(
            mContext, mDevices.front(),
            CL_QUEUE_PROFILING_ENABLE,
            &err
    );

    clCheckError(err, "cl::CommandQueue");
}

} // namespace hmm
