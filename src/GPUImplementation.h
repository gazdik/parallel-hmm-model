/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_GPUIMPLEMENTATION_H
#define GMU_GPUIMPLEMENTATION_H

#include <CL/cl.hpp>

namespace hmm
{

class GPUImplementation
{
public:
    GPUImplementation(cl::Context &context, std::vector<cl::Device> &devices);

protected:
    virtual void compileProgram(const std::vector<std::string> &filenames,
                                    const char *buildOptions = "");


protected:
    cl::Context &mContext;
    std::vector<cl::Device> &mDevices;
    cl::CommandQueue mCmdQueue;

    cl::Program mProgram;
};

}


#endif //GMU_GPUIMPLEMENTATION_H
