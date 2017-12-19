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
    GPUImplementation(cl::Context &context);

private:
    cl::Context &mContext;
};

}


#endif //GMU_GPUIMPLEMENTATION_H
