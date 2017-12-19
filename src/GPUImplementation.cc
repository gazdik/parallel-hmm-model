/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "GPUImplementation.h"

namespace hmm
{

GPUImplementation::GPUImplementation(cl::Context &context) :
        mContext{context}
{

}

} // namespace hmm
