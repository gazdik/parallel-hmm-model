/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_HELPERS_H
#define GMU_HELPERS_H

#include <cstddef>

namespace hmm
{

inline std::size_t index1D(std::size_t x, std::size_t y, std::size_t width)
{
    return y + (x * width);
}

}

#endif //GMU_HELPERS_H
