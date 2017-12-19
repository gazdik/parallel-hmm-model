/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_HELPERS_H
#define GMU_HELPERS_H

#include <cstddef>
#include <cmath>

namespace hmm
{

template<typename T>
inline T index1D(T x, T y, T width)
{
    return y + (x * width);
}

inline float convertToLog(float value)
{
    if (value == 0.0f)
        return - INFINITY;
    else
        return std::log(value);
}

}

#endif //GMU_HELPERS_H
