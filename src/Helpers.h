/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_HELPERS_H
#define GMU_HELPERS_H

inline unsigned index1D(unsigned x, unsigned y, unsigned width)
{
    return x + (y * width);
}

inline unsigned xCoord2D(unsigned index, unsigned width)
{
    return index % width;
}

inline unsigned yCoord2D(unsigned index, unsigned width)
{
    return index / width;
}


#endif //GMU_HELPERS_H
