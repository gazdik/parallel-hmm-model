/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 *              Michal Kula <ikula(at)fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   OpenCL helpers based on source codes from GMU exercises
 */

#ifndef GMU_OCLHELPERS_H
#define GMU_OCLHELPERS_H

#include <ctime>
#include <CL/cl.hpp>

namespace hmm
{

/**
 * Convert opencl error code to string
 */
const char *getCLError(cl_int err_id);

/**
 * Get time in seconds
 */
double getTime(void);

/**
 * Check for errors and throw exception
 */
void clCheckError(cl_int err_id, const char *msg);

/**
 * Align data_size size to align_size
 */
unsigned int alignTo(unsigned int data_size, unsigned int align_size);

/**
 * Get event time
 */
double getEventTime(cl_event i_event);
double getEventTime(cl::Event &i_event);

/**
 * Read file and return it as a C string
 */
char *readFile(const char *filename);

} // namespace hmm

#endif //GMU_OCLHELPERS_H
