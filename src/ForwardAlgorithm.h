/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_FORWARDALGORITHM_H
#define GMU_FORWARDALGORITHM_H

#include "HMMAlgorithm.h"

namespace hmm
{

/**
 * Base class for CPU and GPU implementation
 */
class ForwardAlgorithm : public HMMAlgorithm
{
public:
    ForwardAlgorithm(HiddenMarkovModel &hmm);

    float evaluate(const std::string &observation);

    virtual float evaluate(Array2D<unsigned> & observation);

protected:

};

/**
 * CPU implementation of Forward Algorithm
 */
class ForwardAlgorithmCPU : public ForwardAlgorithm
{
public:
    ForwardAlgorithmCPU(HiddenMarkovModel &hmm);

    virtual float evaluate(Array2D<unsigned> &observation) override;

};

/**
 * GPU implementation of Forward Algorithm
 */
class ForwardAlgorithmGPU : public ForwardAlgorithm
{
public:
    ForwardAlgorithmGPU(HiddenMarkovModel &hmm);

    virtual float evaluate(Array2D<unsigned> &observation) override;

};

} // namespace hmm


#endif //GMU_FORWARDALGORITHM_H
