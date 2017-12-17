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

    virtual float evaluate(const std::string &observation) {}

    virtual float evaluate(Array1D<unsigned int> &observation) {}

    virtual Array2D<float> getAlpha(Array1D<unsigned int> &observation) {}

    virtual Array2D<float> getBeta(Array1D<unsigned int> &observation) {}

protected:

};

/**
 * CPU implementation of Forward Algorithm
 */
class ForwardAlgorithmCPU : public ForwardAlgorithm
{
public:
    ForwardAlgorithmCPU(HiddenMarkovModel &hmm);

    virtual float evaluate(Array1D<unsigned int> &observation) override;

    virtual Array2D<float> getAlpha(Array1D<unsigned int> &observation) {}

    virtual Array2D<float> getBeta(Array1D<unsigned int> &observation) {}

private:
    float logAdd(float x, float y);
    virtual float backward(Array1D<unsigned int> &observation);
    virtual float forward(Array1D<unsigned int> &observation);
};

/**
 * GPU implementation of Forward Algorithm
 */
class ForwardAlgorithmGPU : public ForwardAlgorithm
{
public:
    ForwardAlgorithmGPU(HiddenMarkovModel &hmm);

    virtual float evaluate(Array1D<unsigned> &observation) override;

};

} // namespace hmm


#endif //GMU_FORWARDALGORITHM_H
