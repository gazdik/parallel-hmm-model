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

    virtual float evaluate(const std::string &observation);
    virtual float evaluate(std::vector<uint32_t> &observation) {}

};

/**
 * CPU implementation of Forward Algorithm
 */
class ForwardAlgorithmCPU : public ForwardAlgorithm
{
public:
    ForwardAlgorithmCPU(HiddenMarkovModel &hmm);

    using ForwardAlgorithm::evaluate;
    virtual float evaluate(std::vector<uint32_t> &observation) override;
    virtual Array2D<float> getAlpha(std::vector<uint32_t> &observation);
    virtual Array2D<float> getBeta(std::vector<uint32_t> &observation);

private:
    float logAdd(float x, float y);
    virtual float backward(std::vector<uint32_t> &observation);
    virtual float forward(std::vector<uint32_t> &observation);
};

/**
 * GPU implementation of Forward Algorithm
 */
class ForwardAlgorithmGPU : public ForwardAlgorithm
{
public:
    ForwardAlgorithmGPU(HiddenMarkovModel &hmm);

};

} // namespace hmm


#endif //GMU_FORWARDALGORITHM_H
