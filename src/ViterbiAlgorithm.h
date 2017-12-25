/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_VITERBIALGORITHM_H
#define GMU_VITERBIALGORITHM_H

#include "HMMAlgorithm.h"

namespace hmm
{

/**
 * Base class for CPU and GPU implementation of Viterbi Algorithm
 */
class ViterbiAlgorithm : public HMMAlgorithm
{
public:
    ViterbiAlgorithm(HiddenMarkovModel &hmm);

    std::string evaluate(const std::string &observation);
    virtual std::vector<std::uint32_t> evaluate(
            std::vector<std::uint32_t> &observation) {};
};

/**
 * CPU implementation of Viterbi Algorithm
 */
class ViterbiAlgorithmCPU : public ViterbiAlgorithm
{
public:
    ViterbiAlgorithmCPU(HiddenMarkovModel &hmm);

    using ViterbiAlgorithm::evaluate;
    virtual std::vector<std::uint32_t>
    evaluate(std::vector<std::uint32_t> &observation) override;

    virtual void printStatistics() override;

private:

    // Profiling
    double mStartTime = 0.0;
    double mEndTime = 0.0;
    double mInitializationStartTime = 0.0;
    double mInitializationEndTime = 0.0;
    double mRecursionStartTime = 0.0;
    double mRecursionEndTime = 0.0;
    double mBacktraceStartTime = 0.0;
    double mBacktraceEndTime = 0.0;
    double mTerminationStartTime = 0.0;
    double mTerminationEndTime = 0.0;
};

} // namespace hmm


#endif //GMU_VITERBIALGORITHM_H
