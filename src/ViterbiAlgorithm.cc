/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "ViterbiAlgorithm.h"

using namespace std;

namespace hmm
{

ViterbiAlgorithm::ViterbiAlgorithm(hmm::HiddenMarkovModel &hmm)
        : HMMAlgorithm(hmm)
{}

ViterbiAlgorithmCPU::ViterbiAlgorithmCPU(HiddenMarkovModel &hmm)
        : ViterbiAlgorithm(hmm)
{}

std::vector<std::uint32_t>
ViterbiAlgorithmCPU::evaluate(std::vector<std::uint32_t> &o)
{
    uint32_t T = (uint32_t) o.size();
    uint32_t N = mHmm.getNumStates();
    Array2D<float> viterbi(N, T);
    Array2D<uint32_t> backtrace(N, T - 1);
    vector<uint32_t> path(T);

    float p;

    // Initialization
    for (uint32_t i = 0; i < N; i++) {
        viterbi.at(i, 0) = mHmm.mLogPi->at(i) + mHmm.mLogB->at(i, o.at(0));
    }
    // Recursion
    for (uint32_t t = 1; t < T; t++) {
        for (uint32_t j = 0; j < N; j++) {
            for (uint32_t i = 0; i < N; i++) {
//                p = viterbi.at(i, t - 1) + mHmm.mLogA->at(i, j) + mHmm.mLogB->at(j, o.at(t));
                p = viterbi.at(i, t - 1) + mHmm.mLogA->at(i, j);
                if (p > viterbi.at(j, t)) {
                    viterbi.at(j, t) = p;
                    backtrace.at(j, t - 1) = i;
                }
            }
            viterbi.at(j, t) += mHmm.mLogB->at(j, o.at(t));
        }
    }

    // Termination
    float max = -INFINITY;
    for (uint32_t i = 0; i < N; i++) {
        if (viterbi.at(i, T - 1) > max) {
            path.at(T - 1) = i;
            max = viterbi.at(i, T - 1);
        }
    }

    // Backtrace
    for (uint32_t i = T - 1; i > 0; i--) {
        path.at(i - 1) = backtrace.at(path.at(i), i - 1);
    }

//    // Print Observation sequence
//    printf("Observation sequence:\n");
//    for (uint32_t i = 0; i < T; i++) {
//        printf("%3d ", o.at(i));
//    }
//    printf("\n");
//
    // Print Viterbi matrix
    printf("Viterbi matrix\n");
    for (uint32_t j = 0; j < T; j++) {
        for (uint32_t i = 0; i < N; i++) {
            printf("%+5f ", viterbi.at(i, j));
        }
        printf("\n");
    }

    printf("Backtrace matrix\n");
    for (uint32_t j = 0; j < T - 1; j++) {
        for (uint32_t i = 0; i < N; i++) {
            printf("%3d ", backtrace.at(i, j));
        }
        printf("\n");
    }
//    printf("The most likely path:\n");
//    for (uint32_t i = 0; i < T; i++) {
//        printf("%3d ", path.at(i));
//    }
//    printf("\n");

    return path;
}

string ViterbiAlgorithm::evaluate(const std::string &strObservation)
{
    auto observation = mHmm.translateObservation(strObservation);
    auto sequence = this->evaluate(observation);

    return mHmm.translateStateSequence(sequence);
}

} // namespace hmm
