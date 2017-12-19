/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#ifndef GMU_HIDDENMARKOVMODEL_H
#define GMU_HIDDENMARKOVMODEL_H

#include <map>
#include <string>
#include <vector>
#include "Array.h"

namespace hmm
{

/**
 * Hidden Markov Model specified as tripple M = (A,B,pi),
 * where A is a transition probability matrix, B is a emission probability matrix
 * and pi is a vector of initial probabilities.
 */
class HiddenMarkovModel
{
public:
    HiddenMarkovModel(const std::string &fileTransitions,
                      const std::string &fileEmissions);

    std::uint32_t getNumStates() const;

    std::uint32_t getNumSymbols() const;

    std::vector<uint32_t > translateObservation(
            const std::string &strObservation);

    std::string translateStateSequence(const std::vector<uint32_t> &sequence);

    /**
     * Total number of states.
     */
    unsigned mNumStates;
    /**
     * Total number of symbols in a vocabulary.
     */
    unsigned mNumSymbols;
    /**
     * Transition probability matrix A, each element (i,j) represents the
     * probability of moving from state i to state j.
     */
    Array2D<float> *mLogA;
    /**
     * Emission probability matrix B, each element (i,j) represents
     * the probability of observation symbol j being generated from state i.
     */
    Array2D<float> *mLogB;
    /**
     * Vector of initial state probabilities.
     * Each element i represents the probability that the Markov chain will
     * start in state i.
     */
    Array1D<float> *mLogPi;
    /**
     * Mapping between indices and original vocabulary symbols.
     */
    std::map<std::uint32_t, std::string> mIndexToOutputMap;
    /**
     * Mapping between original vocabulary symbols to indices.
     */
    std::map<std::string, std::uint32_t> mOutputToIndexMap;
    /**
     * Mapping between indices and state names.
     */
    std::map<std::uint32_t, std::string> mIndexToStateMap;
    /**
     * Mapping between state names and indices.
     */
    std::map<std::string, std::uint32_t> mStateToIndexMap;

private:

    /**
     * Load transition probabilities and emission probabilities
     * specified in the given file.
     *
     * Each row of the file 'fileTransitions' is of the form
     * 'fromstate tostate P(tostate|fromstate)'
     * and '#' is a special state that denotes the start state.
     *
     * Each row of the file 'fileEmissions' is of the form
     * 'state output P(output|state)'
     */

    void loadModel(const std::string &fileTransitions,
                   const std::string &fileEmissions);

    void loadSymbols(const std::string &fileTransitions,
                     const std::string &fileEmissions);

    void storeModel(const std::string &fileTransitions,
                    const std::string &fileEmissions);

    void debugPrint();

};

}

#endif //GMU_HIDDENMARKOVMODEL_H
