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

/**
 * Hidden Markov Model specified as tripple M = (A,B,pi),
 * where A is a transition probability matrix, B is a emission probability matrix
 * and pi is a vector of initial probabilities.
 */
class HiddenMarkovModel
{
public:
    HiddenMarkovModel(const std::string & fileTransitions,
                      const std::string & fileEmissions);

    HiddenMarkovModel(unsigned numStates, unsigned numSymbols);

    HiddenMarkovModel(float *A, float *B, float *pi);

    unsigned int getNumStates() const;

    unsigned int getNumSymbols() const;

    float *getLogA() const;

    float *getLogB() const;

    float *getLogPi() const;

private:
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
    float * mLogA;
    /**
     * Emission probability matrix B, each element (i,j) represents
     * the probability of observation symbol j being generated from state i.
     */
    float * mLogB;
    /**
     * Vector of initial state probabilities.
     * Each element i represents the probability that the Markov chain will
     * start in state i.
     */
    float * mLogPi;
    /**
     * Mapping between indices and original vocabulary symbols.
     */
    std::map<unsigned, std::string> mVocabularyMapping;
    /**
     * Mapping between indices and state names.
     */
    std::map<unsigned, std::string> mStatesMapping;
};


#endif //GMU_HIDDENMARKOVMODEL_H
