/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#include "HiddenMarkovModel.h"
#include "ViterbiAlgorithm.h"
#include "ForwardAlgorithm.h"
#include <fstream>
#include <iostream>

using namespace hmm;
using namespace std;

int main(int argc, char *argv[])
{
    // Test the number of parameters
    if (argc < 3)
        return EXIT_FAILURE;

    // Test input files
    if (not ifstream(argv[1]).good()) {
        cerr << "File " << argv[1] << " doesn't exist.";
        return EXIT_FAILURE;
    }
    if (not ifstream(argv[2]).good()) {
        cerr << "File " << argv[2] << " doesn't exist.";
        return EXIT_FAILURE;
    }

    HiddenMarkovModel hmm(argv[1], argv[2]);
    ForwardAlgorithmCPU forward(hmm);
    ViterbiAlgorithmCPU viterbi(hmm);

    if (argc == 4) {
        ifstream ifs(argv[3]);
        char buffer[256];

        while (ifs.getline(buffer, 256)) {
            string observation(buffer);

            auto stateSequence = viterbi.evaluate(observation);
            float likelihood = forward.evaluate(observation);
            cout << "Observation: " << observation << endl;
            cout << "Sequence:    " << stateSequence << endl;
            cout << "Likelihood:  " << likelihood << endl;
        }
    }

    return EXIT_SUCCESS;
}
