/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        02/12/17
 * @copyright   The MIT License (MIT)
 */

#include "HiddenMarkovModel.h"
#include <fstream>
#include <iostream>
#include <cstdlib>


using namespace hmm;
using namespace std;

int main(int argc, char *argv[])
{
    // Test the number of parameters
    if (argc < 3)
        return EXIT_FAILURE;

    // Test input files
    if (not ifstream(argv[1]).good()) {
        cerr << "File " << argv[1] << " doesn't exists";
        return EXIT_FAILURE;
    }
    if (not ifstream(argv[2]).good()) {
        cerr << "File " << argv[2] << " doesn't exists";
        return EXIT_FAILURE;
    }

    // Test array
    Array2D<float> arr;
    arr.at(9, 9);

    for (int i = 0; i < arr.getNumRows(); i++) {
        for (int j = 0; j < arr.getNumCols(); j++) {
            cout << arr.at(i, j) << " ";
        }
        cout << endl;
    }

    HiddenMarkovModel(argv[1], argv[2]);

    return EXIT_SUCCESS;
}
