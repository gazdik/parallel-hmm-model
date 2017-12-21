/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */
#include "HiddenMarkovModel.h"
#include "oclHelpers.h"
#include "ViterbiAlgorithmGPU.h"
#include <fstream>
#include <iostream>
#include <CL/cl.hpp>
#include <cstdlib>

using namespace hmm;
using namespace std;

int main(int argc, char *argv[])
{


    cl_int err;
    vector<cl::Platform> platforms;
    vector<cl::Device> platform_devices;

    // Get available platforms
    clCheckError(cl::Platform::get(&platforms), "cl::Platform::get");
    for (size_t i = 0; i < platforms.size(); i++) {
        // Print platform name
        printf(" %d. platform name: %s.\n", i,
               platforms[i].getInfo<CL_PLATFORM_NAME>(&err).c_str());
        clCheckError(err, "cl::Platform::getInfo<CL_PLATFORM_NAME>");

        // Get number of platform devices
        err = platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
        clCheckError(err, "getDevices");

        if (platform_devices.size() == 0) continue;

        for (size_t j = 0; j < platform_devices.size(); j++) {
            // Get device
            printf("  %d. device name: %s\n", j,
                   platform_devices[j].getInfo<CL_DEVICE_NAME>(&err).c_str());
            clCheckError(err, "cl::Device::getInfo<CL_DEVICE_NAME>");
        }
        platform_devices.clear();
    }

    // Select OpenCL device
    cl::Device device;
    platforms.at(stoi(argv[1])).getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
    device = platform_devices.at(stoi(argv[2]));

    // Print the name of selected device
    printf("Selected device: %s\n", device.getInfo<CL_DEVICE_NAME>(&err).c_str());
    clCheckError(err, "cl::Device::getInfo<CL_DEVICE_NAME>");
    platforms.clear();

    // Create context
    cl::Context context(platform_devices, NULL, NULL, NULL, &err);
    clCheckError(err, "cl::Context");

    // Test the number of parameters
    if (argc < 5) {
        cerr << "Wrong number of parameters." << endl;
        return EXIT_FAILURE;
    }

    // Test input files
    if (not ifstream(argv[3]).good()) {
        cerr << "File " << argv[1] << " doesn't exists";
        return EXIT_FAILURE;
    }
    if (not ifstream(argv[4]).good()) {
        cerr << "File " << argv[2] << " doesn't exists";
        return EXIT_FAILURE;
    }

    HiddenMarkovModel hmm(argv[3], argv[4]);
    ViterbiAlgorithmGPU viterbi(hmm, 20, context, platform_devices);

    if (argc == 6) {
        ifstream ifs(argv[5]);
        char buffer[256];

        while (ifs.getline(buffer, 256)) {
            string observation(buffer);

            auto stateSequence = viterbi.evaluate(observation);
            cout << "Observation: " << observation << endl;
            cout << "Sequence:    " << stateSequence << endl;
        }
    }

    return EXIT_SUCCESS;
}
