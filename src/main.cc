/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */

#include "HiddenMarkovModel.h"
#include "oclHelpers.h"
#include "ViterbiAlgorithmGPU.h"
#include "ForwardAlgorithm.h"
#include "ForwardAlgorithmGPU.h"
#include <fstream>
#include <iostream>
#include <CL/cl.hpp>
#include <cstdlib>
#include <random>
#include <getopt.h>

using namespace hmm;
using namespace std;

static const char *helpMsg =
        "Usage hmm -s numOfStates -o numOfOutputs [-p platform] [-d device] [-l obsLength] [-O numOfObservations]";

int main(int argc, char *argv[])
{
    // Parse program arguments
    int opt;
    int numStates = -1,
        numOutputs = -1;
    int platformIndex = 0,
        deviceIndex = 0;
    uint32_t obsLength = 30;
    uint32_t numObservations = 1;
    while((opt = getopt(argc, argv, "s:o:p:d:l:O:h")) != -1) {
        switch (opt) {
            case 's':
                numStates = atoi(optarg);
                break;
            case 'o':
                numOutputs = atoi(optarg);
                break;
            case 'p':
                platformIndex = atoi(optarg);
                break;
            case 'd':
                deviceIndex = atoi(optarg);
                break;
            case 'h':
                cout << helpMsg << endl;
                exit(EXIT_SUCCESS);
            case 'l':
                obsLength = (uint32_t) atoi(optarg);
                break;
            case 'O':
                numObservations = (uint32_t) atoi(optarg);
                break;
            default:
                cerr << "Wrong program arguments\n\n" << helpMsg << endl;
                exit(EXIT_FAILURE);
        }
    }

    if (numStates == -1 || numOutputs == -1) {
        cerr << "Missing program options\n\n" << helpMsg << endl;
        exit(EXIT_FAILURE);
    }

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

    // Select an OpenCL device
    cl::Device device;
    platforms.at((unsigned long) platformIndex).getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
    device = platform_devices.at((unsigned long) deviceIndex);

    // Print the name of selected device
    printf("Selected device: %s\n", device.getInfo<CL_DEVICE_NAME>(&err).c_str());
    clCheckError(err, "cl::Device::getInfo<CL_DEVICE_NAME>");
    platforms.clear();

    // Create a context
    cl::Context context(platform_devices, NULL, NULL, NULL, &err);
    clCheckError(err, "cl::Context");

    HiddenMarkovModel hmm((size_t) numStates, (size_t) numOutputs);

    /*
     * Generate random observations
     */
    default_random_engine generator(4654646);
    uniform_int_distribution<uint32_t> distribution(0, hmm.getNumSymbols() - 1);
    vector<vector<uint32_t>> observations(numObservations);
    for (auto &o: observations) {
        for (uint32_t i = 0; i < obsLength; i++) {
            o.push_back(distribution(generator));
        }
    }


    printf("-----------------------------------------\n");
    printf("Number of states: %d\n", numStates);
    printf("Number of outputs: %d\n", numOutputs);
    printf("Observation length: %d\n", obsLength);

    /*
     * Forward Algorithm
     */
    float likelihoodCPU, likelihoodGPU;
    ForwardAlgorithmCPU forwardCPU(hmm);
    ForwardAlgorithmGPU forwardGPU(hmm, obsLength, context, platform_devices);

    printf("**********  Forward Algorithm  **********\n");
    for (size_t i = 0; i < observations.size(); i++) {
        printf("----------    Observation %zu    ----------\n", i);

        likelihoodGPU = forwardGPU.evaluate(observations[i]);
        forwardGPU.printStatistics();

        likelihoodCPU = forwardCPU.evaluate(observations[i]);
        forwardCPU.printStatistics();

        // Print results
        printf("Results: %s, CPU: %f, GPU: %f\n",
               ((likelihoodCPU - likelihoodGPU) < 1e-2) ? "correct" : "incorrect",
               likelihoodCPU,
               likelihoodGPU
        );
    }

    /*
     * Viterbi algorithm
     */
    std::vector<uint32_t> vitPathCPU, vitPathGPU;
    bool correctResult;
    ViterbiAlgorithmCPU viterbiCPU(hmm);
    ViterbiAlgorithmGPU viterbiGPU(hmm, obsLength, context, platform_devices);

    printf("**********  Viterbi Algorithm  **********\n");
    for (size_t i = 0; i < observations.size(); i++) {
        printf("----------    Observation %zu    ----------\n", i);

        vitPathGPU = viterbiGPU.evaluate(observations[i]);
        viterbiGPU.printStatistics();

        vitPathCPU = viterbiCPU.evaluate(observations[i]);
        viterbiCPU.printStatistics();

        // Compare results
        correctResult = vitPathCPU.size() == vitPathGPU.size();
        for (size_t i = 0; i < vitPathCPU.size(); i++) {
            if (not correctResult)
                break;

            correctResult = vitPathCPU[i] == vitPathGPU[i];
        }
        printf("Result: %s\n", correctResult ? "correct" : "incorrect");
    }
    printf("\n\n");

    exit(EXIT_SUCCESS);
}
