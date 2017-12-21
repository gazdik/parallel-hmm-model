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
    platforms.at((unsigned long) stoi(argv[1])).getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
    device = platform_devices.at((unsigned long) stoi(argv[2]));

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

    uint32_t obsLength = 30;
    uint32_t numObservations = 1;

    HiddenMarkovModel hmm((size_t) stoi(argv[3]), (size_t) stoi(argv[4]));
    ViterbiAlgorithmCPU viterbiCPU(hmm);
    ViterbiAlgorithmGPU viterbiGPU(hmm, obsLength, context, platform_devices);

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

    std::vector<uint32_t> vitPath;

    double viterbiCPUStart = getTime();
    for (auto &o: observations) {
        vitPath = viterbiCPU.evaluate(o);
    }
    double viterbiCPUEnd = getTime();

    double viterbiGPUStart = getTime();
    for (auto &o: observations) {
        vitPath = viterbiGPU.evaluate(o);
    }
    double viterbiGPUEnd = getTime();

//    for (auto &o: observations) {
//        printf("--------------------------------------------------\n");
//        printf("Observation: ");
//        for (auto i: o) {
//            printf("%5d ", i);
//        }
//        printf("\n");
//
//        vitPath = viterbiGPU.evaluate(o);
//
//        printf("Vit. GPU:    ");
//        for (auto i: vitPath) {
//            printf("%5d ", i);
//        }
//        printf("\n");
//
//        vitPath = viterbiCPU.evaluate(o);
//
//        printf("Vit. CPU:    ");
//        for (auto i: vitPath) {
//            printf("%5d ", i);
//        }
//        printf("\n");
//    }

    ForwardAlgorithmCPU forwardCPU(hmm);
    ForwardAlgorithmGPU forwardGPU(hmm, obsLength, context, platform_devices);

    double forwardCPUStart = getTime();
    float likelihood;

    for (auto &o: observations) {
        likelihood = forwardCPU.evaluate(o);
    }
    double forwardCPUEnd = getTime();

    double forwardGPUStart = getTime();
    for (auto &o: observations) {
        likelihood = forwardGPU.evaluate(o);
    }
    double forwardGPUEnd = getTime();

//    for (auto &o: observations) {
//        printf("--------------------------------------------------\n");
//        printf("Observation: ");
//        for (auto i: o) {
//            printf("%5d ", i);
//        }
//        printf("\n");
//
//        likelihood = forwardGPU.evaluate(o);
//        printf("Likelihood GPU: %f\n", likelihood);
//
//        likelihood = forwardCPU.evaluate(o);
//        printf("Likelihood CPU: %f\n", likelihood);
//
//    }

    // Print results
    printf("Forward Algorithm: \n");
    printf("  Timers: cpu:%.3fms gpu:%.3fms\n",
           (forwardCPUEnd - forwardCPUStart) * 1000,
           (forwardGPUEnd - forwardGPUStart) * 1000
    );

    printf("Viterbi Algorithm: \n");
    printf("  Timers: cpu:%.3fms gpu:%.3fms\n",
           (viterbiCPUEnd - viterbiCPUStart) * 1000,
           (viterbiGPUEnd - viterbiGPUStart) * 1000
    );

    return EXIT_SUCCESS;
}
