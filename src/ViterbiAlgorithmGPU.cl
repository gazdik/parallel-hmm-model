/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        19/12/17
 * @copyright   The MIT License (MIT)
 */

__kernel void viterbi_init(__global float *logA,
                           __global float *logB,
                           __global float *logPi,
                           __global float *V,
                           unsigned int N,
                           unsigned int M,
                           unsigned int o)
{
    uint groupId = get_group_id(0);
    uint iState = groupId;

    // Initialization
    if (iState < N) {
        V[iState] = logPi[iState] + logB[iState * M + o];
    }
}

/**
 * Calculate Viterbi path.
 * @param logA  Transition probility matrix A, each element (i,j) represents
 *              the probability of moving from state i to state j.
 *              The matrix contains NxN elements where N is the number of states
 *              in the model.
 * @param logB  Emission probability matrix B, each element (i,j) represents
 *              the probability of observation symbol j being generated from
 *              state i. The matrix contains NxM where N is the number of states
 *              and M is the number of observation symbols in the model.
 * @param logPi Vector of initial state probabilities. The vector contains
 *              N elements where N is the number of states in the model.
 * @param O     Observation sequence which contains T observation symbols.
 * @param V     Viterbi probability matrix V, each element (i, t) represents
 *              the probability that the HMM is in state i after seeing
 *              the first t observations and passing through the most probable
 *              sequence x_0, x_1, ..., x_t-1.
 *              The matrix contains NxT elements where N is the number of states
 *              and T is the length of observation sequence.
 * @param N     The number of states in the model.
 * @param M     The number of observation symbols in the model.
 * @param T     The length of the observation sequence O
 * @return
 */

void local_maximum(__local float *maxValue,
                   __local int *maxInd)
{
    uint localId = get_local_id(0);
    uint localSize = get_local_size(0);

    int idx;
    float m1, m2, m3;

    for (int s = (localSize >> 1); s > 32; s >>= 1) {
        if (localId < s) {
            m1 = maxValue[localId];
            m2 = maxValue[localId + s];
            m3 = (m1 >= m2) ? m1 : m2;
            idx = (m1 >= m2) ? localId : localId + s;
            maxValue[localId] = m3;
            maxInd[localId] = maxInd[idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId < 32) {
        m1 = maxValue[localId];
        m2 = maxValue[localId + 32];
        m3 = (m1 >= m2) ? m1 : m2;
        idx = (m1 >= m2) ? localId : localId + 32;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        m1 = maxValue[localId];
        m2 = maxValue[localId + 16];
        m3 = (m1 >= m2) ? m1 : m2;
        idx = (m1 >= m2) ? localId : localId + 16;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        m1 = maxValue[localId];
        m2 = maxValue[localId + 8];
        m3 = (m1 >= m2) ? m1 : m2;
        idx = (m1 >= m2) ? localId : localId + 8;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        m1 = maxValue[localId];
        m2 = maxValue[localId + 4];
        m3 = (m1 >= m2) ? m1 : m2;
        idx = (m1 >= m2) ? localId : localId + 4;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        m1 = maxValue[localId];
        m2 = maxValue[localId + 2];
        m3 = (m1 >= m2) ? m1 : m2;
        idx = (m1 >= m2) ? localId : localId + 2;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];

        m1 = maxValue[localId];
        m2 = maxValue[localId + 1];
        m3 = (m1 >= m2) ? m1 : m2;
        idx = (m1 >= m2) ? localId : localId + 1;
        maxValue[localId] = m3;
        maxInd[localId] = maxInd[idx];
    }
}

__kernel void viterbi_recursion_step(__global float *logA,
                                     __global float *logB,
                                     __global float *logPi,
                                     __global float *V,
                                     __global unsigned int *backtrace,
                                     __local float *maxValue,
                                     __local int *maxInd,
                                     uint N,
                                     uint M,
                                     unsigned int o,
                                     uint t)
{
    uint groupId = get_group_id(0);
    uint localId = get_local_id(0);
    uint localSize = get_local_size(0);
    uint iState = groupId;

    maxValue[localId] = -INFINITY;

    float mValue = -INFINITY;
    int mInd = -1;
    float value;
    for (uint i = localId; i < N; i += localSize) {
        value = V[(t - 1) * N + i] + logA[i * N + iState];
        if (value > mValue) {
            mValue = value;
            mInd = i;
        }
    }
    maxValue[localId] = mValue;
    maxInd[localId] = mInd;
    barrier(CLK_LOCAL_MEM_FENCE);

    local_maximum(maxValue, maxInd);

    // Copy results from local to global memory
    if (localId == 0) {
        V[t * N + iState] = maxValue[0] + logB[iState * M + o];
        backtrace[(t - 1) * N + iState] = maxInd[0];
    }

}

__kernel void viterbi_termination(__global float *V,
                                  __global unsigned int *maxState,
                                  __local float maxValue[],
                                  __local int maxInd[],
                                  uint N,
                                  uint T)
{
    uint groupId = get_group_id(0);
    uint localId = get_local_id(0);
    uint localSize = get_local_size(0);

    maxValue[localId] = -INFINITY;

    float mValue = -INFINITY;
    int mInd = -1;
    float value;
    for (int i = localId; i < N; i += localSize) {
        value = V[(T - 1) * N + i];
        if (value > mValue) {
            mValue = value;
            mInd = i;
        }
    }
    maxValue[localId] = mValue;
    maxInd[localId] = mInd;
    barrier(CLK_LOCAL_MEM_FENCE);

    local_maximum(maxValue, maxInd);

    if (localId == 0) {
        *maxState = maxInd[0];
    }
}

__kernel void viterbi_path(__global unsigned int *backtrace,
                           __global unsigned int *maxState,
                           uint N,
                           __global unsigned int *path,
                           uint T
)
{
    path[T - 1] = *maxState;
    for (int t = T - 2; t >= 0; t--) {
        path[t] = backtrace[t * N + path[t + 1]];
    }
}

__kernel void viterbi_recursion_step_seq(__global float *logA,
                                     __global float *logB,
                                     __global float *logPi,
                                     __global float *V,
                                     __global unsigned int *backtrace,
                                     __local float maxValue[],
                                     __local int maxInd[],
                                     uint N,
                                     uint M,
                                     unsigned int o,
                                     uint t)
{
    uint groupId = get_group_id(0);
    uint localId = get_local_id(0);
    uint localSize = get_local_size(0);
    uint iState = groupId;

    float mValue = -INFINITY;
    int mInd = -1;
    float value;

    for (int i = 0; i < N; i++) {
        value = V[(t - 1) * N + i] + logA[i * N + iState];
        if (value > mValue) {
            mValue = value;
            mInd = i;
        }
    }

    // Copy results from local to global memory
    V[t * N + iState] = mValue + logB[iState * M + o];
    backtrace[(t - 1) * N + iState] = mInd;
}
