/**
 * @author      Peter Gazdík <xgazdi03(at)stud.fit.vutbr.cz>
 *              Michal Klčo <xklcom00(at)stud.fit.vutbr.cz>
 * @date        17/12/17
 * @copyright   The MIT License (MIT)
 */


float add_log(float x, float y)
{
    float result = 0;
    if (x == -INFINITY && y == -INFINITY) {
        // just return -INFINITY
        return x;
    }
    if (x >= y) {
        result = x + log1p(exp(y - x));
    } else {
        result = y + log1p(exp(x - y));
    }

    return result;
}

__kernel void forward_init(__global float *logA,
                           __global float *logB,
                           __global float *logPi,
                           __global float *alpha,
                           unsigned int num_states,
                           unsigned int M,
                           unsigned int o)
{
    uint groupId = get_group_id(0);
    uint iState = groupId;

    // Initialization
    if (iState < num_states) {
        alpha[iState] = logPi[iState] + logB[iState * M + o];
    }
}


__kernel void forward_step(__global float *logA,
                         __global float *logB,
                         __global float *logPi,
                         __global float *alpha,
                         __local float *local_sum,
                         uint num_states,
                         uint num_observ,
                         unsigned int o,
                         uint step)
{
    int local_id = (int) get_local_id(0);
    int local_size = (int) get_local_size(0);
    int group_id = (int) get_group_id(0);

    local_sum[local_id] = -INFINITY;

    barrier(CLK_LOCAL_MEM_FENCE);

    float temp = -INFINITY;
    for (uint i = local_id; i < num_states; i += local_size) {
        temp = alpha[(step - 1) * num_states + i] + logA[i * num_states + group_id] +
               logB[group_id * num_observ + o];
//        printf("t: %d, State %d, Temp: %f\n", step, group_id, temp);

        local_sum[local_id] = add_log(temp, local_sum[local_id]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = (local_size >> 1); s > 0; s >>= 1) {
        if (local_id < s) {
            local_sum[local_id] = add_log(local_sum[local_id],
                                          local_sum[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        alpha[step * num_states + group_id] = local_sum[0];
    }
}

__kernel void forward_termination(__global float *alpha,
                                  __global float *likelihood,
                                  __local float local_sum[],
                                  uint num_states,
                                  uint T)
{
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint local_size = get_local_size(0);

    local_sum[local_id] = -INFINITY;

    for (int i = local_id; i < num_states; i += local_size) {
        local_sum[local_id] = add_log(local_sum[local_id], alpha[(T-1) * num_states + i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = (local_size >> 1); s > 0; s >>= 1) {
        if (local_id < s) {
            local_sum[local_id] = add_log(local_sum[local_id],
                                          local_sum[local_id + s]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (local_id == 0) {
        *likelihood = local_sum[0];
    }
}

