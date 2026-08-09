#include <math.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t checked_index(float value, uint64_t upper) {
    if (!isfinite(value) || value < 0.0f || truncf(value) != value || value >= upper) {
        asm("trap;");
    }
    return static_cast<uint64_t>(value);
}

extern "C" __global__ void embedding(
    const float* weight,
    uint64_t weight_len,
    const float* indices,
    uint64_t index_count,
    float* output,
    uint64_t output_len,
    uint64_t vocab_size,
    uint64_t width
) {
    for (uint64_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
         output_index < output_len;
         output_index += blockDim.x * gridDim.x) {
        uint64_t index_position = output_index / width;
        uint64_t column = output_index % width;
        if (index_position >= index_count) {
            asm("trap;");
        }
        uint64_t row = checked_index(indices[index_position], vocab_size);
        uint64_t weight_index = row * width + column;
        if (weight_index >= weight_len) {
            asm("trap;");
        }
        output[output_index] = weight[weight_index];
    }
}

extern "C" __global__ void embedding_backward(
    const float* indices,
    uint64_t index_count,
    const float* grad_output,
    uint64_t grad_output_len,
    float* grad_weight,
    uint64_t grad_weight_len,
    uint64_t vocab_size,
    uint64_t width
) {
    for (uint64_t grad_index = blockIdx.x * blockDim.x + threadIdx.x;
         grad_index < grad_output_len;
         grad_index += blockDim.x * gridDim.x) {
        uint64_t index_position = grad_index / width;
        uint64_t column = grad_index % width;
        if (index_position >= index_count) {
            asm("trap;");
        }
        uint64_t row = checked_index(indices[index_position], vocab_size);
        uint64_t weight_index = row * width + column;
        if (weight_index >= grad_weight_len) {
            asm("trap;");
        }
        atomicAdd(&grad_weight[weight_index], grad_output[grad_index]);
    }
}

extern "C" __global__ void indexed_cross_entropy(
    const float* logits,
    uint64_t logits_len,
    const float* targets,
    uint64_t target_count,
    float* losses,
    uint64_t vocab_size
) {
    for (uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < target_count;
         row += blockDim.x * gridDim.x) {
        uint64_t target = checked_index(targets[row], vocab_size);
        uint64_t start = row * vocab_size;
        if (start + vocab_size > logits_len) {
            asm("trap;");
        }
        float maximum = -INFINITY;
        for (uint64_t column = 0; column < vocab_size; ++column) {
            maximum = fmaxf(maximum, logits[start + column]);
        }
        float denominator = 0.0f;
        for (uint64_t column = 0; column < vocab_size; ++column) {
            denominator += expf(logits[start + column] - maximum);
        }
        losses[row] = maximum + logf(denominator) - logits[start + target];
    }
}

extern "C" __global__ void indexed_cross_entropy_backward(
    const float* logits,
    uint64_t logits_len,
    const float* targets,
    uint64_t target_count,
    const float* grad_output,
    uint64_t grad_output_len,
    float* grad_logits,
    uint64_t vocab_size
) {
    for (uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < target_count;
         row += blockDim.x * gridDim.x) {
        if (row >= grad_output_len) {
            asm("trap;");
        }
        uint64_t target = checked_index(targets[row], vocab_size);
        uint64_t start = row * vocab_size;
        if (start + vocab_size > logits_len) {
            asm("trap;");
        }
        float maximum = -INFINITY;
        for (uint64_t column = 0; column < vocab_size; ++column) {
            maximum = fmaxf(maximum, logits[start + column]);
        }
        float denominator = 0.0f;
        for (uint64_t column = 0; column < vocab_size; ++column) {
            denominator += expf(logits[start + column] - maximum);
        }
        float upstream = grad_output[row];
        for (uint64_t column = 0; column < vocab_size; ++column) {
            float probability = expf(logits[start + column] - maximum) / denominator;
            grad_logits[start + column] = upstream * (probability - (column == target));
        }
    }
}
