#ifndef GPUPLACE_RUDY_SMOOTH_FUNCTIONAL_H
#define GPUPLACE_RUDY_SMOOTH_FUNCTIONAL_H

#include<iostream>

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

// Optimized atomic add for signed long long integers using fixed-point arithmetic
template <typename T>
__device__ long long int atomicAddSigned(long long int* address, T value, long long int scale_factor) {
    // Scale the input value to fixed-point representation
    long long int scaled_value = (long long int)(value * scale_factor);

    // For newer GPU architectures, try to use native atomicAdd if available
    #if __CUDA_ARCH__ >= 600
    // Use native atomicAdd for long long (supported on sm_60+)
    return atomicAdd((unsigned long long int*)address, (unsigned long long int)scaled_value);
    #else
    // Fallback to atomicCAS for older architectures
    long long int old_val, new_val, assumed;
    old_val = *address;
    do {
        assumed = old_val;
        new_val = assumed + scaled_value;
        old_val = atomicCAS((unsigned long long int*)address,
                           (unsigned long long int)assumed,
                           (unsigned long long int)new_val);
    } while (assumed != old_val);
    return old_val;
    #endif
}



template <typename T>
__global__ void pin2pinAttractionCudaForward(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, // Pin pairs (flat array of indices)
  const T *weights, // Weights for each pair, updated to use T
  int num_pairs,
  T *total_distance, // Corrected parameter order and type
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int pin_id1 = pairs[2 * idx];
        int pin_id2 = pairs[2 * idx + 1];
        T dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
        T dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
        T distance = weights[idx] * (dx * dx + dy * dy);
        atomicAdd(total_distance, distance); // Atomic addition to accumulate the total distance
    }
}

template <typename T>
__global__ void pin2pinAttractionCudaBackward(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, // Pin pairs (flat array of indices)
  const T *weights, // Weights for each pair, updated to use T
  int num_pairs,
  T *total_distance, // Corrected parameter order and type
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int pin_id1 = pairs[2 * idx];
        int pin_id2 = pairs[2 * idx + 1];
        T dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
        T dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
        T grad_x = 2 * (*grad_tensor) * weights[idx] * dx; // Assuming grad_input is a scalar
        T grad_y = 2 * (*grad_tensor) * weights[idx] * dy;

        atomicAdd(&grad_x_tensor[pin_id1], grad_x);
        atomicAdd(&grad_y_tensor[pin_id1], grad_y);
        atomicAdd(&grad_x_tensor[pin_id2], -grad_x);
        atomicAdd(&grad_y_tensor[pin_id2], -grad_y);
    }
}

// Deterministic forward kernel using fixed-point arithmetic
template <typename T>
__global__ void pin2pinAttractionCudaDeterministicForward(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, const T *weights, int num_pairs,
  long long int scale_factor,
  long long int *buf_total_distance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int pin_id1 = pairs[2 * idx];
        int pin_id2 = pairs[2 * idx + 1];
        T dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
        T dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
        T distance = weights[idx] * (dx * dx + dy * dy);
        // Use custom signed atomic add
        atomicAddSigned(buf_total_distance, distance, scale_factor);
    }
}

// Deterministic backward kernel using fixed-point arithmetic
template <typename T>
__global__ void pin2pinAttractionCudaDeterministicBackward(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, const T *weights, int num_pairs,
  int num_pins, // Add num_pins parameter for bounds checking
  const T *grad_tensor,
  long long int scale_factor,
  long long int *buf_grad_x,
  long long int *buf_grad_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int pin_id1 = pairs[2 * idx];
        int pin_id2 = pairs[2 * idx + 1];

        // Bounds checking
        if (pin_id1 >= num_pins || pin_id2 >= num_pins || pin_id1 < 0 || pin_id2 < 0) {
            return; // Skip this pair
        }

        T dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
        T dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
        T grad_x = 2 * (*grad_tensor) * weights[idx] * dx;
        T grad_y = 2 * (*grad_tensor) * weights[idx] * dy;

        // Apply atomic operations using custom signed atomic add
        atomicAddSigned(&buf_grad_x[pin_id1], grad_x, scale_factor);
        atomicAddSigned(&buf_grad_y[pin_id1], grad_y, scale_factor);
        atomicAddSigned(&buf_grad_x[pin_id2], -grad_x, scale_factor);
        atomicAddSigned(&buf_grad_y[pin_id2], -grad_y, scale_factor);
    }
}


DREAMPLACE_END_NAMESPACE

#endif