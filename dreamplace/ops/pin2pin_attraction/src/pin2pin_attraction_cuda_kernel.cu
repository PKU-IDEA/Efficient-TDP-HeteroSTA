#include <cfloat>
#include <cmath>
#include "cuda_runtime.h"
#include "pin2pin_attraction/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

// Explicit computation mode enumeration 
enum ComputationMode {
    FORWARD_MODE = 0,
    BACKWARD_MODE = 1
};

template <typename T>
int pin2pinAttractionCudaLauncher(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, // Pin pairs (flat array of indices)
  const T *weights, // Weights for each pair, updated to use T
  int num_pairs,
  int num_pins, // Add actual number of pins parameter
  T *total_distance, // Corrected parameter order and type
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor,
  bool deterministic_flag,
  ComputationMode computation_mode // Explicit mode parameter
) {
  int thread_count = 64;
  int block_count = (num_pairs + thread_count - 1) / thread_count; // Correct calculation of blocks
  dim3 block_size(thread_count, 1, 1); // Simplified block dimensions

  if (deterministic_flag && computation_mode == FORWARD_MODE) {
    
    int fraction_bits = 32;   // determine optimal fraction bits based on experimental value range

    long long int scale_factor = (1LL << fraction_bits);
    // Run deterministic version with calculated scaling factor
    long long int *buf_total_distance;
    allocateCUDA(buf_total_distance, 1, long long int);

    // Initialize buffer to zero since total_distance starts from zero in each call
    checkCUDA(cudaMemset(buf_total_distance, 0, sizeof(long long int)));

    pin2pinAttractionCudaDeterministicForward<<<block_count, block_size>>>(
        pin_pos_x, pin_pos_y, pairs, weights, num_pairs,
        scale_factor, buf_total_distance
    );

    // Convert back to floating point using standard copyScaleArray
    copyScaleArray<<<1, 1>>>(total_distance, buf_total_distance, T(1.0 / scale_factor), 1);

    destroyCUDA(buf_total_distance);

  } else if (deterministic_flag && computation_mode == BACKWARD_MODE) {
    

    int fraction_bits = 32;

    long long int scale_factor = (1LL << fraction_bits);

    // Allocate buffers for deterministic computation
    long long int *buf_grad_x, *buf_grad_y;
    allocateCUDA(buf_grad_x, num_pins, long long int);
    allocateCUDA(buf_grad_y, num_pins, long long int);

    // Initialize buffers to zero to avoid double scaling
    int grad_thread_count = 256;
    int grad_block_count = (num_pins + grad_thread_count - 1) / grad_thread_count;
    checkCUDA(cudaMemset(buf_grad_x, 0, num_pins * sizeof(long long int)));
    checkCUDA(cudaMemset(buf_grad_y, 0, num_pins * sizeof(long long int)));

    // Run deterministic backward pass with optimal scaling
    pin2pinAttractionCudaDeterministicBackward<<<block_count, block_size>>>(
        pin_pos_x, pin_pos_y, pairs, weights, num_pairs, num_pins, grad_tensor,
        scale_factor, buf_grad_x, buf_grad_y
    );

    // Convert back to floating point using standard copyScaleArray
    copyScaleArray<<<grad_block_count, grad_thread_count>>>(
        grad_x_tensor, buf_grad_x, T(1.0 / scale_factor), num_pins);
    copyScaleArray<<<grad_block_count, grad_thread_count>>>(
        grad_y_tensor, buf_grad_y, T(1.0 / scale_factor), num_pins);

    // Clean up temporary buffers
  
    destroyCUDA(buf_grad_x);
    destroyCUDA(buf_grad_y);

  } else {
    // NON-DETERMINISTIC MODE
    if (computation_mode == BACKWARD_MODE) {
      pin2pinAttractionCudaBackward<<<block_count, block_size>>>(
          pin_pos_x, pin_pos_y,
          pairs, weights, num_pairs, total_distance,
          grad_tensor, grad_x_tensor, grad_y_tensor
      );
    } else { // FORWARD_MODE
      pin2pinAttractionCudaForward<<<block_count, block_size>>>(
          pin_pos_x, pin_pos_y,
          pairs, weights, num_pairs, total_distance,
          grad_tensor, grad_x_tensor, grad_y_tensor
      );
    }
  }

  cudaDeviceSynchronize(); // Ensure completion before return
  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                         \
template int pin2pinAttractionCudaLauncher<T>(              \
    const T *pin_pos_x, const T *pin_pos_y,                 \
    const int *pairs,                                       \
    const T *weights,                                       \
    int num_pairs,                                          \
    int num_pins,                                           \
    T *total_distance,                                      \
    const T *grad_tensor,                                   \
    T *grad_x_tensor, T *grad_y_tensor,                     \
    bool deterministic_flag,                                \
    ComputationMode computation_mode                        \
)

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE