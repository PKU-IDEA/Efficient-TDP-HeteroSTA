/**
 * @file   rudy_smooth_cuda.cpp
 * @author Xi Lin
 * @date   Dec 2023
 * @brief  Compute RudySmooth and gradient
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

// Forward declarations
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
  const T *weights, // Weights for each pair
  int num_pairs,
  int num_pins, // Add actual number of pins parameter
  T *total_distance,
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor,
  bool deterministic_flag,
  ComputationMode computation_mode
);


std::vector<at::Tensor> pin2pin_attraction_forward(
    at::Tensor pin_pos,
    at::Tensor pairs,
    at::Tensor weights,
    bool deterministic
) {
  CHECK_FLAT_CUDA(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  CHECK_FLAT_CUDA(pairs);
  CHECK_CONTIGUOUS(pairs);

  CHECK_FLAT_CUDA(weights);
  CHECK_CONTIGUOUS(weights);

  int num_pins = pin_pos.numel() / 2;
  int num_pairs = pairs.numel() / 2;

  at::Tensor total_distance = at::zeros({}, pin_pos.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pin_pos, "pin2pinAttractionCudaLauncher", [&] {
        // template <typename T>
        pin2pinAttractionCudaLauncher<scalar_t>(
            // const T *pin_pos_x, *pin_pos_y
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
            // dict pin2pin_net_weight
            DREAMPLACE_TENSOR_DATA_PTR(pairs, int),
            DREAMPLACE_TENSOR_DATA_PTR(weights, scalar_t),
            // int num_pairs
            num_pairs,
            // int num_pins
            num_pins,
            // forward output total_distance
            DREAMPLACE_TENSOR_DATA_PTR(total_distance, scalar_t),
            // const T *grad_tensor
            nullptr,
            // T *grad_x_tensor, *grad_y_tensor
            nullptr,
            nullptr,
            // bool deterministic_flag
            deterministic,
            // ComputationMode computation_mode
            FORWARD_MODE
        );
      });
  return {total_distance};
}

at::Tensor pin2pin_attraction_backward(
    at::Tensor grad,
    at::Tensor pin_pos,
    at::Tensor pairs,
    at::Tensor weights,
    bool deterministic
) {
  CHECK_FLAT_CUDA(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  CHECK_FLAT_CUDA(pairs);
  CHECK_CONTIGUOUS(pairs);

  CHECK_FLAT_CUDA(weights);
  CHECK_CONTIGUOUS(weights);

  int num_pins = pin_pos.numel() / 2; 
  int num_pairs = pairs.numel() / 2;

  at::Tensor grad_out = at::zeros_like(pin_pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pin_pos, "pin2pinAttractionCudaLauncher", [&] {
        // template <typename T>
        pin2pinAttractionCudaLauncher<scalar_t>(
            // const T *pin_pos_x, *pin_pos_y
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
            // dict pin2pin_net_weight
            DREAMPLACE_TENSOR_DATA_PTR(pairs, int),
            DREAMPLACE_TENSOR_DATA_PTR(weights, scalar_t),
            // int num_pairs
            num_pairs,
            // int num_pins
            num_pins,
            // forward output total_distance
            nullptr,
            // T *grad_tensor
            DREAMPLACE_TENSOR_DATA_PTR(grad, scalar_t),
            // T *grad_x_tensor, *grad_y_tensor
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins,
            // bool deterministic_flag
            deterministic,
            // ComputationMode computation_mode
            BACKWARD_MODE
          );
      });
  return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin2pin_attraction_forward,
        "Pin2PinAttraction forward (CUDA)",
        py::arg("pin_pos"), py::arg("pairs"), py::arg("weights"), py::arg("deterministic") = true);
  m.def("backward", &DREAMPLACE_NAMESPACE::pin2pin_attraction_backward,
        "Pin2PinAttraction backward (CUDA)",
        py::arg("grad"), py::arg("pin_pos"), py::arg("pairs"), py::arg("weights"), py::arg("deterministic") = true);
}
