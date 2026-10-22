#pragma once

#شامل "common.hpp"

#define SYCL_SOLVE_TRI_MAX_N 64
#define SYCL_SOLVE_TRI_MAX_K 64

تهی ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
