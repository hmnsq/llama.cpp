#pragma once

#अन्तर्भाव "common.hpp"

#define SYCL_SOLVE_TRI_MAX_N 64
#define SYCL_SOLVE_TRI_MAX_K 64

शून्य ggml_sycl_solve_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
