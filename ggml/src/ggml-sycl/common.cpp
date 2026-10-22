//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ <sycl/backend.hpp>
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ಅನ್ತರ್ಭಾವ <level_zero/ze_api.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

ಪೂರ್ಣಾಂಕ get_current_device_id() {
  ಹಿಂತಿರುಗಿ dpct::dev_mgr::instance().current_device_id();
}

ಶೂನ್ಯ* ggml_sycl_host_malloc(size_t size) ಪ್ರಯತ್ನಿಸು {
  ಆದರೆ (getenv("GGML_SYCL_NO_PINNED") != ಶೂನ್ಯಸೂಚಕ) {
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
  }

  ಶೂನ್ಯ* ptr = ಶೂನ್ಯಸೂಚಕ;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (ಶೂನ್ಯ*)sycl::malloc_host(size, dpct::get_ಒಳ_ಕ್ರಮ_ಸರತಿ()));

  ಆದರೆ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
  }

  ಹಿಂತಿರುಗಿ ptr;
} ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಶೂನ್ಯ ggml_sycl_host_free(ಶೂನ್ಯ* ptr) ಪ್ರಯತ್ನಿಸು {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಬಿಡುಗಡೆ(ptr, dpct::get_ಒಳ_ಕ್ರಮ_ಸರತಿ())));
} ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ತಾರ್ಕಿಕ gpu_has_xmx(sycl::device &dev) {
    ಹಿಂತಿರುಗಿ dev.has(sycl::aspect::ext_intel_ಮಾತೃಕೆ);
}

ಪೂರ್ಣಾಂಕ ggml_sycl_get_env(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *env_name, ಪೂರ್ಣಾಂಕ default_val) {
    ಅಕ್ಷರ *user_device_string = getenv(env_name);
    ಪೂರ್ಣಾಂಕ user_ಸಂಖ್ಯೆ = default_val;

    ಚಿಹ್ನೆರಹಿತ n;
    ಆದರೆ (user_device_string != ಖಾಲಿ &&
        ಅಕ್ಷರಸರಣಿ_ಓದು(user_device_string, " %u", &n) == 1) {
        user_ಸಂಖ್ಯೆ = (ಪೂರ್ಣಾಂಕ)n;
    } ಇಲ್ಲದಿದ್ದರೆ {
        user_ಸಂಖ್ಯೆ = default_val;
    }
    ಹಿಂತಿರುಗಿ user_ಸಂಖ್ಯೆ;
}

int64_t downsample_sycl_global_range(int64_t accumulate_ಖಂಡ_ಅಂಕೆ, int64_t ಖಂಡ_ಗಾತ್ರ_3) {
  ಸ್ಥಿರಾಂಕ int64_t max_range = std::numeric_limits<ಪೂರ್ಣಾಂಕ>::max();
  int64_t sycl_ಕೆಳಕ್ಕೆ_blk_ಗಾತ್ರ = ಖಂಡ_ಗಾತ್ರ_3;
  int64_t global_range = accumulate_ಖಂಡ_ಅಂಕೆ * sycl_ಕೆಳಕ್ಕೆ_blk_ಗಾತ್ರ;
  ಆಗ(global_range > max_range) {
      sycl_ಕೆಳಕ್ಕೆ_blk_ಗಾತ್ರ /= 2;
      global_range = accumulate_ಖಂಡ_ಅಂಕೆ * sycl_ಕೆಳಕ್ಕೆ_blk_ಗಾತ್ರ;
  }
  ಹಿಂತಿರುಗಿ sycl_ಕೆಳಕ್ಕೆ_blk_ಗಾತ್ರ;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#ಆದರೆ_ಅಂತ್ಯ

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
ಶೂನ್ಯ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ಆದರೆ (ggml_sycl_use_level_zero_device_alloc(q)) {
        ಶೂನ್ಯ *ptr = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ವಯಂಚಾಲಿತ ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ಸ್ವಯಂಚಾಲಿತ ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ZE_RELAXED_ಹಂಚಿಕೆ_LIMITS_EXP_NAME
        ze_relaxed_ಹಂಚಿಕೆ_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ಹಂಚಿಕೆ_LIMITS_EXP_DESC,
            ಶೂನ್ಯಸೂಚಕ,
            ZE_RELAXED_ಹಂಚಿಕೆ_LIMITS_EXP_FLAG_MAX_ಗಾತ್ರ,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#ಇಲ್ಲದಿದ್ದರೆ
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, ಶೂನ್ಯಸೂಚಕ, 0, 0};
#ಆದರೆ_ಅಂತ್ಯ
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_ಫಲಿತಾಂಶ_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_ಬಫರ್_ALIGNMENT, ze_dev, &ptr);
        ಆದರೆ (r == ZE_ಫಲಿತಾಂಶ_SUCCESS && ptr) {
            ggml_sycl_memtrace_ಕೂಡಿಸು(type, ptr, size);
            ಹಿಂತಿರುಗಿ ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಶೂನ್ಯ * ptr = sycl::malloc_device(size, q);
    ಆದರೆ (ptr == ಶೂನ್ಯಸೂಚಕ) {
        ggml_sycl_memtrace_fail(type, size);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ggml_sycl_memtrace_ಕೂಡಿಸು(type, ptr, size);
    ಹಿಂತಿರುಗಿ ptr;
}

ಶೂನ್ಯ ggml_sycl_free_device(ಶೂನ್ಯ *ptr, sycl::queue &q) {
    ಆದರೆ (!ptr) ಹಿಂತಿರುಗಿ;
    ggml_sycl_memtrace_del(ptr);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ಆದರೆ (ggml_sycl_use_level_zero_device_alloc(q)) {
        ಸ್ವಯಂಚಾಲಿತ ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        ಹಿಂತಿರುಗಿ;
    }
#ಆದರೆ_ಅಂತ್ಯ
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಬಿಡುಗಡೆ(ptr, q)));
}

ಶೂನ್ಯ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<ಸರತಿ_ptr> streams) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        ಗಾಗಿ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            ಆದರೆ (extra->events[i][is] != ಶೂನ್ಯಸೂಚಕ) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        ಆದರೆ (extra->ದತ್ತಾಂಶ_device[i] != ಶೂನ್ಯಸೂಚಕ && streams.size()>0) {
            ggml_sycl_ಗಣ_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->ದತ್ತಾಂಶ_device[i], *(streams[i]))));
        }
    }
    ಅಳಿಸು extra;
}
