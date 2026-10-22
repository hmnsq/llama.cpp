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
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ಅನ್ತರ್ಭಾವ <level_zero/ze_api.h>
#endif

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

ಪೂರ್ಣಾಙ್ಕ get_current_device_id() {
  ಪ್ರತ್ಯಾಗಮ dpct::dev_mgr::instance().current_device_id();
}

ಶೂನ್ಯ* ggml_sycl_host_malloc(size_t size) ಯತ್ನ {
  ಯದಿ (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    ಪ್ರತ್ಯಾಗಮ nullptr;
  }

  ಶೂನ್ಯ* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (ಶೂನ್ಯ*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  ಯದಿ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    ಪ್ರತ್ಯಾಗಮ nullptr;
  }

  ಪ್ರತ್ಯಾಗಮ ptr;
} ಗೃಹಾಣ (sycl::exception ಧ್ರುವ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ಶೂನ್ಯ ggml_sycl_host_free(ಶೂನ್ಯ* ptr) ಯತ್ನ {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಮುಕ್ತ(ptr, dpct::get_in_order_queue())));
} ಗೃಹಾಣ (sycl::exception ಧ್ರುವ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ತಾರ್ಕಿಕ gpu_has_xmx(sycl::device &dev) {
    ಪ್ರತ್ಯಾಗಮ dev.has(sycl::aspect::ext_intel_matrix);
}

ಪೂರ್ಣಾಙ್ಕ ggml_sycl_get_env(ಧ್ರುವ ವರ್ಣ *env_name, ಪೂರ್ಣಾಙ್ಕ default_val) {
    ವರ್ಣ *user_device_string = getenv(env_name);
    ಪೂರ್ಣಾಙ್ಕ user_ಸಂಖ್ಯೆ = default_val;

    unsigned n;
    ಯದಿ (user_device_string != ರಿಕ್ತ &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_ಸಂಖ್ಯೆ = (ಪೂರ್ಣಾಙ್ಕ)n;
    } ಅನ್ಯಥಾ {
        user_ಸಂಖ್ಯೆ = default_val;
    }
    ಪ್ರತ್ಯಾಗಮ user_ಸಂಖ್ಯೆ;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_ಅಂಕೆ, int64_t block_size) {
  ಧ್ರುವ int64_t max_range = std::numeric_limits<ಪೂರ್ಣಾಙ್ಕ>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_ಅಂಕೆ * sycl_down_blk_size;
  ಯಾವತ್(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_ಅಂಕೆ * sycl_down_blk_size;
  }
  ಪ್ರತ್ಯಾಗಮ sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
ಶೂನ್ಯ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ಯದಿ (ggml_sycl_use_level_zero_device_alloc(q)) {
        ಶೂನ್ಯ *ptr = nullptr;
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
        ze_relaxed_allocation_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
            nullptr,
            ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#ಅನ್ಯಥಾ
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        ಯದಿ (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            ಪ್ರತ್ಯಾಗಮ ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
#endif
    ಶೂನ್ಯ * ptr = sycl::malloc_device(size, q);
    ಯದಿ (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    ಪ್ರತ್ಯಾಗಮ ptr;
}

ಶೂನ್ಯ ggml_sycl_free_device(ಶೂನ್ಯ *ptr, sycl::queue &q) {
    ಯದಿ (!ptr) ಪ್ರತ್ಯಾಗಮ;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ಯದಿ (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        ಪ್ರತ್ಯಾಗಮ;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಮುಕ್ತ(ptr, q)));
}

ಶೂನ್ಯ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        ಪ್ರತಿ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            ಯದಿ (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        ಯದಿ (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
