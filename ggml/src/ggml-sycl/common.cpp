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

#include "common.hpp"
#include <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#endif

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

정수 get_current_device_id() {
  반환 dpct::dev_mgr::instance().current_device_id();
}

void* ggml_sycl_host_malloc(size_t size) try {
  만약 (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    반환 nullptr;
  }

  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  만약 (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    반환 nullptr;
  }

  반환 ptr;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void* ptr) try {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool gpu_has_xmx(sycl::device &dev) {
    반환 dev.has(sycl::aspect::ext_intel_matrix);
}

정수 ggml_sycl_get_env(const char *env_name, 정수 default_val) {
    char *user_device_string = getenv(env_name);
    정수 user_number = default_val;

    unsigned n;
    만약 (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (정수)n;
    } 아니면 {
        user_number = default_val;
    }
    반환 user_number;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size) {
  const int64_t max_range = std::numeric_limits<정수>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_num * sycl_down_blk_size;
  동안(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_num * sycl_down_blk_size;
  }
  반환 sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    반환 g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    만약 (ggml_sycl_use_level_zero_device_alloc(q)) {
        void *ptr = nullptr;
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
#아니면
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        만약 (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            반환 ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        반환 nullptr;
    }
#endif
    void * ptr = sycl::malloc_device(size, q);
    만약 (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        반환 nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    반환 ptr;
}

void ggml_sycl_free_device(void *ptr, sycl::queue &q) {
    만약 (!ptr) 반환;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    만약 (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        반환;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, q)));
}

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    위해 (정수 i = 0; i < ggml_sycl_info().device_count; ++i) {
        위해 (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            만약 (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        만약 (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
