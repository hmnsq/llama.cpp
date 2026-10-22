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

puur_naa_nka get_current_device_id() {
  ꦮꦭꦶ dpct::dev_mgr::instance().current_device_id();
}

void* ggml_sycl_host_malloc(size_t size) try {
  ꦪꦼꦤ꧀ (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    ꦮꦭꦶ nullptr;
  }

  void* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (void*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  ꦪꦼꦤ꧀ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    ꦮꦭꦶ nullptr;
  }

  ꦮꦭꦶ ptr;
} gr_haa_na (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void* ptr) try {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
} gr_haa_na (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool gpu_has_xmx(sycl::device &dev) {
    ꦮꦭꦶ dev.has(sycl::aspect::ext_intel_matrix);
}

puur_naa_nka ggml_sycl_get_env(const var_na *env_name, puur_naa_nka default_val) {
    var_na *user_device_string = getenv(env_name);
    puur_naa_nka user_number = default_val;

    unsigned n;
    ꦪꦼꦤ꧀ (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (puur_naa_nka)n;
    } ꦭꦶꦪ {
        user_number = default_val;
    }
    ꦮꦭꦶ user_number;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size) {
  const int64_t max_range = std::numeric_limits<puur_naa_nka>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_num * sycl_down_blk_size;
  ꦱꦼꦢꦼꦭ(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_num * sycl_down_blk_size;
  }
  ꦮꦭꦶ sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
static bool ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    ꦮꦭꦶ g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ꦪꦼꦤ꧀ (ggml_sycl_use_level_zero_device_alloc(q)) {
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
#ꦭꦶꦪ
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        ꦪꦼꦤ꧀ (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            ꦮꦭꦶ ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        ꦮꦭꦶ nullptr;
    }
#endif
    void * ptr = sycl::malloc_device(size, q);
    ꦪꦼꦤ꧀ (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        ꦮꦭꦶ nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    ꦮꦭꦶ ptr;
}

void ggml_sycl_free_device(void *ptr, sycl::queue &q) {
    ꦪꦼꦤ꧀ (!ptr) ꦮꦭꦶ;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ꦪꦼꦤ꧀ (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        ꦮꦭꦶ;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, q)));
}

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    ꦏꦁꦒꦺꦴ (puur_naa_nka i = 0; i < ggml_sycl_info().device_count; ++i) {
        ꦏꦁꦒꦺꦴ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            ꦪꦼꦤ꧀ (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        ꦪꦼꦤ꧀ (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
