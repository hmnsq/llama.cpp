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

#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#অন্তর্ভাব <level_zero/ze_api.h>
#endif

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-impl.h"

পূর্ণসংখ্যা get_current_device_id() {
  ফেরত dpct::dev_mgr::instance().current_device_id();
}

শূন্য* ggml_sycl_host_malloc(size_t size) চেষ্টা {
  যদি (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    ফেরত nullptr;
  }

  শূন্য* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (শূন্য*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  যদি (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    ফেরত nullptr;
  }

  ফেরত ptr;
} ধরো (sycl::exception ধ্রুবক& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

শূন্য ggml_sycl_host_free(শূন্য* ptr) চেষ্টা {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::মুক্ত(ptr, dpct::get_in_order_queue())));
} ধরো (sycl::exception ধ্রুবক& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

যৌক্তিক gpu_has_xmx(sycl::device &dev) {
    ফেরত dev.has(sycl::aspect::ext_intel_matrix);
}

পূর্ণসংখ্যা ggml_sycl_get_env(ধ্রুবক অক্ষর *env_name, পূর্ণসংখ্যা default_val) {
    অক্ষর *user_device_string = getenv(env_name);
    পূর্ণসংখ্যা user_সংখ্যা = default_val;

    unsigned n;
    যদি (user_device_string != ফাঁকা &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_সংখ্যা = (পূর্ণসংখ্যা)n;
    } অন্যথা {
        user_সংখ্যা = default_val;
    }
    ফেরত user_সংখ্যা;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_অঙ্ক, int64_t block_আকার_2) {
  ধ্রুবক int64_t max_range = std::numeric_limits<পূর্ণসংখ্যা>::max();
  int64_t sycl_down_blk_আকার = block_আকার_2;
  int64_t global_range = accumulate_block_অঙ্ক * sycl_down_blk_আকার;
  যখন(global_range > max_range) {
      sycl_down_blk_আকার /= 2;
      global_range = accumulate_block_অঙ্ক * sycl_down_blk_আকার;
  }
  ফেরত sycl_down_blk_আকার;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
স্থির যৌক্তিক ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    ফেরত g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
শূন্য * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    যদি (ggml_sycl_use_level_zero_device_alloc(q)) {
        শূন্য *ptr = nullptr;
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
        ze_relaxed_allocation_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
            nullptr,
            ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_আকার,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#অন্যথা
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_ফলাফল_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_বাফার_ALIGNMENT, ze_dev, &ptr);
        যদি (r == ZE_ফলাফল_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            ফেরত ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        ফেরত nullptr;
    }
#endif
    শূন্য * ptr = sycl::malloc_device(size, q);
    যদি (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        ফেরত nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    ফেরত ptr;
}

শূন্য ggml_sycl_free_device(শূন্য *ptr, sycl::queue &q) {
    যদি (!ptr) ফেরত;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    যদি (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        ফেরত;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::মুক্ত(ptr, q)));
}

শূন্য release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    জন্য (পূর্ণসংখ্যা i = 0; i < ggml_sycl_info().device_গুনতি; ++i) {
        জন্য (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            যদি (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        যদি (extra->তথ্য_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->তথ্য_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
