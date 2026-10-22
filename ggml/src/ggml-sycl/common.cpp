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

পূর্ণাঙ্ক get_current_device_id() {
  প্রত্যাগম dpct::dev_mgr::instance().current_device_id();
}

শূন্য* ggml_sycl_host_malloc(size_t size) যত্ন {
  যদি (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    প্রত্যাগম nullptr;
  }

  শূন্য* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (শূন্য*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  যদি (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    প্রত্যাগম nullptr;
  }

  প্রত্যাগম ptr;
} গৃহাণ (sycl::exception ধ্রুব& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

শূন্য ggml_sycl_host_free(শূন্য* ptr) যত্ন {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::মুক্ত(ptr, dpct::get_in_order_queue())));
} গৃহাণ (sycl::exception ধ্রুব& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

তার্কিক gpu_has_xmx(sycl::device &dev) {
    প্রত্যাগম dev.has(sycl::aspect::ext_intel_matrix);
}

পূর্ণাঙ্ক ggml_sycl_get_env(ধ্রুব বর্ণ *env_name, পূর্ণাঙ্ক default_val) {
    বর্ণ *user_device_string = getenv(env_name);
    পূর্ণাঙ্ক user_সংখ্যা = default_val;

    unsigned n;
    যদি (user_device_string != রিক্ত &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_সংখ্যা = (পূর্ণাঙ্ক)n;
    } অন্যথা {
        user_সংখ্যা = default_val;
    }
    প্রত্যাগম user_সংখ্যা;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_অঙ্ক, int64_t block_size) {
  ধ্রুব int64_t max_range = std::numeric_limits<পূর্ণাঙ্ক>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_অঙ্ক * sycl_down_blk_size;
  যাবত্(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_অঙ্ক * sycl_down_blk_size;
  }
  প্রত্যাগম sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
স্থির তার্কিক ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    প্রত্যাগম g_ggml_sycl_use_level_zero_api &&
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
            ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE,
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
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        যদি (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            প্রত্যাগম ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        প্রত্যাগম nullptr;
    }
#endif
    শূন্য * ptr = sycl::malloc_device(size, q);
    যদি (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        প্রত্যাগম nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    প্রত্যাগম ptr;
}

শূন্য ggml_sycl_free_device(শূন্য *ptr, sycl::queue &q) {
    যদি (!ptr) প্রত্যাগম;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    যদি (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        প্রত্যাগম;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::মুক্ত(ptr, q)));
}

শূন্য release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    প্রতি (পূর্ণাঙ্ক i = 0; i < ggml_sycl_info().device_গুনতি; ++i) {
        প্রতি (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            যদি (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        যদি (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
