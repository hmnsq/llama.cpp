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

#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#અન્તર્ભાવ <level_zero/ze_api.h>
#endif

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-impl.h"

પૂર્ણાંક get_current_device_id() {
  પાછા dpct::dev_mgr::instance().current_device_id();
}

શૂન્ય* ggml_sycl_host_malloc(size_t size) પ્રયાસ {
  જો (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    પાછા nullptr;
  }

  શૂન્ય* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (શૂન્ય*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  જો (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    પાછા nullptr;
  }

  પાછા ptr;
} પકડો (sycl::exception અચલ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

શૂન્ય ggml_sycl_host_free(શૂન્ય* ptr) પ્રયાસ {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::મુક્ત(ptr, dpct::get_in_order_queue())));
} પકડો (sycl::exception અચલ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

તાર્કિક gpu_has_xmx(sycl::device &dev) {
    પાછા dev.has(sycl::aspect::ext_intel_matrix);
}

પૂર્ણાંક ggml_sycl_get_env(અચલ અક્ષર *env_name, પૂર્ણાંક default_val) {
    અક્ષર *user_device_string = getenv(env_name);
    પૂર્ણાંક user_સંખ્યા = default_val;

    unsigned n;
    જો (user_device_string != ખાલી &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_સંખ્યા = (પૂર્ણાંક)n;
    } નહીંતર {
        user_સંખ્યા = default_val;
    }
    પાછા user_સંખ્યા;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_અંક, int64_t block_size) {
  અચલ int64_t max_range = std::numeric_limits<પૂર્ણાંક>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_અંક * sycl_down_blk_size;
  જ્યારે(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_અંક * sycl_down_blk_size;
  }
  પાછા sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
સ્થિર તાર્કિક ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    પાછા g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
શૂન્ય * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    જો (ggml_sycl_use_level_zero_device_alloc(q)) {
        શૂન્ય *ptr = nullptr;
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
#નહીંતર
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        જો (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            પાછા ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        પાછા nullptr;
    }
#endif
    શૂન્ય * ptr = sycl::malloc_device(size, q);
    જો (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        પાછા nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    પાછા ptr;
}

શૂન્ય ggml_sycl_free_device(શૂન્ય *ptr, sycl::queue &q) {
    જો (!ptr) પાછા;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    જો (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        પાછા;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::મુક્ત(ptr, q)));
}

શૂન્ય release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        માટે (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            જો (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        જો (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
