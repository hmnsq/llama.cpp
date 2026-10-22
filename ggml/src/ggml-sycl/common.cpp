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

#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ଅନ୍ତର୍ଭାଵ <level_zero/ze_api.h>
#endif

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_current_device_id() {
  ଫେରନ୍ତୁ dpct::dev_mgr::instance().current_device_id();
}

ଶୂନ୍ୟ* ggml_sycl_host_malloc(size_t size) ଚେଷ୍ଟା {
  ଯଦି (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    ଫେରନ୍ତୁ nullptr;
  }

  ଶୂନ୍ୟ* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (ଶୂନ୍ୟ*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  ଯଦି (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    ଫେରନ୍ତୁ nullptr;
  }

  ଫେରନ୍ତୁ ptr;
} ଧର (sycl::exception ଅଚଳ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ଶୂନ୍ୟ ggml_sycl_host_free(ଶୂନ୍ୟ* ptr) ଚେଷ୍ଟା {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::ମୁକ୍ତ(ptr, dpct::get_in_order_queue())));
} ଧର (sycl::exception ଅଚଳ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ତାର୍କିକ gpu_has_xmx(sycl::device &dev) {
    ଫେରନ୍ତୁ dev.has(sycl::aspect::ext_intel_matrix);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_get_env(ଅଚଳ ଅକ୍ଷର *env_name, ପୂର୍ଣ୍ଣସଂଖ୍ୟା default_val) {
    ଅକ୍ଷର *user_device_string = getenv(env_name);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା user_ସଂଖ୍ୟା = default_val;

    unsigned n;
    ଯଦି (user_device_string != ଖାଲି &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_ସଂଖ୍ୟା = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)n;
    } ଅନ୍ୟଥା {
        user_ସଂଖ୍ୟା = default_val;
    }
    ଫେରନ୍ତୁ user_ସଂଖ୍ୟା;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_ଅଙ୍କ, int64_t block_size) {
  ଅଚଳ int64_t max_range = std::numeric_limits<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_ଅଙ୍କ * sycl_down_blk_size;
  ଯେତେବେଳେ(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_ଅଙ୍କ * sycl_down_blk_size;
  }
  ଫେରନ୍ତୁ sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ସ୍ଥିର ତାର୍କିକ ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    ଫେରନ୍ତୁ g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
ଶୂନ୍ୟ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ଯଦି (ggml_sycl_use_level_zero_device_alloc(q)) {
        ଶୂନ୍ୟ *ptr = nullptr;
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
#ଅନ୍ୟଥା
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        ଯଦି (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            ଫେରନ୍ତୁ ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        ଫେରନ୍ତୁ nullptr;
    }
#endif
    ଶୂନ୍ୟ * ptr = sycl::malloc_device(size, q);
    ଯଦି (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        ଫେରନ୍ତୁ nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    ଫେରନ୍ତୁ ptr;
}

ଶୂନ୍ୟ ggml_sycl_free_device(ଶୂନ୍ୟ *ptr, sycl::queue &q) {
    ଯଦି (!ptr) ଫେରନ୍ତୁ;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ଯଦି (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        ଫେରନ୍ତୁ;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::ମୁକ୍ତ(ptr, q)));
}

ଶୂନ୍ୟ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        ପାଇଁ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            ଯଦି (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        ଯଦି (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
