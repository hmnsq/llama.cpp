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

#अन्तर्भाव "common.hpp"
#अन्तर्भाव <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#अन्तर्भाव <level_zero/ze_api.h>
#endif

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"

पूर्णाङ्क get_current_device_id() {
  प्रत्यागम dpct::dev_mgr::instance().current_device_id();
}

शून्य* ggml_sycl_host_malloc(size_t size) यत्न {
  जइ (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    प्रत्यागम nullptr;
  }

  शून्य* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (शून्य*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  जइ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    प्रत्यागम nullptr;
  }

  प्रत्यागम ptr;
} गृहाण (sycl::exception ध्रुव& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

शून्य ggml_sycl_host_free(शून्य* ptr) यत्न {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, dpct::get_in_order_queue())));
} गृहाण (sycl::exception ध्रुव& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

तार्किक gpu_has_xmx(sycl::device &dev) {
    प्रत्यागम dev.has(sycl::aspect::ext_intel_matrix);
}

पूर्णाङ्क ggml_sycl_get_env(ध्रुव वर्ण *env_name, पूर्णाङ्क default_val) {
    वर्ण *user_device_string = getenv(env_name);
    पूर्णाङ्क user_सङ्ख्या = default_val;

    unsigned n;
    जइ (user_device_string != रिक्त &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_सङ्ख्या = (पूर्णाङ्क)n;
    } अण्णहा {
        user_सङ्ख्या = default_val;
    }
    प्रत्यागम user_सङ्ख्या;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_अङ्कः, int64_t block_size) {
  ध्रुव int64_t max_range = std::numeric_limits<पूर्णाङ्क>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_अङ्कः * sycl_down_blk_size;
  जाव(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_अङ्कः * sycl_down_blk_size;
  }
  प्रत्यागम sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
स्थिर तार्किक ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    प्रत्यागम g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
शून्य * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    जइ (ggml_sycl_use_level_zero_device_alloc(q)) {
        शून्य *ptr = nullptr;
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
#अण्णहा
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        जइ (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            प्रत्यागम ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        प्रत्यागम nullptr;
    }
#endif
    शून्य * ptr = sycl::malloc_device(size, q);
    जइ (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        प्रत्यागम nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    प्रत्यागम ptr;
}

शून्य ggml_sycl_free_device(शून्य *ptr, sycl::queue &q) {
    जइ (!ptr) प्रत्यागम;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    जइ (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        प्रत्यागम;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, q)));
}

शून्य release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        प्रति (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            जइ (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        जइ (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
