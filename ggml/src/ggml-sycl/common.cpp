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

पूर्णांक get_current_device_id() {
  परत dpct::dev_mgr::instance().current_device_id();
}

शून्य* ggml_sycl_host_malloc(माप_प्रकार size) प्रयत्न {
  जर (दो_पर्या("GGML_SYCL_NO_PINNED") != nullptr) {
    परत nullptr;
  }

  शून्य* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (शून्य*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  जर (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    परत nullptr;
  }

  परत ptr;
} पकडा (sycl::exception स्थिरांक& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

शून्य ggml_sycl_host_free(शून्य* ptr) प्रयत्न {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, dpct::get_in_order_queue())));
} पकडा (sycl::exception स्थिरांक& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

तार्किक gpu_has_xmx(sycl::device &dev) {
    परत dev.has(sycl::aspect::ext_intel_matrix);
}

पूर्णांक ggml_sycl_get_env(स्थिरांक अक्षर *env_name, पूर्णांक default_val) {
    अक्षर *user_device_string = दो_पर्या(env_name);
    पूर्णांक user_संख्या = default_val;

    अचिन्हित n;
    जर (user_device_string != रिक्त &&
        माला_पूछो(user_device_string, " %u", &n) == 1) {
        user_संख्या = (पूर्णांक)n;
    } अन्यथा {
        user_संख्या = default_val;
    }
    परत user_संख्या;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_अंक, int64_t block_size) {
  स्थिरांक int64_t max_range = std::numeric_limits<पूर्णांक>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_अंक * sycl_down_blk_size;
  पर्यंत(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_अंक * sycl_down_blk_size;
  }
  परत sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
स्थिर तार्किक ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    परत g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
शून्य * ggml_sycl_malloc_device(माप_प्रकार size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    जर (ggml_sycl_use_level_zero_device_alloc(q)) {
        शून्य *ptr = nullptr;
        स्वतः ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        स्वतः ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
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
#अन्यथा
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        जर (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            परत ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        परत nullptr;
    }
#endif
    शून्य * ptr = sycl::malloc_device(size, q);
    जर (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        परत nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    परत ptr;
}

शून्य ggml_sycl_free_device(शून्य *ptr, sycl::queue &q) {
    जर (!ptr) परत;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    जर (ggml_sycl_use_level_zero_device_alloc(q)) {
        स्वतः ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        परत;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, q)));
}

शून्य release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        साठी (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            जर (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        जर (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
