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

#समावेश "common.hpp"
#समावेश <sycl/backend.hpp>
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#समावेश <level_zero/ze_api.h>
#पूर्ण_अगर

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-impl.h"

पूर्णांक get_current_device_id() {
  वापस dpct::dev_mgr::instance().current_device_id();
}

व्योम* ggml_sycl_host_malloc(माप_प्रकार size) प्रयास {
  अगर (दो_पर्या("GGML_SYCL_NO_PINNED") != nullptr) {
    वापस nullptr;
  }

  व्योम* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (व्योम*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  अगर (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    वापस nullptr;
  }

  वापस ptr;
} पकड़ो (sycl::exception स्थिर& exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

व्योम ggml_sycl_host_free(व्योम* ptr) प्रयास {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, dpct::get_in_order_queue())));
} पकड़ो (sycl::exception स्थिर& exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

bool gpu_has_xmx(sycl::device &dev) {
    वापस dev.has(sycl::aspect::ext_intel_matrix);
}

पूर्णांक ggml_sycl_get_env(स्थिर अक्षर *env_name, पूर्णांक default_val) {
    अक्षर *user_device_string = दो_पर्या(env_name);
    पूर्णांक user_संख्या = default_val;

    अचिन्हित n;
    अगर (user_device_string != शून्य &&
        माला_पूछो(user_device_string, " %u", &n) == 1) {
        user_संख्या = (पूर्णांक)n;
    } अन्यथा {
        user_संख्या = default_val;
    }
    वापस user_संख्या;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_अंक, int64_t block_आकार_2) {
  स्थिर int64_t max_range = std::numeric_limits<पूर्णांक>::max();
  int64_t sycl_down_blk_आकार = block_आकार_2;
  int64_t global_range = accumulate_block_अंक * sycl_down_blk_आकार;
  जबतक(global_range > max_range) {
      sycl_down_blk_आकार /= 2;
      global_range = accumulate_block_अंक * sycl_down_blk_आकार;
  }
  वापस sycl_down_blk_आकार;
}

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
अटल bool ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    वापस g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#पूर्ण_अगर

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
व्योम * ggml_sycl_malloc_device(माप_प्रकार size, sycl::queue &q, ggml_sycl_mem_type type) {
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    अगर (ggml_sycl_use_level_zero_device_alloc(q)) {
        व्योम *ptr = nullptr;
        स्वतः ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        स्वतः ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#अगर_घोषित ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME
        ze_relaxed_allocation_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
            nullptr,
            ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_आकार,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#अन्यथा
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#पूर्ण_अगर
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_परिणाम_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_बफर_ALIGNMENT, ze_dev, &ptr);
        अगर (r == ZE_परिणाम_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            वापस ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        वापस nullptr;
    }
#पूर्ण_अगर
    व्योम * ptr = sycl::malloc_device(size, q);
    अगर (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        वापस nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    वापस ptr;
}

व्योम ggml_sycl_free_device(व्योम *ptr, sycl::queue &q) {
    अगर (!ptr) वापस;
    ggml_sycl_memtrace_del(ptr);
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    अगर (ggml_sycl_use_level_zero_device_alloc(q)) {
        स्वतः ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        वापस;
    }
#पूर्ण_अगर
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, q)));
}

व्योम release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        क्रम (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            अगर (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        अगर (extra->तथ्य_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->तथ्य_device[i], *(streams[i]))));
        }
    }
    मिटाओ extra;
}
