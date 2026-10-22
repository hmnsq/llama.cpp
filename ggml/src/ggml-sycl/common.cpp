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

#ضمن "common.hpp"
#ضمن <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#endif

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"

صحيح get_current_device_id() {
  أرجع dpct::dev_mgr::instance().current_device_id();
}

فراغ* ggml_sycl_host_malloc(size_t size) حاول {
  إذا (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    أرجع nullptr;
  }

  فراغ* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (فراغ*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  إذا (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    أرجع nullptr;
  }

  أرجع ptr;
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

فراغ ggml_sycl_host_free(فراغ* ptr) حاول {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, dpct::get_in_order_queue())));
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

منطقي gpu_has_xmx(sycl::device &dev) {
    أرجع dev.has(sycl::aspect::ext_intel_matrix);
}

صحيح ggml_sycl_get_env(ثابت حرف *env_name, صحيح default_val) {
    حرف *user_device_string = getenv(env_name);
    صحيح user_نمبر = default_val;

    unsigned n;
    إذا (user_device_string != عدم &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_نمبر = (صحيح)n;
    } وإلا {
        user_نمبر = default_val;
    }
    أرجع user_نمبر;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_عدد, int64_t block_size) {
  ثابت int64_t max_range = std::numeric_limits<صحيح>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_عدد * sycl_down_blk_size;
  طالما(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_عدد * sycl_down_blk_size;
  }
  أرجع sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساكن منطقي ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    أرجع g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
فراغ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (ggml_sycl_use_level_zero_device_alloc(q)) {
        فراغ *ptr = nullptr;
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
#وإلا
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        إذا (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            أرجع ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        أرجع nullptr;
    }
#endif
    فراغ * ptr = sycl::malloc_device(size, q);
    إذا (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        أرجع nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    أرجع ptr;
}

فراغ ggml_sycl_free_device(فراغ *ptr, sycl::queue &q) {
    إذا (!ptr) أرجع;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        أرجع;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, q)));
}

فراغ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    لكل (صحيح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        لكل (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            إذا (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        إذا (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
