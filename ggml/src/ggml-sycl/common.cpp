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
#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#د_که_پای

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"

صحيح get_current_device_id() {
  أرجع dpct::dev_mgr::instance().current_device_id();
}

فراغ* ggml_sycl_host_malloc(size_t size) حاول {
  إذا (getenv("GGML_SYCL_NO_PINNED") != تش_اشاره) {
    أرجع تش_اشاره;
  }

  فراغ* ptr = تش_اشاره;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (فراغ*)sycl::malloc_host(size, dpct::get_دننه_ترتیب_کتار()));

  إذا (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    أرجع تش_اشاره;
  }

  أرجع ptr;
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

فراغ ggml_sycl_host_free(فراغ* ptr) حاول {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, dpct::get_دننه_ترتیب_کتار())));
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  std::وتل(1);
}

منطقي gpu_has_xmx(sycl::device &dev) {
    أرجع dev.has(sycl::aspect::ext_intel_ماتریکس);
}

صحيح ggml_sycl_get_env(ثابت حرف *env_name, صحيح default_val) {
    حرف *user_device_string = getenv(env_name);
    صحيح user_شمېره = default_val;

    بینښه n;
    إذا (user_device_string != عدم &&
        له_مزي_ولوله(user_device_string, " %u", &n) == 1) {
        user_شمېره = (صحيح)n;
    } وإلا {
        user_شمېره = default_val;
    }
    أرجع user_شمېره;
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلاک_عدد, int64_t بلاک_اندازه_3) {
  ثابت int64_t max_range = std::numeric_limits<صحيح>::max();
  int64_t sycl_ښکته_blk_اندازه = بلاک_اندازه_3;
  int64_t global_range = accumulate_بلاک_عدد * sycl_ښکته_blk_اندازه;
  طالما(global_range > max_range) {
      sycl_ښکته_blk_اندازه /= 2;
      global_range = accumulate_بلاک_عدد * sycl_ښکته_blk_اندازه;
  }
  أرجع sycl_ښکته_blk_اندازه;
}

#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساكن منطقي ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    أرجع g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#د_که_پای

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
فراغ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (ggml_sycl_use_level_zero_device_alloc(q)) {
        فراغ *ptr = تش_اشاره;
        خپلکاری ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        خپلکاری ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#که_تعریف_شوی ZE_RELAXED_ویش_LIMITS_EXP_NAME
        ze_relaxed_ویش_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ویش_LIMITS_EXP_DESC,
            تش_اشاره,
            ZE_RELAXED_ویش_LIMITS_EXP_FLAG_MAX_اندازه,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#وإلا
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, تش_اشاره, 0, 0};
#د_که_پای
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_پایله_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_بفر_ALIGNMENT, ze_dev, &ptr);
        إذا (r == ZE_پایله_SUCCESS && ptr) {
            ggml_sycl_memtrace_جمع_کړه(type, ptr, size);
            أرجع ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        أرجع تش_اشاره;
    }
#د_که_پای
    فراغ * ptr = sycl::malloc_device(size, q);
    إذا (ptr == تش_اشاره) {
        ggml_sycl_memtrace_fail(type, size);
        أرجع تش_اشاره;
    }
    ggml_sycl_memtrace_جمع_کړه(type, ptr, size);
    أرجع ptr;
}

فراغ ggml_sycl_free_device(فراغ *ptr, sycl::queue &q) {
    إذا (!ptr) أرجع;
    ggml_sycl_memtrace_del(ptr);
#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (ggml_sycl_use_level_zero_device_alloc(q)) {
        خپلکاری ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        أرجع;
    }
#د_که_پای
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, q)));
}

فراغ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<کتار_ptr> streams) {
    لكل (صحيح i = 0; i < ggml_sycl_info().device_شمېر; ++i) {
        لكل (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            إذا (extra->events[i][is] != تش_اشاره) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        إذا (extra->معلومات_device[i] != تش_اشاره && streams.size()>0) {
            ggml_sycl_سېټ_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->معلومات_device[i], *(streams[i]))));
        }
    }
    ړنګول extra;
}
