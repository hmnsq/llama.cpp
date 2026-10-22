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
#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#جے_انت

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"

صحيح get_current_device_id() {
  أرجع dpct::dev_mgr::instance().current_device_id();
}

فراغ* ggml_sycl_host_malloc(size_t size) حاول {
  إذا (getenv("GGML_SYCL_NO_PINNED") != خالی_سوچک) {
    أرجع خالی_سوچک;
  }

  فراغ* ptr = خالی_سوچک;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (فراغ*)sycl::malloc_host(size, dpct::get_وچ_ترتیب_قطار()));

  إذا (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    أرجع خالی_سوچک;
  }

  أرجع ptr;
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_انت;
  std::باہر_نکلو(1);
}

فراغ ggml_sycl_host_free(فراغ* ptr) حاول {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, dpct::get_وچ_ترتیب_قطار())));
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_انت;
  std::باہر_نکلو(1);
}

منطقي gpu_has_xmx(sycl::device &dev) {
    أرجع dev.has(sycl::aspect::ext_intel_میٹرکس);
}

صحيح ggml_sycl_get_env(ثابت حرف *env_name, صحيح default_val) {
    حرف *user_device_string = getenv(env_name);
    صحيح user_نمبر = default_val;

    بنانشانی n;
    إذا (user_device_string != عدم &&
        لڑی_توں_پڑھو(user_device_string, " %u", &n) == 1) {
        user_نمبر = (صحيح)n;
    } وإلا {
        user_نمبر = default_val;
    }
    أرجع user_نمبر;
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلاک_عدد, int64_t بلاک_ناپ_3) {
  ثابت int64_t max_range = std::numeric_limits<صحيح>::max();
  int64_t sycl_تھلے_blk_ناپ = بلاک_ناپ_3;
  int64_t global_range = accumulate_بلاک_عدد * sycl_تھلے_blk_ناپ;
  طالما(global_range > max_range) {
      sycl_تھلے_blk_ناپ /= 2;
      global_range = accumulate_بلاک_عدد * sycl_تھلے_blk_ناپ;
  }
  أرجع sycl_تھلے_blk_ناپ;
}

#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساكن منطقي ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    أرجع g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#جے_انت

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
فراغ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (ggml_sycl_use_level_zero_device_alloc(q)) {
        فراغ *ptr = خالی_سوچک;
        خودکار ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        خودکار ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#جے_تعریف_شدہ ZE_RELAXED_ونڈ_LIMITS_EXP_NAME
        ze_relaxed_ونڈ_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ونڈ_LIMITS_EXP_DESC,
            خالی_سوچک,
            ZE_RELAXED_ونڈ_LIMITS_EXP_FLAG_MAX_ناپ,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#وإلا
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, خالی_سوچک, 0, 0};
#جے_انت
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_نتیجہ_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_بفر_ALIGNMENT, ze_dev, &ptr);
        إذا (r == ZE_نتیجہ_SUCCESS && ptr) {
            ggml_sycl_memtrace_جوڑو(type, ptr, size);
            أرجع ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        أرجع خالی_سوچک;
    }
#جے_انت
    فراغ * ptr = sycl::malloc_device(size, q);
    إذا (ptr == خالی_سوچک) {
        ggml_sycl_memtrace_fail(type, size);
        أرجع خالی_سوچک;
    }
    ggml_sycl_memtrace_جوڑو(type, ptr, size);
    أرجع ptr;
}

فراغ ggml_sycl_free_device(فراغ *ptr, sycl::queue &q) {
    إذا (!ptr) أرجع;
    ggml_sycl_memtrace_del(ptr);
#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (ggml_sycl_use_level_zero_device_alloc(q)) {
        خودکار ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        أرجع;
    }
#جے_انت
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, q)));
}

فراغ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<قطار_ptr> streams) {
    لكل (صحيح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        لكل (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            إذا (extra->events[i][is] != خالی_سوچک) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        إذا (extra->انکڑے_device[i] != خالی_سوچک && streams.size()>0) {
            ggml_sycl_سیٹ_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->انکڑے_device[i], *(streams[i]))));
        }
    }
    مٹاؤ extra;
}
