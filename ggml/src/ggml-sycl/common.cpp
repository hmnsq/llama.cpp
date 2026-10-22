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

#شامل "common.hpp"
#شامل <sycl/backend.hpp>
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#شامل <level_zero/ze_api.h>
#اگر_ختم

#شامل "ggml-backend-impl.h"
#شامل "ggml-impl.h"

صحیح get_current_device_id() {
  واپس dpct::dev_mgr::instance().current_device_id();
}

خالی* ggml_sycl_host_malloc(size_t size) کوشش {
  اگر (getenv("GGML_SYCL_NO_PINNED") != خالی_اشاریہ) {
    واپس خالی_اشاریہ;
  }

  خالی* ptr = خالی_اشاریہ;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (خالی*)sycl::malloc_host(size, dpct::get_اندر_ترتیب_قطار()));

  اگر (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    واپس خالی_اشاریہ;
  }

  واپس ptr;
} پکڑو (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

خالی ggml_sycl_host_free(خالی* ptr) کوشش {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::آزاد(ptr, dpct::get_اندر_ترتیب_قطار())));
} پکڑو (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

منطقی gpu_has_xmx(sycl::device &dev) {
    واپس dev.has(sycl::aspect::ext_intel_میٹرکس);
}

صحیح ggml_sycl_get_env(ثابت حرف *env_name, صحیح default_val) {
    حرف *user_device_string = getenv(env_name);
    صحیح user_نمبر = default_val;

    غیرعلامتی n;
    اگر (user_device_string != تہی &&
        لڑی_سے_پڑھو(user_device_string, " %u", &n) == 1) {
        user_نمبر = (صحیح)n;
    } ورنہ {
        user_نمبر = default_val;
    }
    واپس user_نمبر;
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلاک_عدد, int64_t بلاک_سائز_3) {
  ثابت int64_t max_range = std::numeric_limits<صحیح>::max();
  int64_t sycl_نیچے_کو_blk_سائز = بلاک_سائز_3;
  int64_t global_range = accumulate_بلاک_عدد * sycl_نیچے_کو_blk_سائز;
  جب(global_range > max_range) {
      sycl_نیچے_کو_blk_سائز /= 2;
      global_range = accumulate_بلاک_عدد * sycl_نیچے_کو_blk_سائز;
  }
  واپس sycl_نیچے_کو_blk_سائز;
}

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساکن منطقی ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    واپس g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#اگر_ختم

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
خالی * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    اگر (ggml_sycl_use_level_zero_device_alloc(q)) {
        خالی *ptr = خالی_اشاریہ;
        خودکار ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        خودکار ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#اگر_تعریف_شدہ ZE_RELAXED_تخصیص_LIMITS_EXP_NAME
        ze_relaxed_تخصیص_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_تخصیص_LIMITS_EXP_DESC,
            خالی_اشاریہ,
            ZE_RELAXED_تخصیص_LIMITS_EXP_FLAG_MAX_سائز,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#ورنہ
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, خالی_اشاریہ, 0, 0};
#اگر_ختم
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_نتیجہ_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_بفر_ALIGNMENT, ze_dev, &ptr);
        اگر (r == ZE_نتیجہ_SUCCESS && ptr) {
            ggml_sycl_memtrace_جوڑو(type, ptr, size);
            واپس ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        واپس خالی_اشاریہ;
    }
#اگر_ختم
    خالی * ptr = sycl::malloc_device(size, q);
    اگر (ptr == خالی_اشاریہ) {
        ggml_sycl_memtrace_fail(type, size);
        واپس خالی_اشاریہ;
    }
    ggml_sycl_memtrace_جوڑو(type, ptr, size);
    واپس ptr;
}

خالی ggml_sycl_free_device(خالی *ptr, sycl::queue &q) {
    اگر (!ptr) واپس;
    ggml_sycl_memtrace_del(ptr);
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    اگر (ggml_sycl_use_level_zero_device_alloc(q)) {
        خودکار ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        واپس;
    }
#اگر_ختم
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::آزاد(ptr, q)));
}

خالی release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<قطار_ptr> streams) {
    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        برائے (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            اگر (extra->events[i][is] != خالی_اشاریہ) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        اگر (extra->ڈیٹا_device[i] != خالی_اشاریہ && streams.size()>0) {
            ggml_sycl_سیٹ_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->ڈیٹا_device[i], *(streams[i]))));
        }
    }
    حذف extra;
}
