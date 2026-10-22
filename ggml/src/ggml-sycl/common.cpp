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

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/backend.hpp>
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ਅਨ੍ਤਰ੍ਭਾਵ <level_zero/ze_api.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

ਪੂਰਨ get_current_device_id() {
  ਵਾਪਸ dpct::dev_mgr::instance().current_device_id();
}

ਖਾਲੀ* ggml_sycl_host_malloc(size_t size) ਕੋਸ਼ਿਸ਼ {
  ਜੇ (getenv("GGML_SYCL_NO_PINNED") != ਖਾਲੀ_ਸੂਚਕ) {
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
  }

  ਖਾਲੀ* ptr = ਖਾਲੀ_ਸੂਚਕ;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (ਖਾਲੀ*)sycl::malloc_host(size, dpct::get_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ()));

  ਜੇ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
  }

  ਵਾਪਸ ptr;
} ਫੜੋ (sycl::exception ਅਚਲ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਖਾਲੀ ggml_sycl_host_free(ਖਾਲੀ* ptr) ਕੋਸ਼ਿਸ਼ {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::ਮੁਕਤ(ptr, dpct::get_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ())));
} ਫੜੋ (sycl::exception ਅਚਲ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਤਾਰਕਿਕ gpu_has_xmx(sycl::device &dev) {
    ਵਾਪਸ dev.has(sycl::aspect::ext_intel_ਮੈਟ੍ਰਿਕਸ);
}

ਪੂਰਨ ggml_sycl_get_env(ਅਚਲ ਅੱਖਰ *env_name, ਪੂਰਨ default_val) {
    ਅੱਖਰ *user_device_string = getenv(env_name);
    ਪੂਰਨ user_ਸੰਖਿਆ = default_val;

    ਬਿਨਾਨਿਸ਼ਾਨੀ n;
    ਜੇ (user_device_string != ਨਿਰਾ &&
        ਲੜੀ_ਤੋਂ_ਪੜ੍ਹੋ(user_device_string, " %u", &n) == 1) {
        user_ਸੰਖਿਆ = (ਪੂਰਨ)n;
    } ਨਹੀਂਤਾਂ {
        user_ਸੰਖਿਆ = default_val;
    }
    ਵਾਪਸ user_ਸੰਖਿਆ;
}

int64_t downsample_sycl_global_range(int64_t accumulate_ਖੰਡ_ਅੰਕ, int64_t ਖੰਡ_ਮਾਪ_3) {
  ਅਚਲ int64_t max_range = std::numeric_limits<ਪੂਰਨ>::max();
  int64_t sycl_ਥੱਲੇ_blk_ਮਾਪ = ਖੰਡ_ਮਾਪ_3;
  int64_t global_range = accumulate_ਖੰਡ_ਅੰਕ * sycl_ਥੱਲੇ_blk_ਮਾਪ;
  ਜਦੋਂ(global_range > max_range) {
      sycl_ਥੱਲੇ_blk_ਮਾਪ /= 2;
      global_range = accumulate_ਖੰਡ_ਅੰਕ * sycl_ਥੱਲੇ_blk_ਮਾਪ;
  }
  ਵਾਪਸ sycl_ਥੱਲੇ_blk_ਮਾਪ;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    ਵਾਪਸ g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#ਜੇ_ਅੰਤ

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
ਖਾਲੀ * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ਜੇ (ggml_sycl_use_level_zero_device_alloc(q)) {
        ਖਾਲੀ *ptr = ਖਾਲੀ_ਸੂਚਕ;
        ਸਵੈਚਾਲਿਤ ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ਸਵੈਚਾਲਿਤ ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ZE_RELAXED_ਵੰਡ_LIMITS_EXP_NAME
        ze_relaxed_ਵੰਡ_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_ਵੰਡ_LIMITS_EXP_DESC,
            ਖਾਲੀ_ਸੂਚਕ,
            ZE_RELAXED_ਵੰਡ_LIMITS_EXP_FLAG_MAX_ਮਾਪ,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#ਨਹੀਂਤਾਂ
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, ਖਾਲੀ_ਸੂਚਕ, 0, 0};
#ਜੇ_ਅੰਤ
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_ਨਤੀਜਾ_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_ਬਫਰ_ALIGNMENT, ze_dev, &ptr);
        ਜੇ (r == ZE_ਨਤੀਜਾ_SUCCESS && ptr) {
            ggml_sycl_memtrace_ਜੋੜੋ(type, ptr, size);
            ਵਾਪਸ ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
#ਜੇ_ਅੰਤ
    ਖਾਲੀ * ptr = sycl::malloc_device(size, q);
    ਜੇ (ptr == ਖਾਲੀ_ਸੂਚਕ) {
        ggml_sycl_memtrace_fail(type, size);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ggml_sycl_memtrace_ਜੋੜੋ(type, ptr, size);
    ਵਾਪਸ ptr;
}

ਖਾਲੀ ggml_sycl_free_device(ਖਾਲੀ *ptr, sycl::queue &q) {
    ਜੇ (!ptr) ਵਾਪਸ;
    ggml_sycl_memtrace_del(ptr);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ਜੇ (ggml_sycl_use_level_zero_device_alloc(q)) {
        ਸਵੈਚਾਲਿਤ ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        ਵਾਪਸ;
    }
#ਜੇ_ਅੰਤ
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::ਮੁਕਤ(ptr, q)));
}

ਖਾਲੀ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<ਕਤਾਰ_ptr> streams) {
    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        ਲਈ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            ਜੇ (extra->events[i][is] != ਖਾਲੀ_ਸੂਚਕ) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        ਜੇ (extra->ਅੰਕੜੇ_device[i] != ਖਾਲੀ_ਸੂਚਕ && streams.size()>0) {
            ggml_sycl_ਸੈੱਟ_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->ਅੰਕੜੇ_device[i], *(streams[i]))));
        }
    }
    ਮਿਟਾਓ extra;
}
