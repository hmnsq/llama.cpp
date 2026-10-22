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

#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#അന്തര്ഭാവ <level_zero/ze_api.h>
#endif

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"

പൂർണ്ണസംഖ്യ get_current_device_id() {
  തിരികെ dpct::dev_mgr::instance().current_device_id();
}

ശൂന്യം* ggml_sycl_host_malloc(size_t size) ശ്രമിക്കുക {
  എങ്കിൽ (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
    തിരികെ nullptr;
  }

  ശൂന്യം* ptr = nullptr;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (ശൂന്യം*)sycl::malloc_host(size, dpct::get_in_order_queue()));

  എങ്കിൽ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    തിരികെ nullptr;
  }

  തിരികെ ptr;
} പിടിക്കുക (sycl::exception സ്ഥിരം& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ശൂന്യം ggml_sycl_host_free(ശൂന്യം* ptr) ശ്രമിക്കുക {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::സ്വതന്ത്രം(ptr, dpct::get_in_order_queue())));
} പിടിക്കുക (sycl::exception സ്ഥിരം& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

തർക്കം gpu_has_xmx(sycl::device &dev) {
    തിരികെ dev.has(sycl::aspect::ext_intel_matrix);
}

പൂർണ്ണസംഖ്യ ggml_sycl_get_env(സ്ഥിരം അക്ഷരം *env_name, പൂർണ്ണസംഖ്യ default_val) {
    അക്ഷരം *user_device_string = getenv(env_name);
    പൂർണ്ണസംഖ്യ user_സംഖ്യ = default_val;

    unsigned n;
    എങ്കിൽ (user_device_string != ശൂന്യ &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_സംഖ്യ = (പൂർണ്ണസംഖ്യ)n;
    } അല്ലെങ്കിൽ {
        user_സംഖ്യ = default_val;
    }
    തിരികെ user_സംഖ്യ;
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_അക്കം, int64_t block_size) {
  സ്ഥിരം int64_t max_range = std::numeric_limits<പൂർണ്ണസംഖ്യ>::max();
  int64_t sycl_down_blk_size = block_size;
  int64_t global_range = accumulate_block_അക്കം * sycl_down_blk_size;
  ആകുമ്പോൾ(global_range > max_range) {
      sycl_down_blk_size /= 2;
      global_range = accumulate_block_അക്കം * sycl_down_blk_size;
  }
  തിരികെ sycl_down_blk_size;
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
സ്ഥിതം തർക്കം ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    തിരികെ g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#endif

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
ശൂന്യം * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    എങ്കിൽ (ggml_sycl_use_level_zero_device_alloc(q)) {
        ശൂന്യം *ptr = nullptr;
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
#അല്ലെങ്കിൽ
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
#endif
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_result_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_BUFFER_ALIGNMENT, ze_dev, &ptr);
        എങ്കിൽ (r == ZE_RESULT_SUCCESS && ptr) {
            ggml_sycl_memtrace_add(type, ptr, size);
            തിരികെ ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        തിരികെ nullptr;
    }
#endif
    ശൂന്യം * ptr = sycl::malloc_device(size, q);
    എങ്കിൽ (ptr == nullptr) {
        ggml_sycl_memtrace_fail(type, size);
        തിരികെ nullptr;
    }
    ggml_sycl_memtrace_add(type, ptr, size);
    തിരികെ ptr;
}

ശൂന്യം ggml_sycl_free_device(ശൂന്യം *ptr, sycl::queue &q) {
    എങ്കിൽ (!ptr) തിരികെ;
    ggml_sycl_memtrace_del(ptr);
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    എങ്കിൽ (ggml_sycl_use_level_zero_device_alloc(q)) {
        auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        തിരികെ;
    }
#endif
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::സ്വതന്ത്രം(ptr, q)));
}

ശൂന്യം release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        വേണ്ടി (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            എങ്കിൽ (extra->events[i][is] != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        എങ്കിൽ (extra->data_device[i] != nullptr && streams.size()>0) {
            ggml_sycl_set_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->data_device[i], *(streams[i]))));
        }
    }
    delete extra;
}
