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

#అన్తర్భావ "common.hpp"
#అన్తర్భావ <sycl/backend.hpp>
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#అన్తర్భావ <level_zero/ze_api.h>
#అయితే_ముగింపు

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-impl.h"

పూర్ణాంకం get_current_device_id() {
  తిరిగి dpct::dev_mgr::instance().current_device_id();
}

శూన్య* ggml_sycl_host_malloc(size_t size) ప్రయత్నించు {
  ఒకవేళ (getenv("GGML_SYCL_NO_PINNED") != శూన్యసూచిక) {
    తిరిగి శూన్యసూచిక;
  }

  శూన్య* ptr = శూన్యసూచిక;
  // allow to use dpct::get_in_order_queue() for host malloc
  dpct::err0 err = CHECK_TRY_ERROR(
      ptr = (శూన్య*)sycl::malloc_host(size, dpct::get_లోపల_క్రమం_వరుస()));

  ఒకవేళ (err != 0) {
    // clear the error
    GGML_LOG_ERROR("WARNING: failed to allocate %.2f MB of pinned memory: %s\n", size / 1024.0 / 1024.0,    "syclGetErrorString is not supported");
    తిరిగి శూన్యసూచిక;
  }

  తిరిగి ptr;
} పట్టుకో (sycl::exception స్థిరరాశి& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

శూన్య ggml_sycl_host_free(శూన్య* ptr) ప్రయత్నించు {
  // allow to use dpct::get_in_order_queue() for host malloc
  SYCL_CHECK(CHECK_TRY_ERROR(sycl::విడుదల(ptr, dpct::get_లోపల_క్రమం_వరుస())));
} పట్టుకో (sycl::exception స్థిరరాశి& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

తార్కిక gpu_has_xmx(sycl::device &dev) {
    తిరిగి dev.has(sycl::aspect::ext_intel_మాతృక);
}

పూర్ణాంకం ggml_sycl_get_env(స్థిరరాశి అక్షరం *env_name, పూర్ణాంకం default_val) {
    అక్షరం *user_device_string = getenv(env_name);
    పూర్ణాంకం user_సంఖ్య = default_val;

    గుర్తులేని n;
    ఒకవేళ (user_device_string != ఖాళీ &&
        అక్షరశ్రేణి_చదువు(user_device_string, " %u", &n) == 1) {
        user_సంఖ్య = (పూర్ణాంకం)n;
    } లేకపోతే {
        user_సంఖ్య = default_val;
    }
    తిరిగి user_సంఖ్య;
}

int64_t downsample_sycl_global_range(int64_t accumulate_ఖండం_అంకె, int64_t ఖండం_కొలత_3) {
  స్థిరరాశి int64_t max_range = std::numeric_limits<పూర్ణాంకం>::max();
  int64_t sycl_కిందికి_blk_కొలత = ఖండం_కొలత_3;
  int64_t global_range = accumulate_ఖండం_అంకె * sycl_కిందికి_blk_కొలత;
  అయితే(global_range > max_range) {
      sycl_కిందికి_blk_కొలత /= 2;
      global_range = accumulate_ఖండం_అంకె * sycl_కిందికి_blk_కొలత;
  }
  తిరిగి sycl_కిందికి_blk_కొలత;
}

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
స్థిర తార్కిక ggml_sycl_use_level_zero_device_alloc(sycl::queue &q) {
    తిరిగి g_ggml_sycl_use_level_zero_api &&
        q.get_device().is_gpu() &&
        q.get_backend() == sycl::backend::ext_oneapi_level_zero;
}
#అయితే_ముగింపు

// Use Level Zero zeMemAllocDevice to avoid sycl::malloc_device triggering
// DMA-buf/TTM system RAM staging in the xe kernel driver during multi-GPU inference.
శూన్య * ggml_sycl_malloc_device(size_t size, sycl::queue &q, ggml_sycl_mem_type type) {
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ఒకవేళ (ggml_sycl_use_level_zero_device_alloc(q)) {
        శూన్య *ptr = శూన్యసూచిక;
        స్వయంచాలక ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        స్వయంచాలక ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
#నిర్వచితమైతే ZE_RELAXED_కేటాయింపు_LIMITS_EXP_NAME
        ze_relaxed_కేటాయింపు_limits_exp_desc_t relaxed_desc = {
            ZE_STRUCTURE_TYPE_RELAXED_కేటాయింపు_LIMITS_EXP_DESC,
            శూన్యసూచిక,
            ZE_RELAXED_కేటాయింపు_LIMITS_EXP_FLAG_MAX_కొలత,
        };
        ze_device_mem_alloc_desc_t alloc_desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            &relaxed_desc,
            0,
            0,
        };
#లేకపోతే
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, శూన్యసూచిక, 0, 0};
#అయితే_ముగింపు
        // request SYCL_BUFFER_ALIGNMENT so tensor offsets aligned to it stay aligned in absolute terms
        ze_ఫలితం_t r = zeMemAllocDevice(ze_ctx, &alloc_desc, size, SYCL_బఫరు_ALIGNMENT, ze_dev, &ptr);
        ఒకవేళ (r == ZE_ఫలితం_SUCCESS && ptr) {
            ggml_sycl_memtrace_కూడు(type, ptr, size);
            తిరిగి ptr;
        }
        ggml_sycl_memtrace_fail(type, size);
        తిరిగి శూన్యసూచిక;
    }
#అయితే_ముగింపు
    శూన్య * ptr = sycl::malloc_device(size, q);
    ఒకవేళ (ptr == శూన్యసూచిక) {
        ggml_sycl_memtrace_fail(type, size);
        తిరిగి శూన్యసూచిక;
    }
    ggml_sycl_memtrace_కూడు(type, ptr, size);
    తిరిగి ptr;
}

శూన్య ggml_sycl_free_device(శూన్య *ptr, sycl::queue &q) {
    ఒకవేళ (!ptr) తిరిగి;
    ggml_sycl_memtrace_del(ptr);
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ఒకవేళ (ggml_sycl_use_level_zero_device_alloc(q)) {
        స్వయంచాలక ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        zeMemFree(ze_ctx, ptr);
        తిరిగి;
    }
#అయితే_ముగింపు
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::విడుదల(ptr, q)));
}

శూన్య release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<వరుస_ptr> streams) {
    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        కొరకు (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            ఒకవేళ (extra->events[i][is] != శూన్యసూచిక) {
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::destroy_event(extra->events[i][is])));
            }
        }
        ఒకవేళ (extra->దత్తాంశం_device[i] != శూన్యసూచిక && streams.size()>0) {
            ggml_sycl_సమితి_device(i);
            SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(extra->దత్తాంశం_device[i], *(streams[i]))));
        }
    }
    తొలగించు extra;
}
