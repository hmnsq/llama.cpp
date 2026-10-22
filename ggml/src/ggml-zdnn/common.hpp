#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব "zdnn.h"

#অন্তর্ভাব "scratch.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <future>
#অন্তর্ভাব <map>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <utility>
#অন্তর্ভাব <vector>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    কুরু {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } যাবত্ (0);

সংরচনা ggml_backend_zdnn_device_context {
    পূর্ণাঙ্ক zdnn_device;
    পূর্ণাঙ্ক zdnn_device_ref_গুনতি;

    তার্কিক has_parmblkformat_0;
    তার্কিক has_parmblkformat_1;  // checks for z17

    size_t max_size;

    বর্ণ name[128];
};

সংরচনা ggml_backend_zdnn_buffer {
    শূন্য * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<ভিন্ন> scales;          // per-row scales for quantized weights
    std::vector<ভিন্ন> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    বর্ণ name[GGML_MAX_NAME];
};

সংরচনা ggml_backend_zdnn_buffer_context {
    শূন্য * all_data;
    size_t all_size;
    তার্কিক owned;

    পূর্ণাঙ্ক n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

সংরচনা ggml_backend_zdnn_context {
    পূর্ণাঙ্ক device;
    int32_t n_threads;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_buffer>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
স্থির অন্তঃস্থ শূন্য ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ধ্রুব uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
স্থির অন্তঃস্থ শূন্য ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ধ্রুব uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
স্থির অন্তঃস্থ zdnn_ztensor ggml_zdnn_স্তূপ_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ধ্রুব zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    ধ্রুব uint64_t স্তূপ_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    বিকল্প (pre_tfm_desc->layout) {
        পক্ষ ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } ভঙ্গ;
        পক্ষ ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } ভঙ্গ;
        স্বভাব:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*স্তূপ_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (বর্ণ *) src.buffer + s0*স্তূপ_bytes;
    view.buffer_size          = n * স্তূপ_bytes;
    প্রত্যাগম view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
স্থির অন্তঃস্থ ধ্রুব zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    যদি (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<ভিন্ন> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    প্রত্যাগম bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
স্থির অন্তঃস্থ শূন্য ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, ধ্রুব F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<শূন্য>> tasks;
    প্রতি (int32_t ith = 1; ith < n_threads; ith++) {
        ধ্রুব int64_t start = (ith + 0) * n / n_threads;
        ধ্রুব int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    প্রতি (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
