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
    করো {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } যখন (0);

কাঠামো ggml_backend_zdnn_device_context {
    পূর্ণসংখ্যা zdnn_device;
    পূর্ণসংখ্যা zdnn_device_ref_গুনতি;

    যৌক্তিক has_parmblkformat_0;
    যৌক্তিক has_parmblkformat_1;  // checks for z17

    size_t max_size;

    অক্ষর name[128];
};

কাঠামো ggml_backend_zdnn_বাফার {
    শূন্য * তথ্য;
    ggml_backend_zdnn_বাফার * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<দশমিক> scales;          // per-row scales for quantized weights
    std::vector<দশমিক> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    অক্ষর name[GGML_MAX_NAME];
};

কাঠামো ggml_backend_zdnn_বাফার_context {
    শূন্য * all_তথ্য;
    size_t all_আকার;
    যৌক্তিক owned;

    পূর্ণসংখ্যা n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_বাফার>> buffers;
};

কাঠামো ggml_backend_zdnn_context {
    পূর্ণসংখ্যা device;
    int32_t n_threads;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_বাফার>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
স্থির অন্তঃরেখ শূন্য ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ধ্রুবক uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->বাফার      = scratch.alloc(ctx->pool, size);
    ztensor->বাফার_আকার = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
স্থির অন্তঃরেখ শূন্য ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ধ্রুবক uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->বাফার      = scratch.alloc(ctx->pool, size);
    ztensor->বাফার_আকার = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
স্থির অন্তঃরেখ zdnn_ztensor ggml_zdnn_স্তূপ_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ধ্রুবক zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.বাফার_আকার % n_stacks == 0);
    ধ্রুবক uint64_t স্তূপ_bytes = src.বাফার_আকার / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    নির্বাচন (pre_tfm_desc->layout) {
        ক্ষেত্র ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } বিরতি;
        ক্ষেত্র ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } বিরতি;
        পূর্বনির্ধারিত:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*স্তূপ_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.বাফার               = (অক্ষর *) src.বাফার + s0*স্তূপ_bytes;
    view.বাফার_আকার          = n * স্তূপ_bytes;
    ফেরত view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
স্থির অন্তঃরেখ ধ্রুবক zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_বাফার> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    যদি (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_বাফার>();

        std::vector<দশমিক> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.তথ্য()));
    }

    ফেরত bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
স্থির অন্তঃরেখ শূন্য ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, ধ্রুবক F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<শূন্য>> tasks;
    জন্য (int32_t ith = 1; ith < n_threads; ith++) {
        ধ্রুবক int64_t start = (ith + 0) * n / n_threads;
        ধ্রুবক int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    জন্য (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
