#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ "zdnn.h"

#அந்தர்பாவ "scratch.hpp"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <future>
#அந்தர்பாவ <map>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <utility>
#அந்தர்பாவ <vector>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    செய் {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } போது (0);

கட்டமைப்பு ggml_backend_zdnn_device_context {
    முழு zdnn_device;
    முழு zdnn_device_ref_மொத்தம்;

    தருக்க has_parmblkformat_0;
    தருக்க has_parmblkformat_1;  // checks for z17

    size_t max_size;

    எழுத்து name[128];
};

கட்டமைப்பு ggml_backend_zdnn_buffer {
    வெறுமை * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<தசம> scales;          // per-row scales for quantized weights
    std::vector<தசம> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    எழுத்து name[GGML_MAX_NAME];
};

கட்டமைப்பு ggml_backend_zdnn_buffer_context {
    வெறுமை * all_data;
    size_t all_size;
    தருக்க owned;

    முழு n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

கட்டமைப்பு ggml_backend_zdnn_context {
    முழு device;
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
நிலையான உள்வரி வெறுமை ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    மாறிலி uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
நிலையான உள்வரி வெறுமை ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    மாறிலி uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
நிலையான உள்வரி zdnn_ztensor ggml_zdnn_அடுக்கு_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          மாறிலி zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    மாறிலி uint64_t அடுக்கு_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    தேர்வு (pre_tfm_desc->layout) {
        நிலை ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } நிறுத்து;
        நிலை ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } நிறுத்து;
        இயல்பு:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*அடுக்கு_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (எழுத்து *) src.buffer + s0*அடுக்கு_bytes;
    view.buffer_size          = n * அடுக்கு_bytes;
    திரும்பு view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
நிலையான உள்வரி மாறிலி zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    எனில் (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<தசம> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    திரும்பு bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
நிலையான உள்வரி வெறுமை ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, மாறிலி F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<வெறுமை>> tasks;
    க்கு (int32_t ith = 1; ith < n_threads; ith++) {
        மாறிலி int64_t start = (ith + 0) * n / n_threads;
        மாறிலி int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    க்கு (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
