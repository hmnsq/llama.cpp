#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_ZDNN_COMMON_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ZDNN_COMMON_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "zdnn.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "scratch.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <future>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ZDNN_NAME    "zDNN"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ZDNN_ਸੰਸਕਰਣ ZDNN_VERNUM

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ZDNN_CHECK(stmt)                \
    ਕਰੋ {                                \
        zdnn_ਸਥਿਤੀ ਸਥਿਤੀ = (stmt);    \
        GGML_ASSERT(ਸਥਿਤੀ == ZDNN_OK); \
    } ਜਦੋਂ (0);

ਬਣਤਰ ggml_backend_zdnn_device_context {
    ਪੂਰਨ zdnn_device;
    ਪੂਰਨ zdnn_device_ref_ਗਿਣਤੀ;

    ਤਾਰਕਿਕ has_parmblkformat_0;
    ਤਾਰਕਿਕ has_parmblkformat_1;  // checks for z17

    size_t max_size;

    ਅੱਖਰ name[128];
};

ਬਣਤਰ ggml_backend_zdnn_ਬਫਰ {
    ਖਾਲੀ * ਅੰਕੜੇ;
    ggml_backend_zdnn_ਬਫਰ * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<ਦਸ਼ਮਲਵ> scales;          // per-row scales for quantized weights
    std::vector<ਦਸ਼ਮਲਵ> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    ਅੱਖਰ name[GGML_MAX_NAME];
};

ਬਣਤਰ ggml_backend_zdnn_ਬਫਰ_context {
    ਖਾਲੀ * all_ਅੰਕੜੇ;
    size_t all_ਮਾਪ;
    ਤਾਰਕਿਕ owned;

    ਪੂਰਨ n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_ਬਫਰ>> buffers;
};

ਬਣਤਰ ggml_backend_zdnn_context {
    ਪੂਰਨ device;
    int32_t n_ਧਾਗੇ;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_ਬਫਰ>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ਅਚਲ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->ਬਫਰ      = scratch.alloc(ctx->pool, size);
    ztensor->ਬਫਰ_ਮਾਪ = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ਅਚਲ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->ਬਫਰ      = scratch.alloc(ctx->pool, size);
    ztensor->ਬਫਰ_ਮਾਪ = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
ਸਥਿਰ ਸਤਰਅੰਦਰ zdnn_ztensor ggml_zdnn_ਢੇਰ_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ਅਚਲ zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.ਬਫਰ_ਮਾਪ % n_stacks == 0);
    ਅਚਲ uint64_t ਢੇਰ_ਬਾਈਟਾਂ = src.ਬਫਰ_ਮਾਪ / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    ਚੋਣ (pre_tfm_desc->layout) {
        ਹਾਲਤ ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } ਰੋਕੋ;
        ਹਾਲਤ ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*ਢੇਰ_ਬਾਈਟਾਂ);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.ਬਫਰ               = (ਅੱਖਰ *) src.ਬਫਰ + s0*ਢੇਰ_ਬਾਈਟਾਂ;
    view.ਬਫਰ_ਮਾਪ          = n * ਢੇਰ_ਬਾਈਟਾਂ;
    ਵਾਪਸ view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_ਬਫਰ> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    ਜੇ (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_ਬਫਰ>();

        std::vector<ਦਸ਼ਮਲਵ> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.ਅੰਕੜੇ()));
    }

    ਵਾਪਸ bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_zdnn_parallel_for(int32_t n_ਧਾਗੇ, int64_t n, ਅਚਲ F & f) {
    n_ਧਾਗੇ = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_ਧਾਗੇ, n));

    std::vector<std::future<ਖਾਲੀ>> tasks;
    ਲਈ (int32_t ith = 1; ith < n_ਧਾਗੇ; ith++) {
        ਅਚਲ int64_t start = (ith + 0) * n / n_ਧਾਗੇ;
        ਅਚਲ int64_t end   = (ith + 1) * n / n_ਧਾਗੇ;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_ਧਾਗੇ);

    ਲਈ (ਸਵੈਚਾਲਿਤ & task : tasks) {
        task.get();
    }
}

#ਜੇ_ਅੰਤ  // GGML_ZDNN_COMMON_HPP
