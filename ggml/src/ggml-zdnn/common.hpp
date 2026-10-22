#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ "zdnn.h"

#ಅನ್ತರ್ಭಾವ "scratch.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <future>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <vector>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    ಕುರು {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } ಯಾವತ್ (0);

ಸಂರಚನಾ ggml_backend_zdnn_device_context {
    ಪೂರ್ಣಾಙ್ಕ zdnn_device;
    ಪೂರ್ಣಾಙ್ಕ zdnn_device_ref_ಎಣಿಕೆ;

    ತಾರ್ಕಿಕ has_parmblkformat_0;
    ತಾರ್ಕಿಕ has_parmblkformat_1;  // checks for z17

    size_t max_size;

    ವರ್ಣ name[128];
};

ಸಂರಚನಾ ggml_backend_zdnn_buffer {
    ಶೂನ್ಯ * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<ಭಿನ್ನ> scales;          // per-row scales for quantized weights
    std::vector<ಭಿನ್ನ> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    ವರ್ಣ name[GGML_MAX_NAME];
};

ಸಂರಚನಾ ggml_backend_zdnn_buffer_context {
    ಶೂನ್ಯ * all_data;
    size_t all_size;
    ತಾರ್ಕಿಕ owned;

    ಪೂರ್ಣಾಙ್ಕ n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

ಸಂರಚನಾ ggml_backend_zdnn_context {
    ಪೂರ್ಣಾಙ್ಕ device;
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
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ಧ್ರುವ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ಧ್ರುವ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ zdnn_ztensor ggml_zdnn_ರಾಶಿ_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ಧ್ರುವ zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    ಧ್ರುವ uint64_t ರಾಶಿ_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    ವಿಕಲ್ಪ (pre_tfm_desc->layout) {
        ಪಕ್ಷ ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } ಭಙ್ಗ;
        ಪಕ್ಷ ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*ರಾಶಿ_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (ವರ್ಣ *) src.buffer + s0*ರಾಶಿ_bytes;
    view.buffer_size          = n * ರಾಶಿ_bytes;
    ಪ್ರತ್ಯಾಗಮ view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    ಯದಿ (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<ಭಿನ್ನ> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    ಪ್ರತ್ಯಾಗಮ bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, ಧ್ರುವ F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<ಶೂನ್ಯ>> tasks;
    ಪ್ರತಿ (int32_t ith = 1; ith < n_threads; ith++) {
        ಧ್ರುವ int64_t start = (ith + 0) * n / n_threads;
        ಧ್ರುವ int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    ಪ್ರತಿ (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
