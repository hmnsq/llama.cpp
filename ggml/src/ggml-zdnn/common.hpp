#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_ZDNN_COMMON_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ZDNN_COMMON_HPP

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

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ZDNN_NAME    "zDNN"
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ZDNN_ಆವೃತ್ತಿ ZDNN_VERNUM

#ವ್ಯಾಖ್ಯಾನಿಸು ZDNN_CHECK(stmt)                \
    ಮಾಡು {                                \
        zdnn_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = (stmt);    \
        GGML_ASSERT(ಸ್ಥಿತಿ_2 == ZDNN_OK); \
    } ಆಗ (0);

ರಚನೆ ggml_backend_zdnn_device_context {
    ಪೂರ್ಣಾಂಕ zdnn_device;
    ಪೂರ್ಣಾಂಕ zdnn_device_ref_ಎಣಿಕೆ;

    ತಾರ್ಕಿಕ has_parmblkformat_0;
    ತಾರ್ಕಿಕ has_parmblkformat_1;  // checks for z17

    size_t max_size;

    ಅಕ್ಷರ name[128];
};

ರಚನೆ ggml_backend_zdnn_ಬಫರ್ {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ;
    ggml_backend_zdnn_ಬಫರ್ * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<ದಶಮಾಂಶ> scales;          // per-row scales for quantized weights
    std::vector<ದಶಮಾಂಶ> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    ಅಕ್ಷರ name[GGML_MAX_NAME];
};

ರಚನೆ ggml_backend_zdnn_ಬಫರ್_context {
    ಶೂನ್ಯ * all_ದತ್ತಾಂಶ;
    size_t all_ಗಾತ್ರ;
    ತಾರ್ಕಿಕ owned;

    ಪೂರ್ಣಾಂಕ n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_ಬಫರ್>> buffers;
};

ರಚನೆ ggml_backend_zdnn_context {
    ಪೂರ್ಣಾಂಕ device;
    int32_t n_ಎಳೆಗಳು;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_ಬಫರ್>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ಸ್ಥಿರಾಂಕ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->ಬಫರ್      = scratch.alloc(ctx->pool, size);
    ztensor->ಬಫರ್_ಗಾತ್ರ = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ಸ್ಥಿರಾಂಕ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->ಬಫರ್      = scratch.alloc(ctx->pool, size);
    ztensor->ಬಫರ್_ಗಾತ್ರ = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
ಸ್ಥಿರ ಸಾಲೊಳಗೆ zdnn_ztensor ggml_zdnn_ರಾಶಿ_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ಸ್ಥಿರಾಂಕ zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.ಬಫರ್_ಗಾತ್ರ % n_stacks == 0);
    ಸ್ಥಿರಾಂಕ uint64_t ರಾಶಿ_ಬೈಟುಗಳು = src.ಬಫರ್_ಗಾತ್ರ / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    ಆಯ್ಕೆ (pre_tfm_desc->layout) {
        ಸ್ಥಿತಿ ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*ರಾಶಿ_ಬೈಟುಗಳು);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.ಬಫರ್               = (ಅಕ್ಷರ *) src.ಬಫರ್ + s0*ರಾಶಿ_ಬೈಟುಗಳು;
    view.ಬಫರ್_ಗಾತ್ರ          = n * ರಾಶಿ_ಬೈಟುಗಳು;
    ಹಿಂತಿರುಗಿ view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_ಬಫರ್> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    ಆದರೆ (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_ಬಫರ್>();

        std::vector<ದಶಮಾಂಶ> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.ದತ್ತಾಂಶ()));
    }

    ಹಿಂತಿರುಗಿ bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_zdnn_parallel_for(int32_t n_ಎಳೆಗಳು, int64_t n, ಸ್ಥಿರಾಂಕ F & f) {
    n_ಎಳೆಗಳು = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_ಎಳೆಗಳು, n));

    std::vector<std::future<ಶೂನ್ಯ>> tasks;
    ಗಾಗಿ (int32_t ith = 1; ith < n_ಎಳೆಗಳು; ith++) {
        ಸ್ಥಿರಾಂಕ int64_t start = (ith + 0) * n / n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t end   = (ith + 1) * n / n_ಎಳೆಗಳು;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_ಎಳೆಗಳು);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & task : tasks) {
        task.get();
    }
}

#ಆದರೆ_ಅಂತ್ಯ  // GGML_ZDNN_COMMON_HPP
