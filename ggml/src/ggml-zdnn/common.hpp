#यदि_अपरिभाषितम् GGML_ZDNN_COMMON_HPP
#परिभाषय GGML_ZDNN_COMMON_HPP

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव "zdnn.h"

#अन्तर्भाव "scratch.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <future>
#अन्तर्भाव <map>
#अन्तर्भाव <memory>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

#परिभाषय GGML_ZDNN_NAME    "zDNN"
#परिभाषय GGML_ZDNN_संस्करणम् ZDNN_VERNUM

#परिभाषय ZDNN_CHECK(stmt)                \
    कुरु {                                \
        zdnn_स्थितिः स्थितिः = (stmt);    \
        GGML_ASSERT(स्थितिः == ZDNN_OK); \
    } यावत् (0);

संरचना ggml_backend_zdnn_device_context {
    पूर्णाङ्क zdnn_device;
    पूर्णाङ्क zdnn_device_ref_गणनम्;

    तार्किक has_parmblkformat_0;
    तार्किक has_parmblkformat_1;  // checks for z17

    size_t max_size;

    वर्ण name[128];
};

संरचना ggml_backend_zdnn_बफर {
    शून्य * दत्त;
    ggml_backend_zdnn_बफर * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<भिन्न> scales;          // per-row scales for quantized weights
    std::vector<भिन्न> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    वर्ण name[GGML_MAX_NAME];
};

संरचना ggml_backend_zdnn_बफर_context {
    शून्य * all_दत्त;
    size_t all_आकार;
    तार्किक owned;

    पूर्णाङ्क n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_बफर>> buffers;
};

संरचना ggml_backend_zdnn_context {
    पूर्णाङ्क device;
    int32_t n_तन्तवः;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_बफर>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
स्थिर अन्तःस्थ शून्य ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ध्रुव uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->बफर      = scratch.alloc(ctx->pool, size);
    ztensor->बफर_आकार = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
स्थिर अन्तःस्थ शून्य ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ध्रुव uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->बफर      = scratch.alloc(ctx->pool, size);
    ztensor->बफर_आकार = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
स्थिर अन्तःस्थ zdnn_ztensor ggml_zdnn_राशिः_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ध्रुव zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.बफर_आकार % n_stacks == 0);
    ध्रुव uint64_t राशिः_अष्टकानि = src.बफर_आकार / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    विकल्प (pre_tfm_desc->layout) {
        पक्ष ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } भङ्ग;
        पक्ष ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } भङ्ग;
        स्वभाव:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*राशिः_अष्टकानि);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.बफर               = (वर्ण *) src.बफर + s0*राशिः_अष्टकानि;
    view.बफर_आकार          = n * राशिः_अष्टकानि;
    प्रत्यागम view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
स्थिर अन्तःस्थ ध्रुव zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_बफर> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    यदि (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_बफर>();

        std::vector<भिन्न> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.दत्त()));
    }

    प्रत्यागम bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
प्रतिरूपम् <प्रकारनाम F>
स्थिर अन्तःस्थ शून्य ggml_zdnn_parallel_for(int32_t n_तन्तवः, int64_t n, ध्रुव F & f) {
    n_तन्तवः = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_तन्तवः, n));

    std::vector<std::future<शून्य>> tasks;
    प्रति (int32_t ith = 1; ith < n_तन्तवः; ith++) {
        ध्रुव int64_t start = (ith + 0) * n / n_तन्तवः;
        ध्रुव int64_t end   = (ith + 1) * n / n_तन्तवः;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_तन्तवः);

    प्रति (स्वचालित & task : tasks) {
        task.get();
    }
}

#यदि_अन्तः  // GGML_ZDNN_COMMON_HPP
