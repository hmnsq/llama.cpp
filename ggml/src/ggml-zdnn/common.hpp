#अगर_अघोषित GGML_ZDNN_COMMON_HPP
#घोषणा GGML_ZDNN_COMMON_HPP

#समावेश "ggml.h"
#समावेश "ggml-impl.h"

#समावेश "zdnn.h"

#समावेश "scratch.hpp"

#समावेश <algorithm>
#समावेश <cstdlib>
#समावेश <future>
#समावेश <map>
#समावेश <memory>
#समावेश <utility>
#समावेश <vector>

#घोषणा GGML_ZDNN_NAME    "zDNN"
#घोषणा GGML_ZDNN_VERSION ZDNN_VERNUM

#घोषणा ZDNN_CHECK(stmt)                \
    करो {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } जबतक (0);

काष्ठा ggml_backend_zdnn_device_context {
    पूर्णांक zdnn_device;
    पूर्णांक zdnn_device_ref_गिनती;

    bool has_parmblkformat_0;
    bool has_parmblkformat_1;  // checks for z17

    माप_प्रकार max_size;

    अक्षर name[128];
};

काष्ठा ggml_backend_zdnn_बफर {
    व्योम * तथ्य;
    ggml_backend_zdnn_बफर * extra;  // for bias, etc.
    माप_प्रकार size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<भग्न> scales;          // per-row scales for quantized weights
    std::vector<भग्न> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    अक्षर name[GGML_MAX_NAME];
};

काष्ठा ggml_backend_zdnn_बफर_context {
    व्योम * all_तथ्य;
    माप_प्रकार all_आकार;
    bool owned;

    पूर्णांक n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_बफर>> buffers;
};

काष्ठा ggml_backend_zdnn_context {
    पूर्णांक device;
    int32_t n_threads;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_बफर>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
अटल अंतरभूत व्योम ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    स्थिर uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->बफर      = scratch.alloc(ctx->pool, size);
    ztensor->बफर_आकार = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
अटल अंतरभूत व्योम ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    स्थिर uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->बफर      = scratch.alloc(ctx->pool, size);
    ztensor->बफर_आकार = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
अटल अंतरभूत zdnn_ztensor ggml_zdnn_ढेर_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          स्थिर zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.बफर_आकार % n_stacks == 0);
    स्थिर uint64_t ढेर_bytes = src.बफर_आकार / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    चयन (pre_tfm_desc->layout) {
        हाल ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } अवरोध;
        हाल ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } अवरोध;
        शेष:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*ढेर_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.बफर               = (अक्षर *) src.बफर + s0*ढेर_bytes;
    view.बफर_आकार          = n * ढेर_bytes;
    वापस view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
अटल अंतरभूत स्थिर zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_बफर> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    अगर (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_बफर>();

        std::vector<भग्न> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.तथ्य()));
    }

    वापस bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
ढाँचा <typename F>
अटल अंतरभूत व्योम ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, स्थिर F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<व्योम>> tasks;
    क्रम (int32_t ith = 1; ith < n_threads; ith++) {
        स्थिर int64_t start = (ith + 0) * n / n_threads;
        स्थिर int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    क्रम (स्वतः & task : tasks) {
        task.लो();
    }
}

#पूर्ण_अगर  // GGML_ZDNN_COMMON_HPP
