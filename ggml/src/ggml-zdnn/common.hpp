#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

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

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    करा {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } पर्यंत (0);

संरचना ggml_backend_zdnn_device_context {
    पूर्णांक zdnn_device;
    पूर्णांक zdnn_device_ref_मोजणी;

    तार्किक has_parmblkformat_0;
    तार्किक has_parmblkformat_1;  // checks for z17

    माप_प्रकार max_size;

    अक्षर name[128];
};

संरचना ggml_backend_zdnn_buffer {
    शून्य * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    माप_प्रकार size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<दशांश> scales;          // per-row scales for quantized weights
    std::vector<दशांश> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    अक्षर name[GGML_MAX_NAME];
};

संरचना ggml_backend_zdnn_buffer_context {
    शून्य * all_data;
    माप_प्रकार all_size;
    तार्किक owned;

    पूर्णांक n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

संरचना ggml_backend_zdnn_context {
    पूर्णांक device;
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
स्थिर ओळीत शून्य ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    स्थिरांक uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
स्थिर ओळीत शून्य ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    स्थिरांक uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
स्थिर ओळीत zdnn_ztensor ggml_zdnn_चळत_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          स्थिरांक zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    स्थिरांक uint64_t चळत_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    निवड (pre_tfm_desc->layout) {
        स्थिती ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } थांबा;
        स्थिती ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*चळत_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (अक्षर *) src.buffer + s0*चळत_bytes;
    view.buffer_size          = n * चळत_bytes;
    परत view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
स्थिर ओळीत स्थिरांक zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    जर (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<दशांश> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    परत bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
ढाँचा <typename F>
स्थिर ओळीत शून्य ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, स्थिरांक F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<शून्य>> tasks;
    साठी (int32_t ith = 1; ith < n_threads; ith++) {
        स्थिरांक int64_t start = (ith + 0) * n / n_threads;
        स्थिरांक int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    साठी (स्वतः & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
