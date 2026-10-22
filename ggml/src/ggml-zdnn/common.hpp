#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ "zdnn.h"

#અન્તર્ભાવ "scratch.hpp"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <future>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <utility>
#અન્તર્ભાવ <vector>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    કરો {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } જ્યારે (0);

સંરચના ggml_backend_zdnn_device_context {
    પૂર્ણાંક zdnn_device;
    પૂર્ણાંક zdnn_device_ref_ગણતરી;

    તાર્કિક has_parmblkformat_0;
    તાર્કિક has_parmblkformat_1;  // checks for z17

    size_t max_size;

    અક્ષર name[128];
};

સંરચના ggml_backend_zdnn_buffer {
    શૂન્ય * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<દશાંશ> scales;          // per-row scales for quantized weights
    std::vector<દશાંશ> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    અક્ષર name[GGML_MAX_NAME];
};

સંરચના ggml_backend_zdnn_buffer_context {
    શૂન્ય * all_data;
    size_t all_size;
    તાર્કિક owned;

    પૂર્ણાંક n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

સંરચના ggml_backend_zdnn_context {
    પૂર્ણાંક device;
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
સ્થિર અંતઃરેખ શૂન્ય ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    અચલ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
સ્થિર અંતઃરેખ શૂન્ય ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    અચલ uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
સ્થિર અંતઃરેખ zdnn_ztensor ggml_zdnn_ઢગલો_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          અચલ zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    અચલ uint64_t ઢગલો_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    પસંદગી (pre_tfm_desc->layout) {
        સ્થિતિ ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } તોડો;
        સ્થિતિ ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } તોડો;
        મૂળભૂત:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*ઢગલો_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (અક્ષર *) src.buffer + s0*ઢગલો_bytes;
    view.buffer_size          = n * ઢગલો_bytes;
    પાછા view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
સ્થિર અંતઃરેખ અચલ zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    જો (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<દશાંશ> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    પાછા bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
સ્થિર અંતઃરેખ શૂન્ય ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, અચલ F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<શૂન્ય>> tasks;
    માટે (int32_t ith = 1; ith < n_threads; ith++) {
        અચલ int64_t start = (ith + 0) * n / n_threads;
        અચલ int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    માટે (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
