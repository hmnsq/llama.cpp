#ifndef GGML_ZDNN_COMMON_HPP
#define GGML_ZDNN_COMMON_HPP

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ "zdnn.h"

#അന്തര്ഭാവ "scratch.hpp"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <future>
#അന്തര്ഭാവ <map>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <utility>
#അന്തര്ഭാവ <vector>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define ZDNN_CHECK(stmt)                \
    ചെയ്യുക {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } ആകുമ്പോൾ (0);

ഘടന ggml_backend_zdnn_device_context {
    പൂർണ്ണസംഖ്യ zdnn_device;
    പൂർണ്ണസംഖ്യ zdnn_device_ref_എണ്ണം;

    തർക്കം has_parmblkformat_0;
    തർക്കം has_parmblkformat_1;  // checks for z17

    size_t max_size;

    അക്ഷരം name[128];
};

ഘടന ggml_backend_zdnn_buffer {
    ശൂന്യം * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<ദശാംശം> scales;          // per-row scales for quantized weights
    std::vector<ദശാംശം> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    അക്ഷരം name[GGML_MAX_NAME];
};

ഘടന ggml_backend_zdnn_buffer_context {
    ശൂന്യം * all_data;
    size_t all_size;
    തർക്കം owned;

    പൂർണ്ണസംഖ്യ n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

ഘടന ggml_backend_zdnn_context {
    പൂർണ്ണസംഖ്യ device;
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
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    സ്ഥിരം uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    സ്ഥിരം uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->buffer      = scratch.alloc(ctx->pool, size);
    ztensor->buffer_size = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
സ്ഥിതം വരിയിൽ zdnn_ztensor ggml_zdnn_അട്ടി_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          സ്ഥിരം zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.buffer_size % n_stacks == 0);
    സ്ഥിരം uint64_t അട്ടി_bytes = src.buffer_size / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    തിരഞ്ഞെടുപ്പ് (pre_tfm_desc->layout) {
        അവസ്ഥ ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } നിർത്തുക;
        അവസ്ഥ ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*അട്ടി_bytes);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.buffer               = (അക്ഷരം *) src.buffer + s0*അട്ടി_bytes;
    view.buffer_size          = n * അട്ടി_bytes;
    തിരികെ view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
സ്ഥിതം വരിയിൽ സ്ഥിരം zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_buffer> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    എങ്കിൽ (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_buffer>();

        std::vector<ദശാംശം> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.data()));
    }

    തിരികെ bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
template <typename F>
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_zdnn_parallel_for(int32_t n_threads, int64_t n, സ്ഥിരം F & f) {
    n_threads = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_threads, n));

    std::vector<std::future<ശൂന്യം>> tasks;
    വേണ്ടി (int32_t ith = 1; ith < n_threads; ith++) {
        സ്ഥിരം int64_t start = (ith + 0) * n / n_threads;
        സ്ഥിരം int64_t end   = (ith + 1) * n / n_threads;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_threads);

    വേണ്ടി (auto & task : tasks) {
        task.get();
    }
}

#endif  // GGML_ZDNN_COMMON_HPP
