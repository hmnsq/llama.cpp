#اگر_تعریف_نشده GGML_ZDNN_COMMON_HPP
#تعریف_کن GGML_ZDNN_COMMON_HPP

#ضمن "ggml.h"
#ضمن "ggml-impl.h"

#ضمن "zdnn.h"

#ضمن "scratch.hpp"

#ضمن <algorithm>
#ضمن <cstdlib>
#ضمن <future>
#ضمن <map>
#ضمن <memory>
#ضمن <utility>
#ضمن <vector>

#تعریف_کن GGML_ZDNN_NAME    "zDNN"
#تعریف_کن GGML_ZDNN_نسخه ZDNN_VERNUM

#تعریف_کن ZDNN_CHECK(stmt)                \
    افعل {                                \
        zdnn_وضعیت وضعیت = (stmt);    \
        GGML_ASSERT(وضعیت == ZDNN_OK); \
    } طالما (0);

بنية ggml_backend_zdnn_device_context {
    صحيح zdnn_device;
    صحيح zdnn_device_ref_شمار;

    منطقي has_parmblkformat_0;
    منطقي has_parmblkformat_1;  // checks for z17

    size_t max_size;

    حرف name[128];
};

بنية ggml_backend_zdnn_میانگیر {
    فراغ * داده;
    ggml_backend_zdnn_میانگیر * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<عائم> scales;          // per-row scales for quantized weights
    std::vector<عائم> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    حرف name[GGML_MAX_NAME];
};

بنية ggml_backend_zdnn_میانگیر_context {
    فراغ * all_داده;
    size_t all_اندازه;
    منطقي owned;

    صحيح n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_میانگیر>> buffers;
};

بنية ggml_backend_zdnn_context {
    صحيح device;
    int32_t n_نخها;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_میانگیر>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
ساكن مضمن فراغ ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ثابت uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->میانگیر      = scratch.alloc(ctx->pool, size);
    ztensor->میانگیر_اندازه = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
ساكن مضمن فراغ ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    ثابت uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->میانگیر      = scratch.alloc(ctx->pool, size);
    ztensor->میانگیر_اندازه = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
ساكن مضمن zdnn_ztensor ggml_zdnn_پشته_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          ثابت zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.میانگیر_اندازه % n_stacks == 0);
    ثابت uint64_t پشته_بایتها = src.میانگیر_اندازه / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    بدل (pre_tfm_desc->layout) {
        حالة ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } اكسر;
        حالة ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } اكسر;
        افتراضي:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*پشته_بایتها);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.میانگیر               = (حرف *) src.میانگیر + s0*پشته_بایتها;
    view.میانگیر_اندازه          = n * پشته_بایتها;
    أرجع view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
ساكن مضمن ثابت zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_میانگیر> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    إذا (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_میانگیر>();

        std::vector<عائم> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.داده()));
    }

    أرجع bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
قالب <نام_نوع F>
ساكن مضمن فراغ ggml_zdnn_parallel_for(int32_t n_نخها, int64_t n, ثابت F & f) {
    n_نخها = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_نخها, n));

    std::vector<std::future<فراغ>> tasks;
    لكل (int32_t ith = 1; ith < n_نخها; ith++) {
        ثابت int64_t start = (ith + 0) * n / n_نخها;
        ثابت int64_t end   = (ith + 1) * n / n_نخها;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_نخها);

    لكل (خودکار & task : tasks) {
        task.get();
    }
}

#پایان_اگر  // GGML_ZDNN_COMMON_HPP
