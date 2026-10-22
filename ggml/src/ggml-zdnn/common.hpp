#నిర్వచితం_కాకపోతే GGML_ZDNN_COMMON_HPP
#నిర్వచించు GGML_ZDNN_COMMON_HPP

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ "zdnn.h"

#అన్తర్భావ "scratch.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <future>
#అన్తర్భావ <map>
#అన్తర్భావ <memory>
#అన్తర్భావ <utility>
#అన్తర్భావ <vector>

#నిర్వచించు GGML_ZDNN_NAME    "zDNN"
#నిర్వచించు GGML_ZDNN_సంస్కరణ ZDNN_VERNUM

#నిర్వచించు ZDNN_CHECK(stmt)                \
    చేయి {                                \
        zdnn_స్థితి స్థితి_2 = (stmt);    \
        GGML_ASSERT(స్థితి_2 == ZDNN_OK); \
    } అయితే (0);

నిర్మాణం ggml_backend_zdnn_device_context {
    పూర్ణాంకం zdnn_device;
    పూర్ణాంకం zdnn_device_ref_లెక్క;

    తార్కిక has_parmblkformat_0;
    తార్కిక has_parmblkformat_1;  // checks for z17

    size_t max_size;

    అక్షరం name[128];
};

నిర్మాణం ggml_backend_zdnn_బఫరు {
    శూన్య * దత్తాంశం;
    ggml_backend_zdnn_బఫరు * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    std::vector<దశాంశం> scales;          // per-row scales for quantized weights
    std::vector<దశాంశం> channel_scales;  // per-input-channel scales for quantized weights, applied to the inputs

    అక్షరం name[GGML_MAX_NAME];
};

నిర్మాణం ggml_backend_zdnn_బఫరు_context {
    శూన్య * all_దత్తాంశం;
    size_t all_కొలత;
    తార్కిక owned;

    పూర్ణాంకం n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_బఫరు>> buffers;
};

నిర్మాణం ggml_backend_zdnn_context {
    పూర్ణాంకం device;
    int32_t n_దారాలు;  // for the CPU work around zDNN calls
    ggml_cgraph * gf;

    // zero bias for stacked quantized matmuls, keyed by (n_stacks, ne0)
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<ggml_backend_zdnn_బఫరు>> zero_bias;

    ggml_zdnn_pool pool;  // scratch memory of the ops
};

//
// Scratch buffers
//

// the ztensor uses the memory of scratch, so scratch must outlive it
స్థిర అంతఃవరుస శూన్య ggml_zdnn_init_scratch_ztensor(ggml_backend_zdnn_context     * ctx,
                                                  zdnn_tensor_desc              * pre_tfm_desc,
                                                  zdnn_tensor_desc              * tfm_desc,
                                                  zdnn_ztensor                  * ztensor,
                                                  ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    స్థిరరాశి uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_quantized_ztensor(pre_tfm_desc, tfm_desc, 1.0f, 0.0f, ztensor);
    ztensor->బఫరు      = scratch.alloc(ctx->pool, size);
    ztensor->బఫరు_కొలత = size;
}

// same as ggml_zdnn_init_scratch_ztensor, for a DLFLOAT16 ztensor
స్థిర అంతఃవరుస శూన్య ggml_zdnn_init_scratch_ztensor_dlf16(ggml_backend_zdnn_context     * ctx,
                                                        zdnn_tensor_desc              * pre_tfm_desc,
                                                        zdnn_tensor_desc              * tfm_desc,
                                                        zdnn_ztensor                  * ztensor,
                                                        ggml_zdnn_pool_alloc<uint8_t> & scratch) {

    స్థిరరాశి uint64_t size = zdnn_getsize_ztensor(tfm_desc);

    zdnn_init_ztensor(pre_tfm_desc, tfm_desc, ztensor);
    ztensor->బఫరు      = scratch.alloc(ctx->pool, size);
    ztensor->బఫరు_కొలత = size;
}

//
// Stacked ztensors
//

// view of the stacks [s0, s0 + n) of a stacked ztensor, the stacks are the outermost dim of its buffer
// the view descs must outlive the view
స్థిర అంతఃవరుస zdnn_ztensor ggml_zdnn_దొంతర_view(zdnn_tensor_desc * pre_tfm_desc,
                                                zdnn_tensor_desc * tfm_desc,
                                          స్థిరరాశి zdnn_ztensor     & src,
                                                int64_t            n_stacks,
                                                int64_t            s0,
                                                int64_t            n) {

    GGML_ASSERT(src.బఫరు_కొలత % n_stacks == 0);
    స్థిరరాశి uint64_t దొంతర_బైట్లు = src.బఫరు_కొలత / n_stacks;

    *pre_tfm_desc = *src.pre_transformed_desc;
    *tfm_desc     = *src.transformed_desc;

    ఎంపిక (pre_tfm_desc->layout) {
        స్థితి ZDNN_3DS:
            {
                pre_tfm_desc->dim3 = n;
            } ఆపు;
        స్థితి ZDNN_2DS:
            {
                pre_tfm_desc->dim2 = n;
            } ఆపు;
        అప్రమేయ:
            GGML_ABORT("%s: unsupported stacked layout", __func__);
    }

    tfm_desc->dim4 = n;
    GGML_ASSERT(zdnn_getsize_ztensor(tfm_desc) == n*దొంతర_బైట్లు);

    zdnn_ztensor view = src;
    view.pre_transformed_desc = pre_tfm_desc;
    view.transformed_desc     = tfm_desc;
    view.బఫరు               = (అక్షరం *) src.బఫరు + s0*దొంతర_బైట్లు;
    view.బఫరు_కొలత          = n * దొంతర_బైట్లు;
    తిరిగి view;
}

// zero bias of n_stacks stacks, created once per shape as it never changes
స్థిర అంతఃవరుస స్థిరరాశి zdnn_ztensor & ggml_zdnn_zero_bias(ggml_backend_zdnn_context * ctx, int64_t n_stacks, int64_t ne0) {
    std::unique_ptr<ggml_backend_zdnn_బఫరు> & bias = ctx->zero_bias[{ n_stacks, ne0 }];

    ఒకవేళ (!bias) {
        bias = std::make_unique<ggml_backend_zdnn_బఫరు>();

        std::vector<దశాంశం> zeros(n_stacks * ne0, 0.0f);

        zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias->pre_tfm_desc, n_stacks, ne0);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&bias->pre_tfm_desc, &bias->tfm_desc));
        ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&bias->pre_tfm_desc, &bias->tfm_desc, &bias->ztensor));
        ZDNN_CHECK(zdnn_transform_ztensor(&bias->ztensor, zeros.దత్తాంశం()));
    }

    తిరిగి bias->ztensor;
}

//
// Threading
//

// split [0, n) into n_threads ranges, the calling thread runs the first range
మూస <రకం_పేరు F>
స్థిర అంతఃవరుస శూన్య ggml_zdnn_parallel_for(int32_t n_దారాలు, int64_t n, స్థిరరాశి F & f) {
    n_దారాలు = (int32_t)std::max<int64_t>(1, std::min<int64_t>(n_దారాలు, n));

    std::vector<std::future<శూన్య>> tasks;
    కొరకు (int32_t ith = 1; ith < n_దారాలు; ith++) {
        స్థిరరాశి int64_t start = (ith + 0) * n / n_దారాలు;
        స్థిరరాశి int64_t end   = (ith + 1) * n / n_దారాలు;
        tasks.push_back(std::async(std::launch::async, [&f, start, end]() { f(start, end); }));
    }

    f(0, n/n_దారాలు);

    కొరకు (స్వయంచాలక & task : tasks) {
        task.get();
    }
}

#అయితే_ముగింపు  // GGML_ZDNN_COMMON_HPP
