#ifndef HTP_CTX_H
#define HTP_CTX_H

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hmx-queue.h"
#অন্তর্ভাব "htp-dcache.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "work-queue.h"
#অন্তর্ভাব "hex-fastdiv.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <dspqueue.h>
#অন্তর্ভাব <stdatomic.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
কাঠামো htp_mmap {
    uint64_t আকার_2;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
কাঠামো htp_spad {
    ধ্রুবক কাঠামো htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 তথ্য;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  আকার_2;            // total size
    uint32_t                  আকার_per_thread; // size per thread
};

কাঠামো htp_mdev_group {
    uint16_t              idx;
    uint16_t              গুনতি;
    কাঠামো fastdiv_মানসমূহ গুনতি_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

কাঠামো htp_context;

// Context while processing an Op
// TODO: fold this into the main context
কাঠামো htp_ops_context {
    কাঠামো htp_context * ctx;

    গণনা htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ধ্রুবক কাঠামো htp_tensor * src[HTP_OP_MAX_INPUTS];
    সংঘ {
        ধ্রুবক কাঠামো htp_tensor * dst;
        ধ্রুবক কাঠামো htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    কাঠামো htp_spad src0_spad;
    কাঠামো htp_spad src1_spad;
    কাঠামো htp_spad src2_spad;
    কাঠামো htp_spad src3_spad;
    কাঠামো htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    কাঠামো fastdiv_মানসমূহ n_threads_div;
    পূর্ণসংখ্যা                   status;
};

// Main context for htp DSP backend
কাঠামো htp_context {
    dspqueue_t             dsp_queue;

    কাঠামো htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    কাঠামো htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    কাঠামো fastdiv_মানসমূহ  n_threads_div;

    পূর্ণসংখ্যা                    thread_id;
    পূর্ণসংখ্যা                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_আকার;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    কাঠামো htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    শূন্য *                 ddr_spad_base;
    size_t                 ddr_spad_আকার;

    কাঠামো htp_mdev_group  mdev;
    কাঠামো htp_ops_context octx;

    qurt_thread_t          main_thread;
    শূন্য *                 main_স্তূপ;
    atomic_bool            killed;
    size_t                 footprint;
};

স্থির অন্তঃরেখ bool htp_ops_context_set_n_threads(কাঠামো htp_ops_context * octx, uint32_t n_threads) {
    যদি (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        ফেরত false;
    }

    যদি (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_মানসমূহ(n_threads);
    }

    ফেরত true;
}

স্থির অন্তঃরেখ শূন্য htp_ops_context_set_status(কাঠামো htp_ops_context * octx, পূর্ণসংখ্যা status) {
    যদি (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

পূর্ণসংখ্যা op_matmul(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_matmul_id(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_matmul_nx(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_matmul_id_nx(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_দ্বিমিক(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_unary(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_যোগ(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_যোগ_rows(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_argmax(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_activations(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_softmax(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_add_id(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_rope(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_flash_attn_ext(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_set_rows(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_get_rows(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_cpy(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_repeat(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_argsort(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_শীর্ষ_k(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_ssm_conv(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_cumsum(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_fill(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_concat(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_diag(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_solve_tri(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_gated_delta_net(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_pad(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_im2col(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_im2col_3d(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_pool_2d(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_pool_1d(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_allreduce(কাঠামো htp_ops_context * octx);
পূর্ণসংখ্যা op_roll(কাঠামো htp_ops_context * octx);

#endif /* HTP_CTX_H */
