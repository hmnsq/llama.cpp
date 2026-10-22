#ifndef HTP_CTX_H
#define HTP_CTX_H

#অন্তর্ভাৱ "dma-queue.h"
#অন্তর্ভাৱ "hmx-queue.h"
#অন্তর্ভাৱ "htp-dcache.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "hex-profile.h"
#অন্তর্ভাৱ "work-queue.h"
#অন্তর্ভাৱ "hex-fastdiv.h"

#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <dspqueue.h>
#অন্তর্ভাৱ <stdatomic.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
সংরচনা htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
সংরচনা htp_spad {
    ধ্রুৱ সংরচনা htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

সংরচনা htp_mdev_group {
    uint16_t              idx;
    uint16_t              গুনতি;
    সংরচনা fastdiv_মানসমূহ গুনতি_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

সংরচনা htp_context;

// Context while processing an Op
// TODO: fold this into the main context
সংরচনা htp_ops_context {
    সংরচনা htp_context * ctx;

    গণনা htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ধ্রুৱ সংরচনা htp_tensor * src[HTP_OP_MAX_INPUTS];
    সঙ্ঘ {
        ধ্রুৱ সংরচনা htp_tensor * dst;
        ধ্রুৱ সংরচনা htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    সংরচনা htp_spad src0_spad;
    সংরচনা htp_spad src1_spad;
    সংরচনা htp_spad src2_spad;
    সংরচনা htp_spad src3_spad;
    সংরচনা htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    সংরচনা fastdiv_মানসমূহ n_threads_div;
    পূর্ণাঙ্ক                   status;
};

// Main context for htp DSP backend
সংরচনা htp_context {
    dspqueue_t             dsp_queue;

    সংরচনা htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    সংরচনা htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    সংরচনা fastdiv_মানসমূহ  n_threads_div;

    পূর্ণাঙ্ক                    thread_id;
    পূর্ণাঙ্ক                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    সংরচনা htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    শূন্য *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    সংরচনা htp_mdev_group  mdev;
    সংরচনা htp_ops_context octx;

    qurt_thread_t          main_thread;
    শূন্য *                 main_স্তূপ;
    atomic_bool            killed;
    size_t                 footprint;
};

স্থির অন্তঃস্থ bool htp_ops_context_set_n_threads(সংরচনা htp_ops_context * octx, uint32_t n_threads) {
    যদি (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        ঘূৰাই false;
    }

    যদি (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_মানসমূহ(n_threads);
    }

    ঘূৰাই true;
}

স্থির অন্তঃস্থ শূন্য htp_ops_context_set_status(সংরচনা htp_ops_context * octx, পূর্ণাঙ্ক status) {
    যদি (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

পূর্ণাঙ্ক op_matmul(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_matmul_id(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_matmul_nx(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_matmul_id_nx(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_দ্বিমিক(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_unary(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_sum(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_sum_rows(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_argmax(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_activations(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_softmax(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_add_id(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_rope(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_flash_attn_ext(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_set_rows(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_get_rows(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_cpy(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_repeat(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_argsort(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_শীৰ্ষ_k(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_ssm_conv(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_cumsum(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_fill(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_concat(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_diag(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_solve_tri(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_gated_delta_net(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_pad(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_im2col(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_im2col_3d(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_pool_2d(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_pool_1d(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_allreduce(সংরচনা htp_ops_context * octx);
পূর্ণাঙ্ক op_roll(সংরচনা htp_ops_context * octx);

#endif /* HTP_CTX_H */
