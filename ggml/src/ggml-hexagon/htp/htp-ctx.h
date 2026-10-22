#ifndef HTP_CTX_H
#define HTP_CTX_H

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hmx-queue.h"
#ଅନ୍ତର୍ଭାଵ "htp-dcache.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "work-queue.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <dspqueue.h>
#ଅନ୍ତର୍ଭାଵ <stdatomic.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
ସଂରଚନା htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ସଂରଚନା htp_spad {
    ଅଚଳ ସଂରଚନା htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

ସଂରଚନା htp_mdev_group {
    uint16_t              idx;
    uint16_t              ଗଣନା_2;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ ଗଣନା_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

ସଂରଚନା htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ସଂରଚନା htp_ops_context {
    ସଂରଚନା htp_context * ctx;

    ଗଣନା htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ଅଚଳ ସଂରଚନା htp_tensor * src[HTP_OP_MAX_INPUTS];
    ସଂଘ {
        ଅଚଳ ସଂରଚନା htp_tensor * dst;
        ଅଚଳ ସଂରଚନା htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ସଂରଚନା htp_spad src0_spad;
    ସଂରଚନା htp_spad src1_spad;
    ସଂରଚନା htp_spad src2_spad;
    ସଂରଚନା htp_spad src3_spad;
    ସଂରଚନା htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ n_threads_div;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                   status;
};

// Main context for htp DSP backend
ସଂରଚନା htp_context {
    dspqueue_t             dsp_queue;

    ସଂରଚନା htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    ସଂରଚନା htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ  n_threads_div;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                    thread_id;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ସଂରଚନା htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    ଶୂନ୍ୟ *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    ସଂରଚନା htp_mdev_group  mdev;
    ସଂରଚନା htp_ops_context octx;

    qurt_thread_t          main_thread;
    ଶୂନ୍ୟ *                 main_ଗଦା;
    atomic_bool            killed;
    size_t                 footprint;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_ops_context_set_n_threads(ସଂରଚନା htp_ops_context * octx, uint32_t n_threads) {
    ଯଦି (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        ଫେରନ୍ତୁ false;
    }

    ଯଦି (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(n_threads);
    }

    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_ops_context_set_status(ସଂରଚନା htp_ops_context * octx, ପୂର୍ଣ୍ଣସଂଖ୍ୟା status) {
    ଯଦି (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_id(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_nx(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_id_nx(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_ଦ୍ୱିମିକ(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_unary(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_sum(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_sum_rows(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_argmax(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_activations(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_softmax(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_add_id(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_rope(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_flash_attn_ext(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_set_rows(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_get_rows(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_cpy(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_repeat(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_argsort(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_ଶୀର୍ଷ_k(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_ssm_conv(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_cumsum(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_fill(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_concat(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_diag(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_solve_tri(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_gated_delta_net(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_pad(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_im2col(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_im2col_3d(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_pool_2d(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_pool_1d(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_allreduce(ସଂରଚନା htp_ops_context * octx);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_roll(ସଂରଚନା htp_ops_context * octx);

#endif /* HTP_CTX_H */
