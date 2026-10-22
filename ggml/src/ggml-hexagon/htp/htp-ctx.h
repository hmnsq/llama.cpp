#ifndef HTP_CTX_H
#define HTP_CTX_H

#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hmx-queue.h"
#અન્તર્ભાવ "htp-dcache.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "work-queue.h"
#અન્તર્ભાવ "hex-fastdiv.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <dspqueue.h>
#અન્તર્ભાવ <stdatomic.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
સંરચના htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
સંરચના htp_spad {
    અચલ સંરચના htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

સંરચના htp_mdev_group {
    uint16_t              idx;
    uint16_t              ગણતરી;
    સંરચના fastdiv_કિંમતો ગણતરી_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

સંરચના htp_context;

// Context while processing an Op
// TODO: fold this into the main context
સંરચના htp_ops_context {
    સંરચના htp_context * ctx;

    ગણના htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    અચલ સંરચના htp_tensor * src[HTP_OP_MAX_INPUTS];
    સંઘ {
        અચલ સંરચના htp_tensor * dst;
        અચલ સંરચના htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    સંરચના htp_spad src0_spad;
    સંરચના htp_spad src1_spad;
    સંરચના htp_spad src2_spad;
    સંરચના htp_spad src3_spad;
    સંરચના htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    સંરચના fastdiv_કિંમતો n_threads_div;
    પૂર્ણાંક                   status;
};

// Main context for htp DSP backend
સંરચના htp_context {
    dspqueue_t             dsp_queue;

    સંરચના htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    સંરચના htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    સંરચના fastdiv_કિંમતો  n_threads_div;

    પૂર્ણાંક                    thread_id;
    પૂર્ણાંક                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    સંરચના htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    શૂન્ય *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    સંરચના htp_mdev_group  mdev;
    સંરચના htp_ops_context octx;

    qurt_thread_t          main_thread;
    શૂન્ય *                 main_ઢગલો;
    atomic_bool            killed;
    size_t                 footprint;
};

સ્થિર અંતઃરેખ bool htp_ops_context_set_n_threads(સંરચના htp_ops_context * octx, uint32_t n_threads) {
    જો (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        પાછા false;
    }

    જો (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_કિંમતો(n_threads);
    }

    પાછા true;
}

સ્થિર અંતઃરેખ શૂન્ય htp_ops_context_set_status(સંરચના htp_ops_context * octx, પૂર્ણાંક status) {
    જો (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

પૂર્ણાંક op_matmul(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_matmul_id(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_matmul_nx(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_matmul_id_nx(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_દ્વિઅંકી(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_unary(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_sum(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_sum_rows(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_argmax(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_activations(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_softmax(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_add_id(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_rope(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_flash_attn_ext(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_set_rows(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_get_rows(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_cpy(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_repeat(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_argsort(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_ટોચ_k(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_ssm_conv(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_cumsum(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_fill(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_concat(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_diag(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_solve_tri(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_gated_delta_net(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_pad(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_im2col(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_im2col_3d(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_pool_2d(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_pool_1d(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_allreduce(સંરચના htp_ops_context * octx);
પૂર્ણાંક op_roll(સંરચના htp_ops_context * octx);

#endif /* HTP_CTX_H */
