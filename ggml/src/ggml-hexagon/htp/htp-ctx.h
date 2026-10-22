#ifndef HTP_CTX_H
#define HTP_CTX_H

#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hmx-queue.h"
#അന്തര്ഭാവ "htp-dcache.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "work-queue.h"
#അന്തര്ഭാവ "hex-fastdiv.h"

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <dspqueue.h>
#അന്തര്ഭാവ <stdatomic.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
ഘടന htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ഘടന htp_spad {
    സ്ഥിരം ഘടന htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

ഘടന htp_mdev_group {
    uint16_t              idx;
    uint16_t              എണ്ണം;
    ഘടന fastdiv_മൂല്യങ്ങൾ എണ്ണം_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

ഘടന htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ഘടന htp_ops_context {
    ഘടന htp_context * ctx;

    എണ്ണൽ htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    സ്ഥിരം ഘടന htp_tensor * src[HTP_OP_MAX_INPUTS];
    സംഘം {
        സ്ഥിരം ഘടന htp_tensor * dst;
        സ്ഥിരം ഘടന htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ഘടന htp_spad src0_spad;
    ഘടന htp_spad src1_spad;
    ഘടന htp_spad src2_spad;
    ഘടന htp_spad src3_spad;
    ഘടന htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    ഘടന fastdiv_മൂല്യങ്ങൾ n_threads_div;
    പൂർണ്ണസംഖ്യ                   status;
};

// Main context for htp DSP backend
ഘടന htp_context {
    dspqueue_t             dsp_queue;

    ഘടന htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    ഘടന htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    ഘടന fastdiv_മൂല്യങ്ങൾ  n_threads_div;

    പൂർണ്ണസംഖ്യ                    thread_id;
    പൂർണ്ണസംഖ്യ                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ഘടന htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    ശൂന്യം *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    ഘടന htp_mdev_group  mdev;
    ഘടന htp_ops_context octx;

    qurt_thread_t          main_thread;
    ശൂന്യം *                 main_അട്ടി;
    atomic_bool            killed;
    size_t                 footprint;
};

സ്ഥിതം വരിയിൽ bool htp_ops_context_set_n_threads(ഘടന htp_ops_context * octx, uint32_t n_threads) {
    എങ്കിൽ (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        തിരികെ false;
    }

    എങ്കിൽ (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_മൂല്യങ്ങൾ(n_threads);
    }

    തിരികെ true;
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_ops_context_set_status(ഘടന htp_ops_context * octx, പൂർണ്ണസംഖ്യ status) {
    എങ്കിൽ (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

പൂർണ്ണസംഖ്യ op_matmul(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_matmul_id(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_matmul_nx(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_matmul_id_nx(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_ദ്വിമാന(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_unary(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_sum(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_sum_rows(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_argmax(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_activations(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_softmax(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_add_id(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_rope(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_flash_attn_ext(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_set_rows(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_get_rows(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_cpy(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_repeat(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_argsort(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_മുകൾ_k(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_ssm_conv(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_cumsum(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_fill(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_concat(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_diag(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_solve_tri(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_gated_delta_net(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_pad(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_im2col(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_im2col_3d(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_pool_2d(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_pool_1d(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_allreduce(ഘടന htp_ops_context * octx);
പൂർണ്ണസംഖ്യ op_roll(ഘടന htp_ops_context * octx);

#endif /* HTP_CTX_H */
