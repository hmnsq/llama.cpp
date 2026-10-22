#ifndef HTP_CTX_H
#define HTP_CTX_H

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hmx-queue.h"
#ಅನ್ತರ್ಭಾವ "htp-dcache.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <dspqueue.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
ಸಂರಚನಾ htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ಸಂರಚನಾ htp_spad {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

ಸಂರಚನಾ htp_mdev_group {
    uint16_t              idx;
    uint16_t              ಎಣಿಕೆ;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು ಎಣಿಕೆ_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

ಸಂರಚನಾ htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ಸಂರಚನಾ htp_ops_context {
    ಸಂರಚನಾ htp_context * ctx;

    ಗಣನಾ htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src[HTP_OP_MAX_INPUTS];
    ಸಙ್ಘ {
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst;
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ಸಂರಚನಾ htp_spad src0_spad;
    ಸಂರಚನಾ htp_spad src1_spad;
    ಸಂರಚನಾ htp_spad src2_spad;
    ಸಂರಚನಾ htp_spad src3_spad;
    ಸಂರಚನಾ htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು n_threads_div;
    ಪೂರ್ಣಾಙ್ಕ                   status;
};

// Main context for htp DSP backend
ಸಂರಚನಾ htp_context {
    dspqueue_t             dsp_queue;

    ಸಂರಚನಾ htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    ಸಂರಚನಾ htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು  n_threads_div;

    ಪೂರ್ಣಾಙ್ಕ                    thread_id;
    ಪೂರ್ಣಾಙ್ಕ                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ಸಂರಚನಾ htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    ಶೂನ್ಯ *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    ಸಂರಚನಾ htp_mdev_group  mdev;
    ಸಂರಚನಾ htp_ops_context octx;

    qurt_thread_t          main_thread;
    ಶೂನ್ಯ *                 main_ರಾಶಿ;
    atomic_bool            killed;
    size_t                 footprint;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool htp_ops_context_set_n_threads(ಸಂರಚನಾ htp_ops_context * octx, uint32_t n_threads) {
    ಯದಿ (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    ಯದಿ (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_ಮೌಲ್ಯಗಳು(n_threads);
    }

    ಪ್ರತ್ಯಾಗಮ true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_ops_context_set_status(ಸಂರಚನಾ htp_ops_context * octx, ಪೂರ್ಣಾಙ್ಕ status) {
    ಯದಿ (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

ಪೂರ್ಣಾಙ್ಕ op_matmul(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_matmul_id(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_matmul_nx(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_matmul_id_nx(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_ದ್ವಿಮಾನ(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_unary(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_sum(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_sum_rows(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_argmax(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_activations(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_softmax(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_add_id(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_rope(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_flash_attn_ext(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_set_rows(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_get_rows(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_cpy(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_repeat(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_argsort(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_ತುದಿ_k(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_ssm_conv(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_cumsum(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_fill(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_concat(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_diag(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_solve_tri(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_gated_delta_net(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_pad(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_im2col(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_im2col_3d(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_pool_2d(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_pool_1d(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_allreduce(ಸಂರಚನಾ htp_ops_context * octx);
ಪೂರ್ಣಾಙ್ಕ op_roll(ಸಂರಚನಾ htp_ops_context * octx);

#endif /* HTP_CTX_H */
