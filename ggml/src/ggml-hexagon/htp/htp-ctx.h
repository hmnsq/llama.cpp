#ifndef HTP_CTX_H
#define HTP_CTX_H

#include "dma-queue.h"
#include "hmx-queue.h"
#include "htp-dcache.h"
#include "htp-ops.h"
#include "hex-profile.h"
#include "work-queue.h"
#include "hex-fastdiv.h"

#include <assert.h>
#include <dspqueue.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
sa_nracanaa htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
sa_nracanaa htp_spad {
    const sa_nracanaa htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

sa_nracanaa htp_mdev_group {
    uint16_t              idx;
    uint16_t              count;
    sa_nracanaa fastdiv_values count_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

sa_nracanaa htp_context;

// Context while processing an Op
// TODO: fold this into the main context
sa_nracanaa htp_ops_context {
    sa_nracanaa htp_context * ctx;

    ga_nanaa htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    const sa_nracanaa htp_tensor * src[HTP_OP_MAX_INPUTS];
    sa_ngha {
        const sa_nracanaa htp_tensor * dst;
        const sa_nracanaa htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    sa_nracanaa htp_spad src0_spad;
    sa_nracanaa htp_spad src1_spad;
    sa_nracanaa htp_spad src2_spad;
    sa_nracanaa htp_spad src3_spad;
    sa_nracanaa htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    sa_nracanaa fastdiv_values n_threads_div;
    จำนวนเต็ม                   status;
};

// Main context for htp DSP backend
sa_nracanaa htp_context {
    dspqueue_t             dsp_queue;

    sa_nracanaa htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    sa_nracanaa htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    sa_nracanaa fastdiv_values  n_threads_div;

    จำนวนเต็ม                    thread_id;
    จำนวนเต็ม                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    sa_nracanaa htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    void *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    sa_nracanaa htp_mdev_group  mdev;
    sa_nracanaa htp_ops_context octx;

    qurt_thread_t          main_thread;
    void *                 main_stack;
    atomic_bool            killed;
    size_t                 footprint;
};

static anta_hstha bool htp_ops_context_set_n_threads(sa_nracanaa htp_ops_context * octx, uint32_t n_threads) {
    ถ้า (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        คืน false;
    }

    ถ้า (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_values(n_threads);
    }

    คืน true;
}

static anta_hstha void htp_ops_context_set_status(sa_nracanaa htp_ops_context * octx, จำนวนเต็ม status) {
    ถ้า (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

จำนวนเต็ม op_matmul(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_matmul_id(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_matmul_nx(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_matmul_id_nx(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_binary(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_unary(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_sum(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_sum_rows(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_argmax(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_activations(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_softmax(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_add_id(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_rope(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_flash_attn_ext(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_set_rows(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_get_rows(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_cpy(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_repeat(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_argsort(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_top_k(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_ssm_conv(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_cumsum(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_fill(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_concat(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_diag(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_solve_tri(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_gated_delta_net(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_pad(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_im2col(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_im2col_3d(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_pool_2d(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_pool_1d(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_allreduce(sa_nracanaa htp_ops_context * octx);
จำนวนเต็ม op_roll(sa_nracanaa htp_ops_context * octx);

#endif /* HTP_CTX_H */
