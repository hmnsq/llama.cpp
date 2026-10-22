#ifndef HTP_CTX_H
#define HTP_CTX_H

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hmx-queue.h"
#अन्तर्भाव "htp-dcache.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "work-queue.h"
#अन्तर्भाव "hex-fastdiv.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <dspqueue.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
संरचना htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
संरचना htp_spad {
    ध्रुव संरचना htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

संरचना htp_mdev_group {
    uint16_t              idx;
    uint16_t              गिनती;
    संरचना fastdiv_मूल्यावली गिनती_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

संरचना htp_context;

// Context while processing an Op
// TODO: fold this into the main context
संरचना htp_ops_context {
    संरचना htp_context * ctx;

    गणना htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ध्रुव संरचना htp_tensor * src[HTP_OP_MAX_INPUTS];
    सङ्घ {
        ध्रुव संरचना htp_tensor * dst;
        ध्रुव संरचना htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    संरचना htp_spad src0_spad;
    संरचना htp_spad src1_spad;
    संरचना htp_spad src2_spad;
    संरचना htp_spad src3_spad;
    संरचना htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    संरचना fastdiv_मूल्यावली n_threads_div;
    पूर्णाङ्क                   status;
};

// Main context for htp DSP backend
संरचना htp_context {
    dspqueue_t             dsp_queue;

    संरचना htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    संरचना htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    संरचना fastdiv_मूल्यावली  n_threads_div;

    पूर्णाङ्क                    thread_id;
    पूर्णाङ्क                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    माप_प्रकार                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    संरचना htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    शून्य *                 ddr_spad_base;
    माप_प्रकार                 ddr_spad_size;

    संरचना htp_mdev_group  mdev;
    संरचना htp_ops_context octx;

    qurt_thread_t          main_thread;
    शून्य *                 main_ढेर;
    atomic_bool            killed;
    माप_प्रकार                 footprint;
};

स्थिर अन्तःस्थ bool htp_ops_context_set_n_threads(संरचना htp_ops_context * octx, uint32_t n_threads) {
    यदि (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        प्रत्यागम false;
    }

    यदि (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_मूल्यावली(n_threads);
    }

    प्रत्यागम true;
}

स्थिर अन्तःस्थ शून्य htp_ops_context_set_status(संरचना htp_ops_context * octx, पूर्णाङ्क status) {
    यदि (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

पूर्णाङ्क op_matmul(संरचना htp_ops_context * octx);
पूर्णाङ्क op_matmul_id(संरचना htp_ops_context * octx);
पूर्णाङ्क op_matmul_nx(संरचना htp_ops_context * octx);
पूर्णाङ्क op_matmul_id_nx(संरचना htp_ops_context * octx);
पूर्णाङ्क op_द्विआधारी(संरचना htp_ops_context * octx);
पूर्णाङ्क op_unary(संरचना htp_ops_context * octx);
पूर्णाङ्क op_sum(संरचना htp_ops_context * octx);
पूर्णाङ्क op_sum_rows(संरचना htp_ops_context * octx);
पूर्णाङ्क op_argmax(संरचना htp_ops_context * octx);
पूर्णाङ्क op_activations(संरचना htp_ops_context * octx);
पूर्णाङ्क op_softmax(संरचना htp_ops_context * octx);
पूर्णाङ्क op_add_id(संरचना htp_ops_context * octx);
पूर्णाङ्क op_rope(संरचना htp_ops_context * octx);
पूर्णाङ्क op_flash_attn_ext(संरचना htp_ops_context * octx);
पूर्णाङ्क op_set_rows(संरचना htp_ops_context * octx);
पूर्णाङ्क op_get_rows(संरचना htp_ops_context * octx);
पूर्णाङ्क op_cpy(संरचना htp_ops_context * octx);
पूर्णाङ्क op_repeat(संरचना htp_ops_context * octx);
पूर्णाङ्क op_argsort(संरचना htp_ops_context * octx);
पूर्णाङ्क op_शीर्ष_k(संरचना htp_ops_context * octx);
पूर्णाङ्क op_ssm_conv(संरचना htp_ops_context * octx);
पूर्णाङ्क op_cumsum(संरचना htp_ops_context * octx);
पूर्णाङ्क op_fill(संरचना htp_ops_context * octx);
पूर्णाङ्क op_concat(संरचना htp_ops_context * octx);
पूर्णाङ्क op_diag(संरचना htp_ops_context * octx);
पूर्णाङ्क op_solve_tri(संरचना htp_ops_context * octx);
पूर्णाङ्क op_gated_delta_net(संरचना htp_ops_context * octx);
पूर्णाङ्क op_pad(संरचना htp_ops_context * octx);
पूर्णाङ्क op_im2col(संरचना htp_ops_context * octx);
पूर्णाङ्क op_im2col_3d(संरचना htp_ops_context * octx);
पूर्णाङ्क op_pool_2d(संरचना htp_ops_context * octx);
पूर्णाङ्क op_pool_1d(संरचना htp_ops_context * octx);
पूर्णाङ्क op_allreduce(संरचना htp_ops_context * octx);
पूर्णाङ्क op_roll(संरचना htp_ops_context * octx);

#endif /* HTP_CTX_H */
