#ifndef HTP_CTX_H
#define HTP_CTX_H

#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hmx-queue.h"
#அந்தர்பாவ "htp-dcache.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "work-queue.h"
#அந்தர்பாவ "hex-fastdiv.h"

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <dspqueue.h>
#அந்தர்பாவ <stdatomic.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
கட்டமைப்பு htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
கட்டமைப்பு htp_spad {
    மாறிலி கட்டமைப்பு htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

கட்டமைப்பு htp_mdev_group {
    uint16_t              idx;
    uint16_t              மொத்தம்;
    கட்டமைப்பு fastdiv_மதிப்புகள் மொத்தம்_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

கட்டமைப்பு htp_context;

// Context while processing an Op
// TODO: fold this into the main context
கட்டமைப்பு htp_ops_context {
    கட்டமைப்பு htp_context * ctx;

    எண்ணிக்கை htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    மாறிலி கட்டமைப்பு htp_tensor * src[HTP_OP_MAX_INPUTS];
    ஒன்றிப்பு {
        மாறிலி கட்டமைப்பு htp_tensor * dst;
        மாறிலி கட்டமைப்பு htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    கட்டமைப்பு htp_spad src0_spad;
    கட்டமைப்பு htp_spad src1_spad;
    கட்டமைப்பு htp_spad src2_spad;
    கட்டமைப்பு htp_spad src3_spad;
    கட்டமைப்பு htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    கட்டமைப்பு fastdiv_மதிப்புகள் n_threads_div;
    முழு                   status;
};

// Main context for htp DSP backend
கட்டமைப்பு htp_context {
    dspqueue_t             dsp_queue;

    கட்டமைப்பு htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    கட்டமைப்பு htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    கட்டமைப்பு fastdiv_மதிப்புகள்  n_threads_div;

    முழு                    thread_id;
    முழு                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    கட்டமைப்பு htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    வெறுமை *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    கட்டமைப்பு htp_mdev_group  mdev;
    கட்டமைப்பு htp_ops_context octx;

    qurt_thread_t          main_thread;
    வெறுமை *                 main_அடுக்கு;
    atomic_bool            killed;
    size_t                 footprint;
};

நிலையான உள்வரி bool htp_ops_context_set_n_threads(கட்டமைப்பு htp_ops_context * octx, uint32_t n_threads) {
    எனில் (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        திரும்பு false;
    }

    எனில் (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_மதிப்புகள்(n_threads);
    }

    திரும்பு true;
}

நிலையான உள்வரி வெறுமை htp_ops_context_set_status(கட்டமைப்பு htp_ops_context * octx, முழு status) {
    எனில் (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

முழு op_matmul(கட்டமைப்பு htp_ops_context * octx);
முழு op_matmul_id(கட்டமைப்பு htp_ops_context * octx);
முழு op_matmul_nx(கட்டமைப்பு htp_ops_context * octx);
முழு op_matmul_id_nx(கட்டமைப்பு htp_ops_context * octx);
முழு op_இருமம்(கட்டமைப்பு htp_ops_context * octx);
முழு op_unary(கட்டமைப்பு htp_ops_context * octx);
முழு op_sum(கட்டமைப்பு htp_ops_context * octx);
முழு op_sum_rows(கட்டமைப்பு htp_ops_context * octx);
முழு op_argmax(கட்டமைப்பு htp_ops_context * octx);
முழு op_activations(கட்டமைப்பு htp_ops_context * octx);
முழு op_softmax(கட்டமைப்பு htp_ops_context * octx);
முழு op_add_id(கட்டமைப்பு htp_ops_context * octx);
முழு op_rope(கட்டமைப்பு htp_ops_context * octx);
முழு op_flash_attn_ext(கட்டமைப்பு htp_ops_context * octx);
முழு op_set_rows(கட்டமைப்பு htp_ops_context * octx);
முழு op_get_rows(கட்டமைப்பு htp_ops_context * octx);
முழு op_cpy(கட்டமைப்பு htp_ops_context * octx);
முழு op_repeat(கட்டமைப்பு htp_ops_context * octx);
முழு op_argsort(கட்டமைப்பு htp_ops_context * octx);
முழு op_உச்சி_k(கட்டமைப்பு htp_ops_context * octx);
முழு op_ssm_conv(கட்டமைப்பு htp_ops_context * octx);
முழு op_cumsum(கட்டமைப்பு htp_ops_context * octx);
முழு op_fill(கட்டமைப்பு htp_ops_context * octx);
முழு op_concat(கட்டமைப்பு htp_ops_context * octx);
முழு op_diag(கட்டமைப்பு htp_ops_context * octx);
முழு op_solve_tri(கட்டமைப்பு htp_ops_context * octx);
முழு op_gated_delta_net(கட்டமைப்பு htp_ops_context * octx);
முழு op_pad(கட்டமைப்பு htp_ops_context * octx);
முழு op_im2col(கட்டமைப்பு htp_ops_context * octx);
முழு op_im2col_3d(கட்டமைப்பு htp_ops_context * octx);
முழு op_pool_2d(கட்டமைப்பு htp_ops_context * octx);
முழு op_pool_1d(கட்டமைப்பு htp_ops_context * octx);
முழு op_allreduce(கட்டமைப்பு htp_ops_context * octx);
முழு op_roll(கட்டமைப்பு htp_ops_context * octx);

#endif /* HTP_CTX_H */
