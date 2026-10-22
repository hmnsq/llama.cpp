#ifndef HTP_CTX_H
#define HTP_CTX_H

#شامل "dma-queue.h"
#شامل "hmx-queue.h"
#شامل "htp-dcache.h"
#شامل "htp-ops.h"
#شامل "hex-profile.h"
#شامل "work-queue.h"
#شامل "hex-fastdiv.h"

#شامل <assert.h>
#شامل <dspqueue.h>
#شامل <stdatomic.h>
#شامل <stdint.h>
#شامل <stdbool.h>

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_MAX_MMAPS 64

// Memory mapping
ساختار htp_mmap {
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ساختار htp_spad {
    ثابت ساختار htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

ساختار htp_mdev_group {
    uint16_t              idx;
    uint16_t              شمار;
    ساختار fastdiv_مقادیر شمار_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

ساختار htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ساختار htp_ops_context {
    ساختار htp_context * ctx;

    شمارش htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ثابت ساختار htp_tensor * src[HTP_OP_MAX_INPUTS];
    اجتماع {
        ثابت ساختار htp_tensor * dst;
        ثابت ساختار htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ساختار htp_spad src0_spad;
    ساختار htp_spad src1_spad;
    ساختار htp_spad src2_spad;
    ساختار htp_spad src3_spad;
    ساختار htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    ساختار fastdiv_مقادیر n_threads_div;
    صحیح                   status;
};

// Main context for htp DSP backend
ساختار htp_context {
    dspqueue_t             dsp_queue;

    ساختار htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    ساختار htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    ساختار fastdiv_مقادیر  n_threads_div;

    صحیح                    thread_id;
    صحیح                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ساختار htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    تهی *                 ddr_spad_base;
    size_t                 ddr_spad_size;

    ساختار htp_mdev_group  mdev;
    ساختار htp_ops_context octx;

    qurt_thread_t          main_thread;
    تهی *                 main_پشته;
    atomic_bool            killed;
    size_t                 footprint;
};

ایستا درونخطی bool htp_ops_context_set_n_threads(ساختار htp_ops_context * octx, uint32_t n_threads) {
    اگر (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        بازگشت false;
    }

    اگر (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_مقادیر(n_threads);
    }

    بازگشت true;
}

ایستا درونخطی تهی htp_ops_context_set_status(ساختار htp_ops_context * octx, صحیح status) {
    اگر (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

صحیح op_matmul(ساختار htp_ops_context * octx);
صحیح op_matmul_id(ساختار htp_ops_context * octx);
صحیح op_matmul_nx(ساختار htp_ops_context * octx);
صحیح op_matmul_id_nx(ساختار htp_ops_context * octx);
صحیح op_دودویی(ساختار htp_ops_context * octx);
صحیح op_unary(ساختار htp_ops_context * octx);
صحیح op_sum(ساختار htp_ops_context * octx);
صحیح op_sum_rows(ساختار htp_ops_context * octx);
صحیح op_argmax(ساختار htp_ops_context * octx);
صحیح op_activations(ساختار htp_ops_context * octx);
صحیح op_softmax(ساختار htp_ops_context * octx);
صحیح op_add_id(ساختار htp_ops_context * octx);
صحیح op_rope(ساختار htp_ops_context * octx);
صحیح op_flash_attn_ext(ساختار htp_ops_context * octx);
صحیح op_set_rows(ساختار htp_ops_context * octx);
صحیح op_get_rows(ساختار htp_ops_context * octx);
صحیح op_cpy(ساختار htp_ops_context * octx);
صحیح op_repeat(ساختار htp_ops_context * octx);
صحیح op_argsort(ساختار htp_ops_context * octx);
صحیح op_بالاترین_k(ساختار htp_ops_context * octx);
صحیح op_ssm_conv(ساختار htp_ops_context * octx);
صحیح op_cumsum(ساختار htp_ops_context * octx);
صحیح op_fill(ساختار htp_ops_context * octx);
صحیح op_concat(ساختار htp_ops_context * octx);
صحیح op_diag(ساختار htp_ops_context * octx);
صحیح op_solve_tri(ساختار htp_ops_context * octx);
صحیح op_gated_delta_net(ساختار htp_ops_context * octx);
صحیح op_pad(ساختار htp_ops_context * octx);
صحیح op_im2col(ساختار htp_ops_context * octx);
صحیح op_im2col_3d(ساختار htp_ops_context * octx);
صحیح op_pool_2d(ساختار htp_ops_context * octx);
صحیح op_pool_1d(ساختار htp_ops_context * octx);
صحیح op_allreduce(ساختار htp_ops_context * octx);
صحیح op_roll(ساختار htp_ops_context * octx);

#endif /* HTP_CTX_H */
