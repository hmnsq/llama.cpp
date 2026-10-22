#اگر_غیر_تعریف_شدہ HTP_CTX_H
#تعریف_کرو HTP_CTX_H

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

#اگر_غیر_تعریف_شدہ HTP_MAX_NTHREADS
#تعریف_کرو HTP_MAX_NTHREADS 10
#اگر_ختم

#تعریف_کرو HTP_MAX_MMAPS 64

// Memory mapping
ساخت htp_mmap {
    uint64_t سائز;
    uint64_t بنیاد;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ساخت htp_spad {
    ثابت ساخت htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 ڈیٹا;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  سائز;            // total size
    uint32_t                  سائز_فی_thread; // size per thread
};

ساخت htp_mdev_گروہ {
    uint16_t              idx;
    uint16_t              گنتی;
    ساخت fastdiv_اقدار گنتی_div;
    uint8_t *             fence_بنیاد;
    uint32_t              fence_seq;
};

ساخت htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ساخت htp_ops_context {
    ساخت htp_context * ctx;

    شمار htp_op_رمز    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ثابت ساخت htp_tensor * src[HTP_OP_MAX_INPUTS];
    اتحاد {
        ثابت ساخت htp_tensor * dst;
        ثابت ساخت htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ساخت htp_spad src0_spad;
    ساخت htp_spad src1_spad;
    ساخت htp_spad src2_spad;
    ساخت htp_spad src3_spad;
    ساخت htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_دھاگے;
    ساخت fastdiv_اقدار n_دھاگے_div;
    صحیح                   حالت_2;
};

// Main context for htp DSP backend
ساخت htp_context {
    dspqueue_t             dsp_قطار;

    ساخت htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_قطار_t            dma[HTP_MAX_NTHREADS];
    ساخت htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_قطار_t           work_قطار;
    hmx_قطار_t            hmx_قطار;

    uint32_t               n_دھاگے;
    ساخت fastdiv_اقدار  n_دھاگے_div;

    صحیح                    thread_id;
    صحیح                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_بنیاد;
    size_t                 vtcm_سائز;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ساخت htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    خالی *                 ddr_spad_بنیاد;
    size_t                 ddr_spad_سائز;

    ساخت htp_mdev_گروہ  mdev;
    ساخت htp_ops_context octx;

    qurt_thread_t          main_thread;
    خالی *                 main_ڈھیر;
    atomic_bool            killed;
    size_t                 footprint;
};

ساکن اندرونی bool htp_ops_context_سیٹ_n_دھاگے(ساخت htp_ops_context * octx, uint32_t n_دھاگے) {
    اگر (n_دھاگے == 0 || n_دھاگے > octx->ctx->n_دھاگے) {
        واپس false;
    }

    اگر (n_دھاگے != octx->n_دھاگے) {
        octx->n_دھاگے = n_دھاگے;
        octx->n_دھاگے_div = n_دھاگے == octx->ctx->n_دھاگے
            ? octx->ctx->n_دھاگے_div
            : init_fastdiv_اقدار(n_دھاگے);
    }

    واپس true;
}

ساکن اندرونی خالی htp_ops_context_سیٹ_حالت(ساخت htp_ops_context * octx, صحیح حالت_2) {
    اگر (حالت_2 > HTP_حالت_OK && octx->حالت_2 == HTP_حالت_OK) {
        octx->حالت_2 = حالت_2;
    }
}

صحیح op_matmul(ساخت htp_ops_context * octx);
صحیح op_matmul_id(ساخت htp_ops_context * octx);
صحیح op_matmul_nx(ساخت htp_ops_context * octx);
صحیح op_matmul_id_nx(ساخت htp_ops_context * octx);
صحیح op_ثنائی(ساخت htp_ops_context * octx);
صحیح op_unary(ساخت htp_ops_context * octx);
صحیح op_مجموع(ساخت htp_ops_context * octx);
صحیح op_مجموع_صفیں(ساخت htp_ops_context * octx);
صحیح op_argmax(ساخت htp_ops_context * octx);
صحیح op_activations(ساخت htp_ops_context * octx);
صحیح op_softmax(ساخت htp_ops_context * octx);
صحیح op_جوڑو_id(ساخت htp_ops_context * octx);
صحیح op_rope(ساخت htp_ops_context * octx);
صحیح op_flash_attn_ext(ساخت htp_ops_context * octx);
صحیح op_سیٹ_صفیں(ساخت htp_ops_context * octx);
صحیح op_get_صفیں(ساخت htp_ops_context * octx);
صحیح op_cpy(ساخت htp_ops_context * octx);
صحیح op_repeat(ساخت htp_ops_context * octx);
صحیح op_argsort(ساخت htp_ops_context * octx);
صحیح op_چوٹی_k(ساخت htp_ops_context * octx);
صحیح op_ssm_conv(ساخت htp_ops_context * octx);
صحیح op_cumsum(ساخت htp_ops_context * octx);
صحیح op_fill(ساخت htp_ops_context * octx);
صحیح op_concat(ساخت htp_ops_context * octx);
صحیح op_diag(ساخت htp_ops_context * octx);
صحیح op_solve_tri(ساخت htp_ops_context * octx);
صحیح op_gated_delta_net(ساخت htp_ops_context * octx);
صحیح op_pad(ساخت htp_ops_context * octx);
صحیح op_im2col(ساخت htp_ops_context * octx);
صحیح op_im2col_3d(ساخت htp_ops_context * octx);
صحیح op_pool_2d(ساخت htp_ops_context * octx);
صحیح op_pool_1d(ساخت htp_ops_context * octx);
صحیح op_allreduce(ساخت htp_ops_context * octx);
صحیح op_roll(ساخت htp_ops_context * octx);

#اگر_ختم /* HTP_CTX_H */
