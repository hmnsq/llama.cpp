#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_CTX_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_CTX_H

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-dcache.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <dspqueue.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_MAX_NTHREADS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MAX_NTHREADS 10
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MAX_MMAPS 64

// Memory mapping
ਬਣਤਰ htp_mmap {
    uint64_t ਮਾਪ;
    uint64_t ਅਧਾਰ;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ਬਣਤਰ htp_spad {
    ਅਚਲ ਬਣਤਰ htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 ਅੰਕੜੇ;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  ਮਾਪ;            // total size
    uint32_t                  ਮਾਪ_ਪ੍ਰਤੀ_thread; // size per thread
};

ਬਣਤਰ htp_mdev_ਸਮੂਹ {
    uint16_t              idx;
    uint16_t              ਗਿਣਤੀ_2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਗਿਣਤੀ_div;
    uint8_t *             fence_ਅਧਾਰ;
    uint32_t              fence_seq;
};

ਬਣਤਰ htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ਬਣਤਰ htp_ops_context {
    ਬਣਤਰ htp_context * ctx;

    ਗਿਣਤੀ htp_op_ਕੋਡ    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ਅਚਲ ਬਣਤਰ htp_tensor * src[HTP_OP_MAX_INPUTS];
    ਸੰਘ {
        ਅਚਲ ਬਣਤਰ htp_tensor * dst;
        ਅਚਲ ਬਣਤਰ htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ਬਣਤਰ htp_spad src0_spad;
    ਬਣਤਰ htp_spad src1_spad;
    ਬਣਤਰ htp_spad src2_spad;
    ਬਣਤਰ htp_spad src3_spad;
    ਬਣਤਰ htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_ਧਾਗੇ;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ n_ਧਾਗੇ_div;
    ਪੂਰਨ                   ਸਥਿਤੀ;
};

// Main context for htp DSP backend
ਬਣਤਰ htp_context {
    dspqueue_t             dsp_ਕਤਾਰ;

    ਬਣਤਰ htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_ਕਤਾਰ_t            dma[HTP_MAX_NTHREADS];
    ਬਣਤਰ htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_ਕਤਾਰ_t           work_ਕਤਾਰ;
    hmx_ਕਤਾਰ_t            hmx_ਕਤਾਰ;

    uint32_t               n_ਧਾਗੇ;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ  n_ਧਾਗੇ_div;

    ਪੂਰਨ                    thread_id;
    ਪੂਰਨ                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_ਅਧਾਰ;
    size_t                 vtcm_ਮਾਪ;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ਬਣਤਰ htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    ਖਾਲੀ *                 ddr_spad_ਅਧਾਰ;
    size_t                 ddr_spad_ਮਾਪ;

    ਬਣਤਰ htp_mdev_ਸਮੂਹ  mdev;
    ਬਣਤਰ htp_ops_context octx;

    qurt_thread_t          main_thread;
    ਖਾਲੀ *                 main_ਢੇਰ;
    atomic_bool            killed;
    size_t                 footprint;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(ਬਣਤਰ htp_ops_context * octx, uint32_t n_ਧਾਗੇ) {
    ਜੇ (n_ਧਾਗੇ == 0 || n_ਧਾਗੇ > octx->ctx->n_ਧਾਗੇ) {
        ਵਾਪਸ false;
    }

    ਜੇ (n_ਧਾਗੇ != octx->n_ਧਾਗੇ) {
        octx->n_ਧਾਗੇ = n_ਧਾਗੇ;
        octx->n_ਧਾਗੇ_div = n_ਧਾਗੇ == octx->ctx->n_ਧਾਗੇ
            ? octx->ctx->n_ਧਾਗੇ_div
            : init_fastdiv_ਮੁੱਲਾਂ(n_ਧਾਗੇ);
    }

    ਵਾਪਸ true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(ਬਣਤਰ htp_ops_context * octx, ਪੂਰਨ ਸਥਿਤੀ) {
    ਜੇ (ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK && octx->ਸਥਿਤੀ == HTP_ਸਥਿਤੀ_OK) {
        octx->ਸਥਿਤੀ = ਸਥਿਤੀ;
    }
}

ਪੂਰਨ op_matmul(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_matmul_id(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_matmul_nx(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_matmul_id_nx(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ਦੁਆਧਾਰੀ(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_unary(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ਜੋੜ(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ਜੋੜ_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_argmax(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_activations(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_softmax(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ਜੋੜੋ_id(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_rope(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_flash_attn_ext(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_get_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_cpy(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_repeat(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_argsort(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ਸਿਖਰ_k(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_ssm_conv(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_cumsum(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_fill(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_concat(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_diag(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_solve_tri(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_gated_delta_net(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_pad(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_im2col(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_im2col_3d(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_pool_2d(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_pool_1d(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_allreduce(ਬਣਤਰ htp_ops_context * octx);
ਪੂਰਨ op_roll(ਬਣਤਰ htp_ops_context * octx);

#ਜੇ_ਅੰਤ /* HTP_CTX_H */
