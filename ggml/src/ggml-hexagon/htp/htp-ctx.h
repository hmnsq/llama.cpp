#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_CTX_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_CTX_H

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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_MAX_NTHREADS
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MAX_NTHREADS 10
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MAX_MMAPS 64

// Memory mapping
ರಚನೆ htp_mmap {
    uint64_t ಗಾತ್ರ_2;
    uint64_t ತಳ;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
ರಚನೆ htp_spad {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 ದತ್ತಾಂಶ;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  ಗಾತ್ರ_2;            // total size
    uint32_t                  ಗಾತ್ರ_ಪ್ರತಿ_thread; // size per thread
};

ರಚನೆ htp_mdev_ಗುಂಪು {
    uint16_t              idx;
    uint16_t              ಎಣಿಕೆ_2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಎಣಿಕೆ_div;
    uint8_t *             fence_ತಳ;
    uint32_t              fence_seq;
};

ರಚನೆ htp_context;

// Context while processing an Op
// TODO: fold this into the main context
ರಚನೆ htp_ops_context {
    ರಚನೆ htp_context * ctx;

    ಎಣಿಕೆ htp_op_ಸಂಕೇತ    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src[HTP_OP_MAX_INPUTS];
    ಒಕ್ಕೂಟ {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    ರಚನೆ htp_spad src0_spad;
    ರಚನೆ htp_spad src1_spad;
    ರಚನೆ htp_spad src2_spad;
    ರಚನೆ htp_spad src3_spad;
    ರಚನೆ htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_ಎಳೆಗಳು;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು n_ಎಳೆಗಳು_div;
    ಪೂರ್ಣಾಂಕ                   ಸ್ಥಿತಿ_2;
};

// Main context for htp DSP backend
ರಚನೆ htp_context {
    dspqueue_t             dsp_ಸರತಿ;

    ರಚನೆ htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_ಸರತಿ_t            dma[HTP_MAX_NTHREADS];
    ರಚನೆ htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_ಸರತಿ_t           work_ಸರತಿ;
    hmx_ಸರತಿ_t            hmx_ಸರತಿ;

    uint32_t               n_ಎಳೆಗಳು;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು  n_ಎಳೆಗಳು_div;

    ಪೂರ್ಣಾಂಕ                    thread_id;
    ಪೂರ್ಣಾಂಕ                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_ತಳ;
    size_t                 vtcm_ಗಾತ್ರ;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    ರಚನೆ htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    ಶೂನ್ಯ *                 ddr_spad_ತಳ;
    size_t                 ddr_spad_ಗಾತ್ರ;

    ರಚನೆ htp_mdev_ಗುಂಪು  mdev;
    ರಚನೆ htp_ops_context octx;

    qurt_thread_t          main_thread;
    ಶೂನ್ಯ *                 main_ರಾಶಿ;
    atomic_bool            killed;
    size_t                 footprint;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(ರಚನೆ htp_ops_context * octx, uint32_t n_ಎಳೆಗಳು) {
    ಆದರೆ (n_ಎಳೆಗಳು == 0 || n_ಎಳೆಗಳು > octx->ctx->n_ಎಳೆಗಳು) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಆದರೆ (n_ಎಳೆಗಳು != octx->n_ಎಳೆಗಳು) {
        octx->n_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
        octx->n_ಎಳೆಗಳು_div = n_ಎಳೆಗಳು == octx->ctx->n_ಎಳೆಗಳು
            ? octx->ctx->n_ಎಳೆಗಳು_div
            : init_fastdiv_ಮೌಲ್ಯಗಳು(n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(ರಚನೆ htp_ops_context * octx, ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2) {
    ಆದರೆ (ಸ್ಥಿತಿ_2 > HTP_ಸ್ಥಿತಿ_OK && octx->ಸ್ಥಿತಿ_2 == HTP_ಸ್ಥಿತಿ_OK) {
        octx->ಸ್ಥಿತಿ_2 = ಸ್ಥಿತಿ_2;
    }
}

ಪೂರ್ಣಾಂಕ op_matmul(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_matmul_id(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_matmul_nx(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_matmul_id_nx(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ದ್ವಿಮಾನ(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_unary(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ಮೊತ್ತ(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_argmax(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_activations(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_softmax(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ಕೂಡಿಸು_id(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_rope(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_flash_attn_ext(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_get_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_cpy(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_repeat(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_argsort(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ತುದಿ_k(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_ssm_conv(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_cumsum(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_fill(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_concat(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_diag(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_solve_tri(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_gated_delta_net(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_pad(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_im2col(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_im2col_3d(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_pool_2d(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_pool_1d(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_allreduce(ರಚನೆ htp_ops_context * octx);
ಪೂರ್ಣಾಂಕ op_roll(ರಚನೆ htp_ops_context * octx);

#ಆದರೆ_ಅಂತ್ಯ /* HTP_CTX_H */
