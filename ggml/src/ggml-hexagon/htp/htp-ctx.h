#నిర్వచితం_కాకపోతే HTP_CTX_H
#నిర్వచించు HTP_CTX_H

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hmx-queue.h"
#అన్తర్భావ "htp-dcache.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "work-queue.h"
#అన్తర్భావ "hex-fastdiv.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <dspqueue.h>
#అన్తర్భావ <stdatomic.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdbool.h>

#నిర్వచితం_కాకపోతే HTP_MAX_NTHREADS
#నిర్వచించు HTP_MAX_NTHREADS 10
#అయితే_ముగింపు

#నిర్వచించు HTP_MAX_MMAPS 64

// Memory mapping
నిర్మాణం htp_mmap {
    uint64_t కొలత;
    uint64_t పునాది;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
నిర్మాణం htp_spad {
    స్థిరరాశి నిర్మాణం htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 దత్తాంశం;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  కొలత;            // total size
    uint32_t                  కొలత_చొప్పున_thread; // size per thread
};

నిర్మాణం htp_mdev_సమూహం {
    uint16_t              idx;
    uint16_t              లెక్క;
    నిర్మాణం fastdiv_విలువలు లెక్క_div;
    uint8_t *             fence_పునాది;
    uint32_t              fence_seq;
};

నిర్మాణం htp_context;

// Context while processing an Op
// TODO: fold this into the main context
నిర్మాణం htp_ops_context {
    నిర్మాణం htp_context * ctx;

    గణన htp_op_సంకేతం    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    స్థిరరాశి నిర్మాణం htp_tensor * src[HTP_OP_MAX_INPUTS];
    సంఘం {
        స్థిరరాశి నిర్మాణం htp_tensor * dst;
        స్థిరరాశి నిర్మాణం htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    నిర్మాణం htp_spad src0_spad;
    నిర్మాణం htp_spad src1_spad;
    నిర్మాణం htp_spad src2_spad;
    నిర్మాణం htp_spad src3_spad;
    నిర్మాణం htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_దారాలు;
    నిర్మాణం fastdiv_విలువలు n_దారాలు_div;
    పూర్ణాంకం                   స్థితి_2;
};

// Main context for htp DSP backend
నిర్మాణం htp_context {
    dspqueue_t             dsp_వరుస;

    నిర్మాణం htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_వరుస_t            dma[HTP_MAX_NTHREADS];
    నిర్మాణం htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_వరుస_t           work_వరుస;
    hmx_వరుస_t            hmx_వరుస;

    uint32_t               n_దారాలు;
    నిర్మాణం fastdiv_విలువలు  n_దారాలు_div;

    పూర్ణాంకం                    thread_id;
    పూర్ణాంకం                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_పునాది;
    size_t                 vtcm_కొలత;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    నిర్మాణం htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    శూన్య *                 ddr_spad_పునాది;
    size_t                 ddr_spad_కొలత;

    నిర్మాణం htp_mdev_సమూహం  mdev;
    నిర్మాణం htp_ops_context octx;

    qurt_thread_t          main_thread;
    శూన్య *                 main_దొంతర;
    atomic_bool            killed;
    size_t                 footprint;
};

స్థిర అంతఃవరుస bool htp_ops_context_సమితి_n_దారాలు(నిర్మాణం htp_ops_context * octx, uint32_t n_దారాలు) {
    ఒకవేళ (n_దారాలు == 0 || n_దారాలు > octx->ctx->n_దారాలు) {
        తిరిగి false;
    }

    ఒకవేళ (n_దారాలు != octx->n_దారాలు) {
        octx->n_దారాలు = n_దారాలు;
        octx->n_దారాలు_div = n_దారాలు == octx->ctx->n_దారాలు
            ? octx->ctx->n_దారాలు_div
            : init_fastdiv_విలువలు(n_దారాలు);
    }

    తిరిగి true;
}

స్థిర అంతఃవరుస శూన్య htp_ops_context_సమితి_స్థితి(నిర్మాణం htp_ops_context * octx, పూర్ణాంకం స్థితి_2) {
    ఒకవేళ (స్థితి_2 > HTP_స్థితి_OK && octx->స్థితి_2 == HTP_స్థితి_OK) {
        octx->స్థితి_2 = స్థితి_2;
    }
}

పూర్ణాంకం op_matmul(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_matmul_id(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_matmul_nx(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_matmul_id_nx(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_ద్విమాన(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_unary(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_కూడిక(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_కూడిక_అడ్డువరుసలు(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_argmax(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_activations(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_softmax(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_కూడు_id(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_rope(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_flash_attn_ext(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_సమితి_అడ్డువరుసలు(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_get_అడ్డువరుసలు(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_cpy(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_repeat(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_argsort(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_శిఖరం_k(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_ssm_conv(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_cumsum(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_fill(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_concat(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_diag(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_solve_tri(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_gated_delta_net(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_pad(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_im2col(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_im2col_3d(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_pool_2d(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_pool_1d(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_allreduce(నిర్మాణం htp_ops_context * octx);
పూర్ణాంకం op_roll(నిర్మాణం htp_ops_context * octx);

#అయితే_ముగింపు /* HTP_CTX_H */
