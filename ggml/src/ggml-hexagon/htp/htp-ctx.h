#अगर_अघोषित HTP_CTX_H
#घोषणा HTP_CTX_H

#समावेश "dma-queue.h"
#समावेश "hmx-queue.h"
#समावेश "htp-dcache.h"
#समावेश "htp-ops.h"
#समावेश "hex-profile.h"
#समावेश "work-queue.h"
#समावेश "hex-fastdiv.h"

#समावेश <assert.h>
#समावेश <dspqueue.h>
#समावेश <stdatomic.h>
#समावेश <stdint.h>
#समावेश <stdbool.h>

#अगर_अघोषित HTP_MAX_NTHREADS
#घोषणा HTP_MAX_NTHREADS 10
#पूर्ण_अगर

#घोषणा HTP_MAX_MMAPS 64

// Memory mapping
काष्ठा htp_mmap {
    uint64_t आकार;
    uint64_t base;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
काष्ठा htp_spad {
    स्थिर काष्ठा htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 तथ्य;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  आकार;            // total size
    uint32_t                  आकार_per_thread; // size per thread
};

काष्ठा htp_mdev_group {
    uint16_t              idx;
    uint16_t              गिनती;
    काष्ठा fastdiv_मूल्यावली गिनती_div;
    uint8_t *             fence_base;
    uint32_t              fence_seq;
};

काष्ठा htp_context;

// Context while processing an Op
// TODO: fold this into the main context
काष्ठा htp_ops_context {
    काष्ठा htp_context * ctx;

    क्रमागत htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    स्थिर काष्ठा htp_tensor * src[HTP_OP_MAX_INPUTS];
    जोड़ {
        स्थिर काष्ठा htp_tensor * dst;
        स्थिर काष्ठा htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    काष्ठा htp_spad src0_spad;
    काष्ठा htp_spad src1_spad;
    काष्ठा htp_spad src2_spad;
    काष्ठा htp_spad src3_spad;
    काष्ठा htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_threads;
    काष्ठा fastdiv_मूल्यावली n_threads_div;
    पूर्णांक                   status;
};

// Main context for htp DSP backend
काष्ठा htp_context {
    dspqueue_t             dsp_queue;

    काष्ठा htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_queue_t            dma[HTP_MAX_NTHREADS];
    काष्ठा htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_queue_t           work_queue;
    hmx_queue_t            hmx_queue;

    uint32_t               n_threads;
    काष्ठा fastdiv_मूल्यावली  n_threads_div;

    पूर्णांक                    thread_id;
    पूर्णांक                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    माप_प्रकार                 vtcm_आकार;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    काष्ठा htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    व्योम *                 ddr_spad_base;
    माप_प्रकार                 ddr_spad_आकार;

    काष्ठा htp_mdev_group  mdev;
    काष्ठा htp_ops_context octx;

    qurt_thread_t          main_thread;
    व्योम *                 main_ढेर;
    atomic_bool            killed;
    माप_प्रकार                 footprint;
};

अटल अंतरभूत bool htp_ops_context_set_n_threads(काष्ठा htp_ops_context * octx, uint32_t n_threads) {
    अगर (n_threads == 0 || n_threads > octx->ctx->n_threads) {
        वापस false;
    }

    अगर (n_threads != octx->n_threads) {
        octx->n_threads = n_threads;
        octx->n_threads_div = n_threads == octx->ctx->n_threads
            ? octx->ctx->n_threads_div
            : init_fastdiv_मूल्यावली(n_threads);
    }

    वापस true;
}

अटल अंतरभूत व्योम htp_ops_context_set_status(काष्ठा htp_ops_context * octx, पूर्णांक status) {
    अगर (status > HTP_STATUS_OK && octx->status == HTP_STATUS_OK) {
        octx->status = status;
    }
}

पूर्णांक op_matmul(काष्ठा htp_ops_context * octx);
पूर्णांक op_matmul_id(काष्ठा htp_ops_context * octx);
पूर्णांक op_matmul_nx(काष्ठा htp_ops_context * octx);
पूर्णांक op_matmul_id_nx(काष्ठा htp_ops_context * octx);
पूर्णांक op_द्विआधारी(काष्ठा htp_ops_context * octx);
पूर्णांक op_unary(काष्ठा htp_ops_context * octx);
पूर्णांक op_योग(काष्ठा htp_ops_context * octx);
पूर्णांक op_योग_rows(काष्ठा htp_ops_context * octx);
पूर्णांक op_argmax(काष्ठा htp_ops_context * octx);
पूर्णांक op_activations(काष्ठा htp_ops_context * octx);
पूर्णांक op_softmax(काष्ठा htp_ops_context * octx);
पूर्णांक op_add_id(काष्ठा htp_ops_context * octx);
पूर्णांक op_rope(काष्ठा htp_ops_context * octx);
पूर्णांक op_flash_attn_ext(काष्ठा htp_ops_context * octx);
पूर्णांक op_set_rows(काष्ठा htp_ops_context * octx);
पूर्णांक op_get_rows(काष्ठा htp_ops_context * octx);
पूर्णांक op_cpy(काष्ठा htp_ops_context * octx);
पूर्णांक op_repeat(काष्ठा htp_ops_context * octx);
पूर्णांक op_argsort(काष्ठा htp_ops_context * octx);
पूर्णांक op_शीर्ष_k(काष्ठा htp_ops_context * octx);
पूर्णांक op_ssm_conv(काष्ठा htp_ops_context * octx);
पूर्णांक op_cumsum(काष्ठा htp_ops_context * octx);
पूर्णांक op_fill(काष्ठा htp_ops_context * octx);
पूर्णांक op_concat(काष्ठा htp_ops_context * octx);
पूर्णांक op_diag(काष्ठा htp_ops_context * octx);
पूर्णांक op_solve_tri(काष्ठा htp_ops_context * octx);
पूर्णांक op_gated_delta_net(काष्ठा htp_ops_context * octx);
पूर्णांक op_pad(काष्ठा htp_ops_context * octx);
पूर्णांक op_im2col(काष्ठा htp_ops_context * octx);
पूर्णांक op_im2col_3d(काष्ठा htp_ops_context * octx);
पूर्णांक op_pool_2d(काष्ठा htp_ops_context * octx);
पूर्णांक op_pool_1d(काष्ठा htp_ops_context * octx);
पूर्णांक op_allreduce(काष्ठा htp_ops_context * octx);
पूर्णांक op_roll(काष्ठा htp_ops_context * octx);

#पूर्ण_अगर /* HTP_CTX_H */
