#جے_بنا_تعریف HTP_CTX_H
#تعریف_کرو HTP_CTX_H

#ضمن "dma-queue.h"
#ضمن "hmx-queue.h"
#ضمن "htp-dcache.h"
#ضمن "htp-ops.h"
#ضمن "hex-profile.h"
#ضمن "work-queue.h"
#ضمن "hex-fastdiv.h"

#ضمن <assert.h>
#ضمن <dspqueue.h>
#ضمن <stdatomic.h>
#ضمن <stdint.h>
#ضمن <stdbool.h>

#جے_بنا_تعریف HTP_MAX_NTHREADS
#تعریف_کرو HTP_MAX_NTHREADS 10
#جے_انت

#تعریف_کرو HTP_MAX_MMAPS 64

// Memory mapping
بنية htp_mmap {
    uint64_t ناپ;
    uint64_t ادھار;
    uint32_t fd;
    uint32_t flags;
};

// Scratchpad state
بنية htp_spad {
    ثابت بنية htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 انکڑے;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  ناپ;            // total size
    uint32_t                  ناپ_فی_thread; // size per thread
};

بنية htp_mdev_گروہ {
    uint16_t              idx;
    uint16_t              گنتی;
    بنية fastdiv_اقدار گنتی_div;
    uint8_t *             fence_ادھار;
    uint32_t              fence_seq;
};

بنية htp_context;

// Context while processing an Op
// TODO: fold this into the main context
بنية htp_ops_context {
    بنية htp_context * ctx;

    تعداد htp_op_کوڈ    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];
    int32_t             kernel_params[HTP_OP_MAX_KERN_PARAMS];

    ثابت بنية htp_tensor * src[HTP_OP_MAX_INPUTS];
    اتحاد {
        ثابت بنية htp_tensor * dst;
        ثابت بنية htp_tensor * dsts[HTP_OP_MAX_OUTPUTS];
    };

    // TODO convert these to an array
    بنية htp_spad src0_spad;
    بنية htp_spad src1_spad;
    بنية htp_spad src2_spad;
    بنية htp_spad src3_spad;
    بنية htp_spad dst_spad;

    uint32_t              flags;
    uint32_t              n_دھاگے;
    بنية fastdiv_اقدار n_دھاگے_div;
    صحيح                   حالت;
};

// Main context for htp DSP backend
بنية htp_context {
    dspqueue_t             dsp_قطار;

    بنية htp_mmap        mmap[HTP_MAX_MMAPS];
    dma_قطار_t            dma[HTP_MAX_NTHREADS];
    بنية htp_thread_trace trace[HTP_MAX_NTHREADS + 1];
    work_قطار_t           work_قطار;
    hmx_قطار_t            hmx_قطار;

    uint32_t               n_دھاگے;
    بنية fastdiv_اقدار  n_دھاگے_div;

    صحيح                    thread_id;
    صحيح                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_ادھار;
    size_t                 vtcm_ناپ;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;
    بنية htp_dirty_range dirty_ranges[HTP_MAX_DIRTY_RANGES];

    // Persistent DDR scratchpad for MUL_MAT_ID mappings
    فراغ *                 ddr_spad_ادھار;
    size_t                 ddr_spad_ناپ;

    بنية htp_mdev_گروہ  mdev;
    بنية htp_ops_context octx;

    qurt_thread_t          main_thread;
    فراغ *                 main_ڈھیر;
    atomic_bool            killed;
    size_t                 footprint;
};

ساكن مضمن bool htp_ops_context_سیٹ_n_دھاگے(بنية htp_ops_context * octx, uint32_t n_دھاگے) {
    إذا (n_دھاگے == 0 || n_دھاگے > octx->ctx->n_دھاگے) {
        أرجع false;
    }

    إذا (n_دھاگے != octx->n_دھاگے) {
        octx->n_دھاگے = n_دھاگے;
        octx->n_دھاگے_div = n_دھاگے == octx->ctx->n_دھاگے
            ? octx->ctx->n_دھاگے_div
            : init_fastdiv_اقدار(n_دھاگے);
    }

    أرجع true;
}

ساكن مضمن فراغ htp_ops_context_سیٹ_حالت(بنية htp_ops_context * octx, صحيح حالت) {
    إذا (حالت > HTP_حالت_OK && octx->حالت == HTP_حالت_OK) {
        octx->حالت = حالت;
    }
}

صحيح op_matmul(بنية htp_ops_context * octx);
صحيح op_matmul_id(بنية htp_ops_context * octx);
صحيح op_matmul_nx(بنية htp_ops_context * octx);
صحيح op_matmul_id_nx(بنية htp_ops_context * octx);
صحيح op_ثنائی(بنية htp_ops_context * octx);
صحيح op_unary(بنية htp_ops_context * octx);
صحيح op_جوڑ(بنية htp_ops_context * octx);
صحيح op_جوڑ_صفاں(بنية htp_ops_context * octx);
صحيح op_argmax(بنية htp_ops_context * octx);
صحيح op_activations(بنية htp_ops_context * octx);
صحيح op_softmax(بنية htp_ops_context * octx);
صحيح op_جوڑو_id(بنية htp_ops_context * octx);
صحيح op_rope(بنية htp_ops_context * octx);
صحيح op_flash_attn_ext(بنية htp_ops_context * octx);
صحيح op_سیٹ_صفاں(بنية htp_ops_context * octx);
صحيح op_get_صفاں(بنية htp_ops_context * octx);
صحيح op_cpy(بنية htp_ops_context * octx);
صحيح op_repeat(بنية htp_ops_context * octx);
صحيح op_argsort(بنية htp_ops_context * octx);
صحيح op_چوٹی_k(بنية htp_ops_context * octx);
صحيح op_ssm_conv(بنية htp_ops_context * octx);
صحيح op_cumsum(بنية htp_ops_context * octx);
صحيح op_fill(بنية htp_ops_context * octx);
صحيح op_concat(بنية htp_ops_context * octx);
صحيح op_diag(بنية htp_ops_context * octx);
صحيح op_solve_tri(بنية htp_ops_context * octx);
صحيح op_gated_delta_net(بنية htp_ops_context * octx);
صحيح op_pad(بنية htp_ops_context * octx);
صحيح op_im2col(بنية htp_ops_context * octx);
صحيح op_im2col_3d(بنية htp_ops_context * octx);
صحيح op_pool_2d(بنية htp_ops_context * octx);
صحيح op_pool_1d(بنية htp_ops_context * octx);
صحيح op_allreduce(بنية htp_ops_context * octx);
صحيح op_roll(بنية htp_ops_context * octx);

#جے_انت /* HTP_CTX_H */
