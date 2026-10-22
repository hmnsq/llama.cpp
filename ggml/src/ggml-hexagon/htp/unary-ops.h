#جے_بنا_تعریف HTP_UNARY_OPS_H
#تعریف_کرو HTP_UNARY_OPS_H

#ضمن "hex-common.h"
#ضمن "htp-ops.h"

// Op-specific struct for precomputed unary params
بنية htp_unary_kernel_params {
    uint32_t  n_دھاگے;
    uint32_t  خانہ_tile;
    uint32_t  vtcm_صف_فی_thread;
    uint32_t  بلاک;
    uint32_t  نشر_بھار;

    uint32_t  vtcm_src0_ناپ_فی_thread;
    uint32_t  vtcm_src1_ناپ_فی_thread;
    uint32_t  vtcm_dst_ناپ_فی_thread;

    uint32_t  vtcm_src0_ناپ;
    uint32_t  vtcm_src1_ناپ;
    uint32_t  vtcm_dst_ناپ;

    uint32_t  src0_صف_ناپ_aligned;
    uint32_t  src1_صف_ناپ_aligned;
    uint32_t  dst_صف_ناپ_aligned;

    uint32_t  vtcm_ناپ;

    // Fastdiv helpers
    بنية fastdiv_اقدار div_ne01;
    بنية fastdiv_اقدار div_ne02;
    بنية fastdiv_اقدار div_ne012;
    بنية fastdiv_اقدار div_tpr;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#جے_انت

ساكن مضمن bool htp_op_is_unary(uint32_t opcode) {
    بدل (opcode) {
        حالة HTP_OP_CLAMP:
        حالة HTP_OP_LEAKY_RELU:
        حالة HTP_OP_NORM:
        حالة HTP_OP_RMS_NORM:
        حالة HTP_OP_RMS_NORM_MUL:
        حالة HTP_OP_SCALE:
        حالة HTP_OP_SQR:
        حالة HTP_OP_SQRT:
        حالة HTP_OP_UNARY_NEG:
        حالة HTP_OP_UNARY_EXP:
        حالة HTP_OP_UNARY_SIGMOID:
        حالة HTP_OP_UNARY_SILU:
        حالة HTP_OP_UNARY_GELU:
        حالة HTP_OP_UNARY_GELU_تیز:
        حالة HTP_OP_UNARY_GELU_ERF:
        حالة HTP_OP_UNARY_SOFTPLUS:
        حالة HTP_OP_UNARY_TANH:
        حالة HTP_OP_UNARY_ABS:
        حالة HTP_OP_UNARY_LOG:
        حالة HTP_OP_UNARY_RELU:
        حالة HTP_OP_UNARY_قدم:
        حالة HTP_OP_L2_NORM:
        حالة HTP_OP_TRI:
        حالة HTP_OP_HADAMARD:
            أرجع true;
        افتراضي:
            أرجع false;
    }
}

بنية htp_unary_vtcm_layout {
    size_t کل_بائٹاں;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بائٹاں;
    size_t src1_بائٹاں;
    size_t dst_بائٹاں;
};

ساكن مضمن فراغ htp_unary_vtcm_layout_build(
    بنية htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool نشر_بھار,
    uint32_t n_دھاگے,
    size_t vtcm_ناپ,
    size_t elem_ناپ,
    uint32_t * باہرلا_خانہ_tile,
    uint32_t * باہرلا_vtcm_صف_فی_thread
) {
    ثابت size_t dst_elem_ناپ      = (op == HTP_OP_HADAMARD) ? حجم(عائم) : elem_ناپ;
    ثابت size_t src0_انکڑے_صف_ناپ = ne00 * elem_ناپ;
    ثابت size_t dst_انکڑے_صف_ناپ  = ne10 * dst_elem_ناپ;

    ثابت size_t src0_صف_ناپ_aligned = hex_گیڑا_اتے(src0_انکڑے_صف_ناپ, 128);
    ثابت size_t dst_صف_ناپ_aligned  = hex_گیڑا_اتے(dst_انکڑے_صف_ناپ,  128);

    size_t src1_صف_ناپ_aligned = 0;
    إذا (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ثابت size_t src1_انکڑے_صف_ناپ = ne11 * حجم(عائم);
        src1_صف_ناپ_aligned = hex_گیڑا_اتے(src1_انکڑے_صف_ناپ, 128);
    }

    size_t vtcm_ناپ_فی_صف = 0;
    size_t vtcm_صف_فی_thread = 0;

    إذا (op == HTP_OP_RMS_NORM_MUL) {
        إذا (نشر_بھار) {
            size_t دستیاب_vtcm = vtcm_ناپ;
            size_t src1_vtcm_کل = n_دھاگے * src1_صف_ناپ_aligned;
            إذا (دستیاب_vtcm > src1_vtcm_کل) {
                دستیاب_vtcm -= src1_vtcm_کل;
            } وإلا {
                دستیاب_vtcm = 0;
            }
            vtcm_ناپ_فی_صف = 2 * (src0_صف_ناپ_aligned + dst_صف_ناپ_aligned);
            vtcm_صف_فی_thread = دستیاب_vtcm / (n_دھاگے * vtcm_ناپ_فی_صف);
        } وإلا {
            vtcm_ناپ_فی_صف = 2 * (src0_صف_ناپ_aligned + dst_صف_ناپ_aligned + src1_صف_ناپ_aligned);
            vtcm_صف_فی_thread = vtcm_ناپ / (n_دھاگے * vtcm_ناپ_فی_صف);
        }
    } وإلا {
        vtcm_ناپ_فی_صف   = 2 * (src0_صف_ناپ_aligned + dst_صف_ناپ_aligned);
        vtcm_صف_فی_thread = vtcm_ناپ / (n_دھاگے * vtcm_ناپ_فی_صف);
    }

    ثابت bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    ثابت bool is_f16 = (elem_ناپ == حجم(_Float16));
    uint32_t خانہ_tile = 0;

    إذا (vtcm_صف_فی_thread == 0 && !is_reduction && !is_f16) {
        ثابت size_t فی_thread_budget = vtcm_ناپ / n_دھاگے;
        ثابت size_t خانہ_tile_بائٹاں = hex_align_تھلے(فی_thread_budget / 4, 128);
        خانہ_tile = (uint32_t) (خانہ_tile_بائٹاں / elem_ناپ);

        L->src0_بائٹاں = خانہ_tile_بائٹاں * 2;
        L->dst_بائٹاں  = خانہ_tile_بائٹاں * 2;
        L->src1_بائٹاں = 0;
    } وإلا {
        L->src0_بائٹاں = src0_صف_ناپ_aligned * vtcm_صف_فی_thread * 2;
        L->dst_بائٹاں  = dst_صف_ناپ_aligned * vtcm_صف_فی_thread * 2;
        إذا (op == HTP_OP_RMS_NORM_MUL) {
            إذا (نشر_بھار) {
                L->src1_بائٹاں = src1_صف_ناپ_aligned;
            } وإلا {
                L->src1_بائٹاں = src1_صف_ناپ_aligned * vtcm_صف_فی_thread * 2;
            }
        } وإلا {
            L->src1_بائٹاں = 0;
        }
    }

    L->off_src0 = 0;
    إذا (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_بائٹاں * n_دھاگے;
        L->off_dst  = L->off_src1 + L->src1_بائٹاں * n_دھاگے;
    } وإلا {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_بائٹاں * n_دھاگے;
    }

    L->کل_بائٹاں = L->off_dst + L->dst_بائٹاں * n_دھاگے;

    *باہرلا_خانہ_tile = خانہ_tile;
    *باہرلا_vtcm_صف_فی_thread = vtcm_صف_فی_thread;
}

#جے_انت /* HTP_UNARY_OPS_H */
