#جيڪڏهن_وصف_نه_ٿيل HTP_UNARY_OPS_H
#وصف_ڪريو HTP_UNARY_OPS_H

#ضمن "hex-common.h"
#ضمن "htp-ops.h"

// Op-specific struct for precomputed unary params
بنية htp_unary_kernel_params {
    uint32_t  n_ڌاڳا;
    uint32_t  کانو_tile;
    uint32_t  vtcm_صف_في_thread;
    uint32_t  بلاڪ;
    uint32_t  نشر_وزن;

    uint32_t  vtcm_src0_سائيز_في_thread;
    uint32_t  vtcm_src1_سائيز_في_thread;
    uint32_t  vtcm_dst_سائيز_في_thread;

    uint32_t  vtcm_src0_سائيز;
    uint32_t  vtcm_src1_سائيز;
    uint32_t  vtcm_dst_سائيز;

    uint32_t  src0_صف_سائيز_aligned;
    uint32_t  src1_صف_سائيز_aligned;
    uint32_t  dst_صف_سائيز_aligned;

    uint32_t  vtcm_سائيز;

    // Fastdiv helpers
    بنية fastdiv_قدرون div_ne01;
    بنية fastdiv_قدرون div_ne02;
    بنية fastdiv_قدرون div_ne012;
    بنية fastdiv_قدرون div_tpr;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

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
        حالة HTP_OP_UNARY_GELU_تيز:
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
    size_t ڪل_بائيٽون;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بائيٽون;
    size_t src1_بائيٽون;
    size_t dst_بائيٽون;
};

ساكن مضمن فراغ htp_unary_vtcm_layout_build(
    بنية htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool نشر_وزن,
    uint32_t n_ڌاڳا,
    size_t vtcm_سائيز,
    size_t elem_سائيز,
    uint32_t * ٻاهريون_کانو_tile,
    uint32_t * ٻاهريون_vtcm_صف_في_thread
) {
    ثابت size_t dst_elem_سائيز      = (op == HTP_OP_HADAMARD) ? حجم(عائم) : elem_سائيز;
    ثابت size_t src0_معلومات_صف_سائيز = ne00 * elem_سائيز;
    ثابت size_t dst_معلومات_صف_سائيز  = ne10 * dst_elem_سائيز;

    ثابت size_t src0_صف_سائيز_aligned = hex_ڦيرو_مٿي(src0_معلومات_صف_سائيز, 128);
    ثابت size_t dst_صف_سائيز_aligned  = hex_ڦيرو_مٿي(dst_معلومات_صف_سائيز,  128);

    size_t src1_صف_سائيز_aligned = 0;
    إذا (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ثابت size_t src1_معلومات_صف_سائيز = ne11 * حجم(عائم);
        src1_صف_سائيز_aligned = hex_ڦيرو_مٿي(src1_معلومات_صف_سائيز, 128);
    }

    size_t vtcm_سائيز_في_صف = 0;
    size_t vtcm_صف_في_thread = 0;

    إذا (op == HTP_OP_RMS_NORM_MUL) {
        إذا (نشر_وزن) {
            size_t موجود_vtcm = vtcm_سائيز;
            size_t src1_vtcm_ڪل = n_ڌاڳا * src1_صف_سائيز_aligned;
            إذا (موجود_vtcm > src1_vtcm_ڪل) {
                موجود_vtcm -= src1_vtcm_ڪل;
            } وإلا {
                موجود_vtcm = 0;
            }
            vtcm_سائيز_في_صف = 2 * (src0_صف_سائيز_aligned + dst_صف_سائيز_aligned);
            vtcm_صف_في_thread = موجود_vtcm / (n_ڌاڳا * vtcm_سائيز_في_صف);
        } وإلا {
            vtcm_سائيز_في_صف = 2 * (src0_صف_سائيز_aligned + dst_صف_سائيز_aligned + src1_صف_سائيز_aligned);
            vtcm_صف_في_thread = vtcm_سائيز / (n_ڌاڳا * vtcm_سائيز_في_صف);
        }
    } وإلا {
        vtcm_سائيز_في_صف   = 2 * (src0_صف_سائيز_aligned + dst_صف_سائيز_aligned);
        vtcm_صف_في_thread = vtcm_سائيز / (n_ڌاڳا * vtcm_سائيز_في_صف);
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
    ثابت bool is_f16 = (elem_سائيز == حجم(_Float16));
    uint32_t کانو_tile = 0;

    إذا (vtcm_صف_في_thread == 0 && !is_reduction && !is_f16) {
        ثابت size_t في_thread_budget = vtcm_سائيز / n_ڌاڳا;
        ثابت size_t کانو_tile_بائيٽون = hex_align_هيٺ_ڏانهن(في_thread_budget / 4, 128);
        کانو_tile = (uint32_t) (کانو_tile_بائيٽون / elem_سائيز);

        L->src0_بائيٽون = کانو_tile_بائيٽون * 2;
        L->dst_بائيٽون  = کانو_tile_بائيٽون * 2;
        L->src1_بائيٽون = 0;
    } وإلا {
        L->src0_بائيٽون = src0_صف_سائيز_aligned * vtcm_صف_في_thread * 2;
        L->dst_بائيٽون  = dst_صف_سائيز_aligned * vtcm_صف_في_thread * 2;
        إذا (op == HTP_OP_RMS_NORM_MUL) {
            إذا (نشر_وزن) {
                L->src1_بائيٽون = src1_صف_سائيز_aligned;
            } وإلا {
                L->src1_بائيٽون = src1_صف_سائيز_aligned * vtcm_صف_في_thread * 2;
            }
        } وإلا {
            L->src1_بائيٽون = 0;
        }
    }

    L->off_src0 = 0;
    إذا (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_بائيٽون * n_ڌاڳا;
        L->off_dst  = L->off_src1 + L->src1_بائيٽون * n_ڌاڳا;
    } وإلا {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_بائيٽون * n_ڌاڳا;
    }

    L->ڪل_بائيٽون = L->off_dst + L->dst_بائيٽون * n_ڌاڳا;

    *ٻاهريون_کانو_tile = کانو_tile;
    *ٻاهريون_vtcm_صف_في_thread = vtcm_صف_في_thread;
}

#جيڪڏهن_پورو /* HTP_UNARY_OPS_H */
