#اگر_تعریف_نشده HTP_UNARY_OPS_H
#تعریف_کن HTP_UNARY_OPS_H

#ضمن "hex-common.h"
#ضمن "htp-ops.h"

// Op-specific struct for precomputed unary params
بنية htp_unary_kernel_params {
    uint32_t  n_نخها;
    uint32_t  ستون_tile;
    uint32_t  vtcm_ردیف_در_هر_thread;
    uint32_t  بلوک;
    uint32_t  پخش_وزن;

    uint32_t  vtcm_src0_اندازه_در_هر_thread;
    uint32_t  vtcm_src1_اندازه_در_هر_thread;
    uint32_t  vtcm_dst_اندازه_در_هر_thread;

    uint32_t  vtcm_src0_اندازه;
    uint32_t  vtcm_src1_اندازه;
    uint32_t  vtcm_dst_اندازه;

    uint32_t  src0_ردیف_اندازه_aligned;
    uint32_t  src1_ردیف_اندازه_aligned;
    uint32_t  dst_ردیف_اندازه_aligned;

    uint32_t  vtcm_اندازه;

    // Fastdiv helpers
    بنية fastdiv_مقادیر div_ne01;
    بنية fastdiv_مقادیر div_ne02;
    بنية fastdiv_مقادیر div_ne012;
    بنية fastdiv_مقادیر div_tpr;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#پایان_اگر

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
        حالة HTP_OP_UNARY_GELU_سریع:
        حالة HTP_OP_UNARY_GELU_ERF:
        حالة HTP_OP_UNARY_SOFTPLUS:
        حالة HTP_OP_UNARY_TANH:
        حالة HTP_OP_UNARY_ABS:
        حالة HTP_OP_UNARY_LOG:
        حالة HTP_OP_UNARY_RELU:
        حالة HTP_OP_UNARY_گام:
        حالة HTP_OP_L2_NORM:
        حالة HTP_OP_TRI:
        حالة HTP_OP_HADAMARD:
            أرجع true;
        افتراضي:
            أرجع false;
    }
}

بنية htp_unary_vtcm_layout {
    size_t مجموع_بایتها;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بایتها;
    size_t src1_بایتها;
    size_t dst_بایتها;
};

ساكن مضمن فراغ htp_unary_vtcm_layout_build(
    بنية htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool پخش_وزن,
    uint32_t n_نخها,
    size_t vtcm_اندازه,
    size_t elem_اندازه,
    uint32_t * بیرون_ستون_tile,
    uint32_t * بیرون_vtcm_ردیف_در_هر_thread
) {
    ثابت size_t dst_elem_اندازه      = (op == HTP_OP_HADAMARD) ? حجم(عائم) : elem_اندازه;
    ثابت size_t src0_داده_ردیف_اندازه = ne00 * elem_اندازه;
    ثابت size_t dst_داده_ردیف_اندازه  = ne10 * dst_elem_اندازه;

    ثابت size_t src0_ردیف_اندازه_aligned = hex_دور_به_بالا(src0_داده_ردیف_اندازه, 128);
    ثابت size_t dst_ردیف_اندازه_aligned  = hex_دور_به_بالا(dst_داده_ردیف_اندازه,  128);

    size_t src1_ردیف_اندازه_aligned = 0;
    إذا (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ثابت size_t src1_داده_ردیف_اندازه = ne11 * حجم(عائم);
        src1_ردیف_اندازه_aligned = hex_دور_به_بالا(src1_داده_ردیف_اندازه, 128);
    }

    size_t vtcm_اندازه_در_هر_ردیف = 0;
    size_t vtcm_ردیف_در_هر_thread = 0;

    إذا (op == HTP_OP_RMS_NORM_MUL) {
        إذا (پخش_وزن) {
            size_t موجود_vtcm = vtcm_اندازه;
            size_t src1_vtcm_مجموع = n_نخها * src1_ردیف_اندازه_aligned;
            إذا (موجود_vtcm > src1_vtcm_مجموع) {
                موجود_vtcm -= src1_vtcm_مجموع;
            } وإلا {
                موجود_vtcm = 0;
            }
            vtcm_اندازه_در_هر_ردیف = 2 * (src0_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned);
            vtcm_ردیف_در_هر_thread = موجود_vtcm / (n_نخها * vtcm_اندازه_در_هر_ردیف);
        } وإلا {
            vtcm_اندازه_در_هر_ردیف = 2 * (src0_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned + src1_ردیف_اندازه_aligned);
            vtcm_ردیف_در_هر_thread = vtcm_اندازه / (n_نخها * vtcm_اندازه_در_هر_ردیف);
        }
    } وإلا {
        vtcm_اندازه_در_هر_ردیف   = 2 * (src0_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned);
        vtcm_ردیف_در_هر_thread = vtcm_اندازه / (n_نخها * vtcm_اندازه_در_هر_ردیف);
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
    ثابت bool is_f16 = (elem_اندازه == حجم(_Float16));
    uint32_t ستون_tile = 0;

    إذا (vtcm_ردیف_در_هر_thread == 0 && !is_reduction && !is_f16) {
        ثابت size_t در_هر_thread_budget = vtcm_اندازه / n_نخها;
        ثابت size_t ستون_tile_بایتها = hex_align_پایین(در_هر_thread_budget / 4, 128);
        ستون_tile = (uint32_t) (ستون_tile_بایتها / elem_اندازه);

        L->src0_بایتها = ستون_tile_بایتها * 2;
        L->dst_بایتها  = ستون_tile_بایتها * 2;
        L->src1_بایتها = 0;
    } وإلا {
        L->src0_بایتها = src0_ردیف_اندازه_aligned * vtcm_ردیف_در_هر_thread * 2;
        L->dst_بایتها  = dst_ردیف_اندازه_aligned * vtcm_ردیف_در_هر_thread * 2;
        إذا (op == HTP_OP_RMS_NORM_MUL) {
            إذا (پخش_وزن) {
                L->src1_بایتها = src1_ردیف_اندازه_aligned;
            } وإلا {
                L->src1_بایتها = src1_ردیف_اندازه_aligned * vtcm_ردیف_در_هر_thread * 2;
            }
        } وإلا {
            L->src1_بایتها = 0;
        }
    }

    L->off_src0 = 0;
    إذا (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_بایتها * n_نخها;
        L->off_dst  = L->off_src1 + L->src1_بایتها * n_نخها;
    } وإلا {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_بایتها * n_نخها;
    }

    L->مجموع_بایتها = L->off_dst + L->dst_بایتها * n_نخها;

    *بیرون_ستون_tile = ستون_tile;
    *بیرون_vtcm_ردیف_در_هر_thread = vtcm_ردیف_در_هر_thread;
}

#پایان_اگر /* HTP_UNARY_OPS_H */
