#اگر_غیر_تعریف_شدہ HTP_UNARY_OPS_H
#تعریف_کرو HTP_UNARY_OPS_H

#شامل "hex-common.h"
#شامل "htp-ops.h"

// Op-specific struct for precomputed unary params
ساخت htp_unary_kernel_params {
    uint32_t  n_دھاگے;
    uint32_t  ستون_tile;
    uint32_t  vtcm_صف_فی_thread;
    uint32_t  بلاک;
    uint32_t  نشر_وزن;

    uint32_t  vtcm_src0_سائز_فی_thread;
    uint32_t  vtcm_src1_سائز_فی_thread;
    uint32_t  vtcm_dst_سائز_فی_thread;

    uint32_t  vtcm_src0_سائز;
    uint32_t  vtcm_src1_سائز;
    uint32_t  vtcm_dst_سائز;

    uint32_t  src0_صف_سائز_aligned;
    uint32_t  src1_صف_سائز_aligned;
    uint32_t  dst_صف_سائز_aligned;

    uint32_t  vtcm_سائز;

    // Fastdiv helpers
    ساخت fastdiv_اقدار div_ne01;
    ساخت fastdiv_اقدار div_ne02;
    ساخت fastdiv_اقدار div_ne012;
    ساخت fastdiv_اقدار div_tpr;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساکن اندرونی bool htp_op_is_unary(uint32_t opcode) {
    انتخاب (opcode) {
        حالت HTP_OP_CLAMP:
        حالت HTP_OP_LEAKY_RELU:
        حالت HTP_OP_NORM:
        حالت HTP_OP_RMS_NORM:
        حالت HTP_OP_RMS_NORM_MUL:
        حالت HTP_OP_SCALE:
        حالت HTP_OP_SQR:
        حالت HTP_OP_SQRT:
        حالت HTP_OP_UNARY_NEG:
        حالت HTP_OP_UNARY_EXP:
        حالت HTP_OP_UNARY_SIGMOID:
        حالت HTP_OP_UNARY_SILU:
        حالت HTP_OP_UNARY_GELU:
        حالت HTP_OP_UNARY_GELU_تیز:
        حالت HTP_OP_UNARY_GELU_ERF:
        حالت HTP_OP_UNARY_SOFTPLUS:
        حالت HTP_OP_UNARY_TANH:
        حالت HTP_OP_UNARY_ABS:
        حالت HTP_OP_UNARY_LOG:
        حالت HTP_OP_UNARY_RELU:
        حالت HTP_OP_UNARY_قدم:
        حالت HTP_OP_L2_NORM:
        حالت HTP_OP_TRI:
        حالت HTP_OP_HADAMARD:
            واپس true;
        طےشدہ:
            واپس false;
    }
}

ساخت htp_unary_vtcm_layout {
    size_t کل_بائٹیں;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بائٹیں;
    size_t src1_بائٹیں;
    size_t dst_بائٹیں;
};

ساکن اندرونی خالی htp_unary_vtcm_layout_build(
    ساخت htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool نشر_وزن,
    uint32_t n_دھاگے,
    size_t vtcm_سائز,
    size_t elem_سائز,
    uint32_t * برآمد_ستون_tile,
    uint32_t * برآمد_vtcm_صف_فی_thread
) {
    ثابت size_t dst_elem_سائز      = (op == HTP_OP_HADAMARD) ? حجم(اعشاری) : elem_سائز;
    ثابت size_t src0_ڈیٹا_صف_سائز = ne00 * elem_سائز;
    ثابت size_t dst_ڈیٹا_صف_سائز  = ne10 * dst_elem_سائز;

    ثابت size_t src0_صف_سائز_aligned = hex_باری_اوپر(src0_ڈیٹا_صف_سائز, 128);
    ثابت size_t dst_صف_سائز_aligned  = hex_باری_اوپر(dst_ڈیٹا_صف_سائز,  128);

    size_t src1_صف_سائز_aligned = 0;
    اگر (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ثابت size_t src1_ڈیٹا_صف_سائز = ne11 * حجم(اعشاری);
        src1_صف_سائز_aligned = hex_باری_اوپر(src1_ڈیٹا_صف_سائز, 128);
    }

    size_t vtcm_سائز_فی_صف = 0;
    size_t vtcm_صف_فی_thread = 0;

    اگر (op == HTP_OP_RMS_NORM_MUL) {
        اگر (نشر_وزن) {
            size_t دستیاب_vtcm = vtcm_سائز;
            size_t src1_vtcm_کل = n_دھاگے * src1_صف_سائز_aligned;
            اگر (دستیاب_vtcm > src1_vtcm_کل) {
                دستیاب_vtcm -= src1_vtcm_کل;
            } ورنہ {
                دستیاب_vtcm = 0;
            }
            vtcm_سائز_فی_صف = 2 * (src0_صف_سائز_aligned + dst_صف_سائز_aligned);
            vtcm_صف_فی_thread = دستیاب_vtcm / (n_دھاگے * vtcm_سائز_فی_صف);
        } ورنہ {
            vtcm_سائز_فی_صف = 2 * (src0_صف_سائز_aligned + dst_صف_سائز_aligned + src1_صف_سائز_aligned);
            vtcm_صف_فی_thread = vtcm_سائز / (n_دھاگے * vtcm_سائز_فی_صف);
        }
    } ورنہ {
        vtcm_سائز_فی_صف   = 2 * (src0_صف_سائز_aligned + dst_صف_سائز_aligned);
        vtcm_صف_فی_thread = vtcm_سائز / (n_دھاگے * vtcm_سائز_فی_صف);
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
    ثابت bool is_f16 = (elem_سائز == حجم(_Float16));
    uint32_t ستون_tile = 0;

    اگر (vtcm_صف_فی_thread == 0 && !is_reduction && !is_f16) {
        ثابت size_t فی_thread_budget = vtcm_سائز / n_دھاگے;
        ثابت size_t ستون_tile_بائٹیں = hex_align_نیچے_کو(فی_thread_budget / 4, 128);
        ستون_tile = (uint32_t) (ستون_tile_بائٹیں / elem_سائز);

        L->src0_بائٹیں = ستون_tile_بائٹیں * 2;
        L->dst_بائٹیں  = ستون_tile_بائٹیں * 2;
        L->src1_بائٹیں = 0;
    } ورنہ {
        L->src0_بائٹیں = src0_صف_سائز_aligned * vtcm_صف_فی_thread * 2;
        L->dst_بائٹیں  = dst_صف_سائز_aligned * vtcm_صف_فی_thread * 2;
        اگر (op == HTP_OP_RMS_NORM_MUL) {
            اگر (نشر_وزن) {
                L->src1_بائٹیں = src1_صف_سائز_aligned;
            } ورنہ {
                L->src1_بائٹیں = src1_صف_سائز_aligned * vtcm_صف_فی_thread * 2;
            }
        } ورنہ {
            L->src1_بائٹیں = 0;
        }
    }

    L->off_src0 = 0;
    اگر (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_بائٹیں * n_دھاگے;
        L->off_dst  = L->off_src1 + L->src1_بائٹیں * n_دھاگے;
    } ورنہ {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_بائٹیں * n_دھاگے;
    }

    L->کل_بائٹیں = L->off_dst + L->dst_بائٹیں * n_دھاگے;

    *برآمد_ستون_tile = ستون_tile;
    *برآمد_vtcm_صف_فی_thread = vtcm_صف_فی_thread;
}

#اگر_ختم /* HTP_UNARY_OPS_H */
