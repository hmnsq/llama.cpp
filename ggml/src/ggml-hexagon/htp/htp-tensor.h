#اگر_تعریف_نشده HTP_TENSOR_H
#تعریف_کن HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#تعریف_کن HTP_TENSOR_MDEV_سطر_اندازه   128

// Currently HMX requires 32x32 tile formats (do not change)
#تعریف_کن HTP_TENSOR_TILE_N_تعداد_ستون      32
#تعریف_کن HTP_TENSOR_TILE_N_ردیفها      32

#تعریف_کن HTP_TENSOR_TILE_اندازه_F16    2048
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q4_0   576
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q4_1   640
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q8_0   1088
#تعریف_کن HTP_TENSOR_TILE_اندازه_IQ4_NL 576
#تعریف_کن HTP_TENSOR_TILE_اندازه_MXFP4  544
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q4_K   640
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q5_K   768
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q6_K   896
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q3_K   512
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q2_K   336
#تعریف_کن HTP_TENSOR_TILE_K_Q1_0      128
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q1_0   576
#تعریف_کن HTP_TENSOR_TILE_K_Q2_0      64
#تعریف_کن HTP_TENSOR_TILE_اندازه_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

بنية htp_tensor_mdev_range {
    uint32_t شروع;
    uint32_t شمار;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    أرجع t && (t->داده >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_اندازه) {
    uint32_t بعدی_nb = type_اندازه;
    إذا (t->ne[0] != 1 && t->nb[0] != بعدی_nb) {
        أرجع false;
    }
    بعدی_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        إذا (t->ne[i] != 1 && t->nb[i] != بعدی_nb) {
            أرجع false;
        }
        بعدی_nb *= t->ne[i];
    }
    أرجع true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    أرجع t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_داده_aligned(ثابت بنية htp_tensor * t) {
    أرجع ((uintptr_t) t->داده & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_ردیف_افراز(ثابت بنية htp_tensor * t, uint32_t elem_اندازه) {
    إذا (!htp_tensor_mdev_داده_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_اندازه) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) != 0) أرجع false;
    إذا (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) != 0) أرجع false;
    إذا (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) != 0) أرجع false;
    أرجع true;
}

ساكن مضمن bool htp_tensor_mdev_ردیفها_در_هر_chunk(ثابت بنية htp_tensor * t, uint32_t elem_اندازه, uint32_t ردیف_اندازه, uint32_t * ردیفها_در_هر_chunk) {
    *ردیفها_در_هر_chunk = 0;

    إذا (!htp_tensor_mdev_داده_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_اندازه) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0)) {
        *ردیفها_در_هر_chunk = 1;
        أرجع true;
    }
    إذا (t->nb[1] == ردیف_اندازه &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *ردیفها_در_هر_chunk = (ردیف_اندازه > 0) ? (HTP_TENSOR_MDEV_سطر_اندازه / hex_بزرگترین_مقسوم_علیه_u32(ردیف_اندازه, HTP_TENSOR_MDEV_سطر_اندازه)) : 1;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_افراز(uint32_t مجموع_units, uint32_t units_در_هر_chunk, uint32_t mdev_idx, uint32_t mdev_شمار, ثابت بنية fastdiv_مقادیر * mdev_شمار_div) {
    بنية htp_tensor_mdev_range range = { 0, مجموع_units };

    إذا (mdev_شمار <= 1) {
        أرجع range;
    }

    إذا (units_در_هر_chunk == 0) {
        range.شروع = (mdev_idx == 0) ? 0 : مجموع_units;
        range.شمار = (mdev_idx == 0) ? مجموع_units : 0;
        أرجع range;
    }

    ثابت uint32_t مجموع_chunks = مجموع_units / units_در_هر_chunk;
    إذا (مجموع_chunks < mdev_شمار) {
        range.شروع = (mdev_idx == 0) ? 0 : مجموع_units;
        range.شمار = (mdev_idx == 0) ? مجموع_units : 0;
        أرجع range;
    }

    ثابت uint32_t chunks_در_هر_mdev = fastdiv(مجموع_chunks + mdev_شمار - 1, mdev_شمار_div);
    range.شروع = MIN(mdev_idx * chunks_در_هر_mdev * units_در_هر_chunk, مجموع_units);
    إذا (mdev_idx == mdev_شمار - 1) {
        range.شمار = مجموع_units - range.شروع;
    } وإلا {
        range.شمار = MIN(chunks_در_هر_mdev * units_در_هر_chunk, مجموع_units - range.شروع);
    }
    أرجع range;
}

ساكن مضمن uint32_t htp_tensor_get_tile_اندازه(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:   أرجع HTP_TENSOR_TILE_اندازه_Q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع HTP_TENSOR_TILE_اندازه_IQ4_NL;
        حالة HTP_TYPE_Q4_1:   أرجع HTP_TENSOR_TILE_اندازه_Q4_1;
        حالة HTP_TYPE_Q4_K:   أرجع HTP_TENSOR_TILE_اندازه_Q4_K;
        حالة HTP_TYPE_Q8_0:   أرجع HTP_TENSOR_TILE_اندازه_Q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع HTP_TENSOR_TILE_اندازه_Q5_K;
        حالة HTP_TYPE_Q6_K:   أرجع HTP_TENSOR_TILE_اندازه_Q6_K;
        حالة HTP_TYPE_Q3_K:   أرجع HTP_TENSOR_TILE_اندازه_Q3_K;
        حالة HTP_TYPE_Q2_K:   أرجع HTP_TENSOR_TILE_اندازه_Q2_K;
        حالة HTP_TYPE_Q1_0:   أرجع HTP_TENSOR_TILE_اندازه_Q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع HTP_TENSOR_TILE_اندازه_Q2_0;
        حالة HTP_TYPE_MXFP4:  أرجع HTP_TENSOR_TILE_اندازه_MXFP4;
        حالة HTP_TYPE_F16:    أرجع HTP_TENSOR_TILE_اندازه_F16;
        افتراضي:              أرجع 0;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_k(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q1_0: أرجع HTP_TENSOR_TILE_K_Q1_0;
        حالة HTP_TYPE_Q2_0: أرجع HTP_TENSOR_TILE_K_Q2_0;
        افتراضي:            أرجع 32;
    }
}

ساكن مضمن uint32_t htp_tensor_get_tile_padded_اندازه(صحيح type) {
    أرجع (htp_tensor_get_tile_اندازه(type) + 127u) & ~127u;
}

ساكن مضمن bool htp_tensor_is_tiled(ثابت بنية htp_tensor * t) {
    أرجع t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساكن مضمن uint32_t htp_tensor_get_ردیف_اندازه(صحيح type, uint32_t ne00) {
    بدل (type) {
        حالة HTP_TYPE_F32:    أرجع ne00 * 4;
        حالة HTP_TYPE_F16:
        حالة HTP_TYPE_BF16:   أرجع ne00 * 2;
        حالة HTP_TYPE_Q8_0:   أرجع (ne00 / 32) * 34;
        حالة HTP_TYPE_Q4_0:
        حالة HTP_TYPE_IQ4_NL: أرجع (ne00 / 32) * 18;
        حالة HTP_TYPE_Q4_1:   أرجع (ne00 / 32) * 20;
        حالة HTP_TYPE_MXFP4:  أرجع (ne00 / 32) * 17;
        حالة HTP_TYPE_Q2_K:   أرجع (ne00 / 256) * 84;
        حالة HTP_TYPE_Q2_0:   أرجع (ne00 / 64) * 18;
        حالة HTP_TYPE_Q3_K:   أرجع (ne00 / 256) * 110;
        حالة HTP_TYPE_Q4_K:   أرجع (ne00 / 256) * 144;
        حالة HTP_TYPE_Q5_K:   أرجع (ne00 / 256) * 176;
        حالة HTP_TYPE_Q6_K:   أرجع (ne00 / 256) * 210;
        حالة HTP_TYPE_Q1_0:   أرجع (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        حالة HTP_TYPE_I32:    أرجع ne00 * 4;
        حالة HTP_TYPE_I64:    أرجع ne00 * 8;
        افتراضي:              أرجع 0;
    }
}

#پایان_اگر // HTP_TENSOR_H
