#که_نه_تعریف_شوی HTP_TENSOR_H
#تعریف_کړه HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#تعریف_کړه HTP_TENSOR_MDEV_کرښه_اندازه   128

// Currently HMX requires 32x32 tile formats (do not change)
#تعریف_کړه HTP_TENSOR_TILE_N_ستون_شمېر      32
#تعریف_کړه HTP_TENSOR_TILE_N_لیکې      32

#تعریف_کړه HTP_TENSOR_TILE_اندازه_F16    2048
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q4_0   576
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q4_1   640
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q8_0   1088
#تعریف_کړه HTP_TENSOR_TILE_اندازه_IQ4_NL 576
#تعریف_کړه HTP_TENSOR_TILE_اندازه_MXFP4  544
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q4_K   640
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q5_K   768
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q6_K   896
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q3_K   512
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q2_K   336
#تعریف_کړه HTP_TENSOR_TILE_K_Q1_0      128
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q1_0   576
#تعریف_کړه HTP_TENSOR_TILE_K_Q2_0      64
#تعریف_کړه HTP_TENSOR_TILE_اندازه_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

بنية htp_tensor_mdev_range {
    uint32_t پیل;
    uint32_t شمېر;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    أرجع t && (t->معلومات >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_اندازه) {
    uint32_t بل_nb = type_اندازه;
    إذا (t->ne[0] != 1 && t->nb[0] != بل_nb) {
        أرجع false;
    }
    بل_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        إذا (t->ne[i] != 1 && t->nb[i] != بل_nb) {
            أرجع false;
        }
        بل_nb *= t->ne[i];
    }
    أرجع true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    أرجع t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_معلومات_aligned(ثابت بنية htp_tensor * t) {
    أرجع ((uintptr_t) t->معلومات & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_لیکه_برخه_کول(ثابت بنية htp_tensor * t, uint32_t elem_اندازه) {
    إذا (!htp_tensor_mdev_معلومات_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_اندازه) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) != 0) أرجع false;
    إذا (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) != 0) أرجع false;
    إذا (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) != 0) أرجع false;
    أرجع true;
}

ساكن مضمن bool htp_tensor_mdev_لیکې_فی_chunk(ثابت بنية htp_tensor * t, uint32_t elem_اندازه, uint32_t لیکه_اندازه, uint32_t * لیکې_فی_chunk) {
    *لیکې_فی_chunk = 0;

    إذا (!htp_tensor_mdev_معلومات_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_اندازه) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0)) {
        *لیکې_فی_chunk = 1;
        أرجع true;
    }
    إذا (t->nb[1] == لیکه_اندازه &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *لیکې_فی_chunk = (لیکه_اندازه > 0) ? (HTP_TENSOR_MDEV_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(لیکه_اندازه, HTP_TENSOR_MDEV_کرښه_اندازه)) : 1;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_برخه_کول(uint32_t ټول_units, uint32_t units_فی_chunk, uint32_t mdev_idx, uint32_t mdev_شمېر, ثابت بنية fastdiv_ارزښتونه * mdev_شمېر_div) {
    بنية htp_tensor_mdev_range range = { 0, ټول_units };

    إذا (mdev_شمېر <= 1) {
        أرجع range;
    }

    إذا (units_فی_chunk == 0) {
        range.پیل = (mdev_idx == 0) ? 0 : ټول_units;
        range.شمېر = (mdev_idx == 0) ? ټول_units : 0;
        أرجع range;
    }

    ثابت uint32_t ټول_chunks = ټول_units / units_فی_chunk;
    إذا (ټول_chunks < mdev_شمېر) {
        range.پیل = (mdev_idx == 0) ? 0 : ټول_units;
        range.شمېر = (mdev_idx == 0) ? ټول_units : 0;
        أرجع range;
    }

    ثابت uint32_t chunks_فی_mdev = fastdiv(ټول_chunks + mdev_شمېر - 1, mdev_شمېر_div);
    range.پیل = MIN(mdev_idx * chunks_فی_mdev * units_فی_chunk, ټول_units);
    إذا (mdev_idx == mdev_شمېر - 1) {
        range.شمېر = ټول_units - range.پیل;
    } وإلا {
        range.شمېر = MIN(chunks_فی_mdev * units_فی_chunk, ټول_units - range.پیل);
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

ساكن مضمن uint32_t htp_tensor_get_لیکه_اندازه(صحيح type, uint32_t ne00) {
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

#د_که_پای // HTP_TENSOR_H
