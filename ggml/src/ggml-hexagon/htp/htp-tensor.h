#جيڪڏهن_وصف_نه_ٿيل HTP_TENSOR_H
#وصف_ڪريو HTP_TENSOR_H

#ضمن <stdint.h>
#ضمن <stdbool.h>
#ضمن "htp-ops.h"
#ضمن "hex-bitmap.h"
#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#وصف_ڪريو HTP_TENSOR_MDEV_سٽ_سائيز   128

// Currently HMX requires 32x32 tile formats (do not change)
#وصف_ڪريو HTP_TENSOR_TILE_N_کانن_ڳڻپ      32
#وصف_ڪريو HTP_TENSOR_TILE_N_صفون      32

#وصف_ڪريو HTP_TENSOR_TILE_سائيز_F16    2048
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q4_0   576
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q4_1   640
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q8_0   1088
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_IQ4_NL 576
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_MXFP4  544
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q4_K   640
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q5_K   768
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q6_K   896
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q3_K   512
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q2_K   336
#وصف_ڪريو HTP_TENSOR_TILE_K_Q1_0      128
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q1_0   576
#وصف_ڪريو HTP_TENSOR_TILE_K_Q2_0      64
#وصف_ڪريو HTP_TENSOR_TILE_سائيز_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

بنية htp_tensor_mdev_range {
    uint32_t شروعات;
    uint32_t ڳڻپ;
};

ساكن مضمن bool htp_tensor_is_extended(ثابت بنية htp_tensor * t) {
    أرجع t && (t->معلومات >> 32) != 0;
}

ساكن مضمن bool htp_tensor_is_contiguous(ثابت بنية htp_tensor * t, uint32_t type_سائيز) {
    uint32_t ايندڙ_nb = type_سائيز;
    إذا (t->ne[0] != 1 && t->nb[0] != ايندڙ_nb) {
        أرجع false;
    }
    ايندڙ_nb *= t->ne[0];
    لكل (صحيح i = 1; i < HTP_OP_MAX_DIMS; i++) {
        إذا (t->ne[i] != 1 && t->nb[i] != ايندڙ_nb) {
            أرجع false;
        }
        ايندڙ_nb *= t->ne[i];
    }
    أرجع true;
}

ساكن مضمن bool htp_tensor_is_permuted(ثابت بنية htp_tensor * t) {
    أرجع t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ساكن مضمن bool htp_tensor_mdev_معلومات_aligned(ثابت بنية htp_tensor * t) {
    أرجع ((uintptr_t) t->معلومات & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0;
}

ساكن مضمن bool htp_tensor_can_صف_حصا_ڪريو(ثابت بنية htp_tensor * t, uint32_t elem_سائيز) {
    إذا (!htp_tensor_mdev_معلومات_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_سائيز) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) != 0) أرجع false;
    إذا (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) != 0) أرجع false;
    إذا (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) != 0) أرجع false;
    أرجع true;
}

ساكن مضمن bool htp_tensor_mdev_صفون_في_chunk(ثابت بنية htp_tensor * t, uint32_t elem_سائيز, uint32_t صف_سائيز, uint32_t * صفون_في_chunk) {
    *صفون_في_chunk = 0;

    إذا (!htp_tensor_mdev_معلومات_aligned(t)) {
        أرجع false;
    }
    إذا (t->ne[0] != 1 && t->nb[0] != elem_سائيز) {
        أرجع false;
    }
    إذا (htp_tensor_is_permuted(t)) {
        أرجع false;
    }
    إذا (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0)) {
        *صفون_في_chunk = 1;
        أرجع true;
    }
    إذا (t->nb[1] == صف_سائيز &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *صفون_في_chunk = (صف_سائيز > 0) ? (HTP_TENSOR_MDEV_سٽ_سائيز / hex_وڏو_گڏيل_ونڊيندڙ_u32(صف_سائيز, HTP_TENSOR_MDEV_سٽ_سائيز)) : 1;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن بنية htp_tensor_mdev_range htp_tensor_mdev_حصا_ڪريو(uint32_t ڪل_units, uint32_t units_في_chunk, uint32_t mdev_idx, uint32_t mdev_ڳڻپ, ثابت بنية fastdiv_قدرون * mdev_ڳڻپ_div) {
    بنية htp_tensor_mdev_range range = { 0, ڪل_units };

    إذا (mdev_ڳڻپ <= 1) {
        أرجع range;
    }

    إذا (units_في_chunk == 0) {
        range.شروعات = (mdev_idx == 0) ? 0 : ڪل_units;
        range.ڳڻپ = (mdev_idx == 0) ? ڪل_units : 0;
        أرجع range;
    }

    ثابت uint32_t ڪل_chunks = ڪل_units / units_في_chunk;
    إذا (ڪل_chunks < mdev_ڳڻپ) {
        range.شروعات = (mdev_idx == 0) ? 0 : ڪل_units;
        range.ڳڻپ = (mdev_idx == 0) ? ڪل_units : 0;
        أرجع range;
    }

    ثابت uint32_t chunks_في_mdev = fastdiv(ڪل_chunks + mdev_ڳڻپ - 1, mdev_ڳڻپ_div);
    range.شروعات = MIN(mdev_idx * chunks_في_mdev * units_في_chunk, ڪل_units);
    إذا (mdev_idx == mdev_ڳڻپ - 1) {
        range.ڳڻپ = ڪل_units - range.شروعات;
    } وإلا {
        range.ڳڻپ = MIN(chunks_في_mdev * units_في_chunk, ڪل_units - range.شروعات);
    }
    أرجع range;
}

ساكن مضمن uint32_t htp_tensor_get_tile_سائيز(صحيح type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:   أرجع HTP_TENSOR_TILE_سائيز_Q4_0;
        حالة HTP_TYPE_IQ4_NL: أرجع HTP_TENSOR_TILE_سائيز_IQ4_NL;
        حالة HTP_TYPE_Q4_1:   أرجع HTP_TENSOR_TILE_سائيز_Q4_1;
        حالة HTP_TYPE_Q4_K:   أرجع HTP_TENSOR_TILE_سائيز_Q4_K;
        حالة HTP_TYPE_Q8_0:   أرجع HTP_TENSOR_TILE_سائيز_Q8_0;
        حالة HTP_TYPE_Q5_K:   أرجع HTP_TENSOR_TILE_سائيز_Q5_K;
        حالة HTP_TYPE_Q6_K:   أرجع HTP_TENSOR_TILE_سائيز_Q6_K;
        حالة HTP_TYPE_Q3_K:   أرجع HTP_TENSOR_TILE_سائيز_Q3_K;
        حالة HTP_TYPE_Q2_K:   أرجع HTP_TENSOR_TILE_سائيز_Q2_K;
        حالة HTP_TYPE_Q1_0:   أرجع HTP_TENSOR_TILE_سائيز_Q1_0;
        حالة HTP_TYPE_Q2_0:   أرجع HTP_TENSOR_TILE_سائيز_Q2_0;
        حالة HTP_TYPE_MXFP4:  أرجع HTP_TENSOR_TILE_سائيز_MXFP4;
        حالة HTP_TYPE_F16:    أرجع HTP_TENSOR_TILE_سائيز_F16;
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

ساكن مضمن uint32_t htp_tensor_get_tile_padded_سائيز(صحيح type) {
    أرجع (htp_tensor_get_tile_سائيز(type) + 127u) & ~127u;
}

ساكن مضمن bool htp_tensor_is_tiled(ثابت بنية htp_tensor * t) {
    أرجع t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ساكن مضمن uint32_t htp_tensor_get_صف_سائيز(صحيح type, uint32_t ne00) {
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

#جيڪڏهن_پورو // HTP_TENSOR_H
