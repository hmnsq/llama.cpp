#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_TENSOR_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-bitmap.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ   128

// Currently HMX requires 32x32 tile formats (do not change)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ      32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_N_ਪੰਕਤੀਆਂ      32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_F16    2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q4_0   576
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q4_1   640
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q8_0   1088
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_IQ4_NL 576
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_MXFP4  544
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q4_K   640
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q5_K   768
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q6_K   896
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q3_K   512
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q2_K   336
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_K_Q1_0      128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q1_0   576
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_K_Q2_0      64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TENSOR_TILE_ਮਾਪ_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

ਬਣਤਰ htp_tensor_mdev_range {
    uint32_t ਸ਼ੁਰੂ;
    uint32_t ਗਿਣਤੀ_2;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_is_extended(ਅਚਲ ਬਣਤਰ htp_tensor * t) {
    ਵਾਪਸ t && (t->ਅੰਕੜੇ >> 32) != 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_is_contiguous(ਅਚਲ ਬਣਤਰ htp_tensor * t, uint32_t type_ਮਾਪ) {
    uint32_t ਅਗਲਾ_nb = type_ਮਾਪ;
    ਜੇ (t->ne[0] != 1 && t->nb[0] != ਅਗਲਾ_nb) {
        ਵਾਪਸ false;
    }
    ਅਗਲਾ_nb *= t->ne[0];
    ਲਈ (ਪੂਰਨ i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ਜੇ (t->ne[i] != 1 && t->nb[i] != ਅਗਲਾ_nb) {
            ਵਾਪਸ false;
        }
        ਅਗਲਾ_nb *= t->ne[i];
    }
    ਵਾਪਸ true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_is_permuted(ਅਚਲ ਬਣਤਰ htp_tensor * t) {
    ਵਾਪਸ t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_mdev_ਅੰਕੜੇ_aligned(ਅਚਲ ਬਣਤਰ htp_tensor * t) {
    ਵਾਪਸ ((uintptr_t) t->ਅੰਕੜੇ & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(ਅਚਲ ਬਣਤਰ htp_tensor * t, uint32_t elem_ਮਾਪ) {
    ਜੇ (!htp_tensor_mdev_ਅੰਕੜੇ_aligned(t)) {
        ਵਾਪਸ false;
    }
    ਜੇ (t->ne[0] != 1 && t->nb[0] != elem_ਮਾਪ) {
        ਵਾਪਸ false;
    }
    ਜੇ (htp_tensor_is_permuted(t)) {
        ਵਾਪਸ false;
    }
    ਜੇ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) != 0) ਵਾਪਸ false;
    ਜੇ (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) != 0) ਵਾਪਸ false;
    ਜੇ (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) != 0) ਵਾਪਸ false;
    ਵਾਪਸ true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(ਅਚਲ ਬਣਤਰ htp_tensor * t, uint32_t elem_ਮਾਪ, uint32_t ਪੰਕਤੀ_ਮਾਪ, uint32_t * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk) {
    *ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;

    ਜੇ (!htp_tensor_mdev_ਅੰਕੜੇ_aligned(t)) {
        ਵਾਪਸ false;
    }
    ਜੇ (t->ne[0] != 1 && t->nb[0] != elem_ਮਾਪ) {
        ਵਾਪਸ false;
    }
    ਜੇ (htp_tensor_is_permuted(t)) {
        ਵਾਪਸ false;
    }
    ਜੇ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0)) {
        *ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 1;
        ਵਾਪਸ true;
    }
    ਜੇ (t->nb[1] == ਪੰਕਤੀ_ਮਾਪ &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = (ਪੰਕਤੀ_ਮਾਪ > 0) ? (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(ਪੰਕਤੀ_ਮਾਪ, HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ)) : 1;
        ਵਾਪਸ true;
    }
    ਵਾਪਸ false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ htp_tensor_mdev_range htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(uint32_t ਕੁੱਲ_units, uint32_t units_ਪ੍ਰਤੀ_chunk, uint32_t mdev_idx, uint32_t mdev_ਗਿਣਤੀ, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * mdev_ਗਿਣਤੀ_div) {
    ਬਣਤਰ htp_tensor_mdev_range range = { 0, ਕੁੱਲ_units };

    ਜੇ (mdev_ਗਿਣਤੀ <= 1) {
        ਵਾਪਸ range;
    }

    ਜੇ (units_ਪ੍ਰਤੀ_chunk == 0) {
        range.ਸ਼ੁਰੂ = (mdev_idx == 0) ? 0 : ਕੁੱਲ_units;
        range.ਗਿਣਤੀ_2 = (mdev_idx == 0) ? ਕੁੱਲ_units : 0;
        ਵਾਪਸ range;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_chunks = ਕੁੱਲ_units / units_ਪ੍ਰਤੀ_chunk;
    ਜੇ (ਕੁੱਲ_chunks < mdev_ਗਿਣਤੀ) {
        range.ਸ਼ੁਰੂ = (mdev_idx == 0) ? 0 : ਕੁੱਲ_units;
        range.ਗਿਣਤੀ_2 = (mdev_idx == 0) ? ਕੁੱਲ_units : 0;
        ਵਾਪਸ range;
    }

    ਅਚਲ uint32_t chunks_ਪ੍ਰਤੀ_mdev = fastdiv(ਕੁੱਲ_chunks + mdev_ਗਿਣਤੀ - 1, mdev_ਗਿਣਤੀ_div);
    range.ਸ਼ੁਰੂ = MIN(mdev_idx * chunks_ਪ੍ਰਤੀ_mdev * units_ਪ੍ਰਤੀ_chunk, ਕੁੱਲ_units);
    ਜੇ (mdev_idx == mdev_ਗਿਣਤੀ - 1) {
        range.ਗਿਣਤੀ_2 = ਕੁੱਲ_units - range.ਸ਼ੁਰੂ;
    } ਨਹੀਂਤਾਂ {
        range.ਗਿਣਤੀ_2 = MIN(chunks_ਪ੍ਰਤੀ_mdev * units_ਪ੍ਰਤੀ_chunk, ਕੁੱਲ_units - range.ਸ਼ੁਰੂ);
    }
    ਵਾਪਸ range;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t htp_tensor_get_tile_ਮਾਪ(ਪੂਰਨ type) {
    ਚੋਣ (type) {
        ਹਾਲਤ HTP_TYPE_Q4_0:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q4_0;
        ਹਾਲਤ HTP_TYPE_IQ4_NL: ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_IQ4_NL;
        ਹਾਲਤ HTP_TYPE_Q4_1:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q4_1;
        ਹਾਲਤ HTP_TYPE_Q4_K:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q4_K;
        ਹਾਲਤ HTP_TYPE_Q8_0:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q8_0;
        ਹਾਲਤ HTP_TYPE_Q5_K:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q5_K;
        ਹਾਲਤ HTP_TYPE_Q6_K:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q6_K;
        ਹਾਲਤ HTP_TYPE_Q3_K:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q3_K;
        ਹਾਲਤ HTP_TYPE_Q2_K:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q2_K;
        ਹਾਲਤ HTP_TYPE_Q1_0:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q1_0;
        ਹਾਲਤ HTP_TYPE_Q2_0:   ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_Q2_0;
        ਹਾਲਤ HTP_TYPE_MXFP4:  ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_MXFP4;
        ਹਾਲਤ HTP_TYPE_F16:    ਵਾਪਸ HTP_TENSOR_TILE_ਮਾਪ_F16;
        ਮੂਲ:              ਵਾਪਸ 0;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t htp_tensor_get_tile_k(ਪੂਰਨ type) {
    ਚੋਣ (type) {
        ਹਾਲਤ HTP_TYPE_Q1_0: ਵਾਪਸ HTP_TENSOR_TILE_K_Q1_0;
        ਹਾਲਤ HTP_TYPE_Q2_0: ਵਾਪਸ HTP_TENSOR_TILE_K_Q2_0;
        ਮੂਲ:            ਵਾਪਸ 32;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t htp_tensor_get_tile_padded_ਮਾਪ(ਪੂਰਨ type) {
    ਵਾਪਸ (htp_tensor_get_tile_ਮਾਪ(type) + 127u) & ~127u;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_tensor_is_tiled(ਅਚਲ ਬਣਤਰ htp_tensor * t) {
    ਵਾਪਸ t && (t->flags & HTP_TENSOR_TILED) != 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(ਪੂਰਨ type, uint32_t ne00) {
    ਚੋਣ (type) {
        ਹਾਲਤ HTP_TYPE_F32:    ਵਾਪਸ ne00 * 4;
        ਹਾਲਤ HTP_TYPE_F16:
        ਹਾਲਤ HTP_TYPE_BF16:   ਵਾਪਸ ne00 * 2;
        ਹਾਲਤ HTP_TYPE_Q8_0:   ਵਾਪਸ (ne00 / 32) * 34;
        ਹਾਲਤ HTP_TYPE_Q4_0:
        ਹਾਲਤ HTP_TYPE_IQ4_NL: ਵਾਪਸ (ne00 / 32) * 18;
        ਹਾਲਤ HTP_TYPE_Q4_1:   ਵਾਪਸ (ne00 / 32) * 20;
        ਹਾਲਤ HTP_TYPE_MXFP4:  ਵਾਪਸ (ne00 / 32) * 17;
        ਹਾਲਤ HTP_TYPE_Q2_K:   ਵਾਪਸ (ne00 / 256) * 84;
        ਹਾਲਤ HTP_TYPE_Q2_0:   ਵਾਪਸ (ne00 / 64) * 18;
        ਹਾਲਤ HTP_TYPE_Q3_K:   ਵਾਪਸ (ne00 / 256) * 110;
        ਹਾਲਤ HTP_TYPE_Q4_K:   ਵਾਪਸ (ne00 / 256) * 144;
        ਹਾਲਤ HTP_TYPE_Q5_K:   ਵਾਪਸ (ne00 / 256) * 176;
        ਹਾਲਤ HTP_TYPE_Q6_K:   ਵਾਪਸ (ne00 / 256) * 210;
        ਹਾਲਤ HTP_TYPE_Q1_0:   ਵਾਪਸ (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        ਹਾਲਤ HTP_TYPE_I32:    ਵਾਪਸ ne00 * 4;
        ਹਾਲਤ HTP_TYPE_I64:    ਵਾਪਸ ne00 * 8;
        ਮੂਲ:              ਵਾਪਸ 0;
    }
}

#ਜੇ_ਅੰਤ // HTP_TENSOR_H
