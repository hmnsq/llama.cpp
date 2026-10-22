#यदि_अपरिभाषितम् HTP_TENSOR_H
#परिभाषय HTP_TENSOR_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "hex-bitmap.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-fastdiv.h"

#परिभाषय HTP_TENSOR_MDEV_पङ्क्तिः_आकार   128

// Currently HMX requires 32x32 tile formats (do not change)
#परिभाषय HTP_TENSOR_TILE_N_स्तम्भसङ्ख्या      32
#परिभाषय HTP_TENSOR_TILE_N_पङ्क्तिकाः      32

#परिभाषय HTP_TENSOR_TILE_आकार_F16    2048
#परिभाषय HTP_TENSOR_TILE_आकार_Q4_0   576
#परिभाषय HTP_TENSOR_TILE_आकार_Q4_1   640
#परिभाषय HTP_TENSOR_TILE_आकार_Q8_0   1088
#परिभाषय HTP_TENSOR_TILE_आकार_IQ4_NL 576
#परिभाषय HTP_TENSOR_TILE_आकार_MXFP4  544
#परिभाषय HTP_TENSOR_TILE_आकार_Q4_K   640
#परिभाषय HTP_TENSOR_TILE_आकार_Q5_K   768
#परिभाषय HTP_TENSOR_TILE_आकार_Q6_K   896
#परिभाषय HTP_TENSOR_TILE_आकार_Q3_K   512
#परिभाषय HTP_TENSOR_TILE_आकार_Q2_K   336
#परिभाषय HTP_TENSOR_TILE_K_Q1_0      128
#परिभाषय HTP_TENSOR_TILE_आकार_Q1_0   576
#परिभाषय HTP_TENSOR_TILE_K_Q2_0      64
#परिभाषय HTP_TENSOR_TILE_आकार_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

संरचना htp_tensor_mdev_range {
    uint32_t start;
    uint32_t गणनम्;
};

स्थिर अन्तःस्थ bool htp_tensor_is_extended(ध्रुव संरचना htp_tensor * t) {
    प्रत्यागम t && (t->दत्त >> 32) != 0;
}

स्थिर अन्तःस्थ bool htp_tensor_is_contiguous(ध्रुव संरचना htp_tensor * t, uint32_t type_आकार) {
    uint32_t अग्र_nb = type_आकार;
    यदि (t->ne[0] != 1 && t->nb[0] != अग्र_nb) {
        प्रत्यागम false;
    }
    अग्र_nb *= t->ne[0];
    प्रति (पूर्णाङ्क i = 1; i < HTP_OP_MAX_DIMS; i++) {
        यदि (t->ne[i] != 1 && t->nb[i] != अग्र_nb) {
            प्रत्यागम false;
        }
        अग्र_nb *= t->ne[i];
    }
    प्रत्यागम true;
}

स्थिर अन्तःस्थ bool htp_tensor_is_permuted(ध्रुव संरचना htp_tensor * t) {
    प्रत्यागम t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

स्थिर अन्तःस्थ bool htp_tensor_mdev_दत्त_aligned(ध्रुव संरचना htp_tensor * t) {
    प्रत्यागम ((uintptr_t) t->दत्त & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0;
}

स्थिर अन्तःस्थ bool htp_tensor_can_row_partition(ध्रुव संरचना htp_tensor * t, uint32_t elem_आकार) {
    यदि (!htp_tensor_mdev_दत्त_aligned(t)) {
        प्रत्यागम false;
    }
    यदि (t->ne[0] != 1 && t->nb[0] != elem_आकार) {
        प्रत्यागम false;
    }
    यदि (htp_tensor_is_permuted(t)) {
        प्रत्यागम false;
    }
    यदि (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) != 0) प्रत्यागम false;
    यदि (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) != 0) प्रत्यागम false;
    यदि (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) != 0) प्रत्यागम false;
    प्रत्यागम true;
}

स्थिर अन्तःस्थ bool htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(ध्रुव संरचना htp_tensor * t, uint32_t elem_आकार, uint32_t row_आकार, uint32_t * पङ्क्तिकाः_प्रत्येकम्_chunk) {
    *पङ्क्तिकाः_प्रत्येकम्_chunk = 0;

    यदि (!htp_tensor_mdev_दत्त_aligned(t)) {
        प्रत्यागम false;
    }
    यदि (t->ne[0] != 1 && t->nb[0] != elem_आकार) {
        प्रत्यागम false;
    }
    यदि (htp_tensor_is_permuted(t)) {
        प्रत्यागम false;
    }
    यदि (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0)) {
        *पङ्क्तिकाः_प्रत्येकम्_chunk = 1;
        प्रत्यागम true;
    }
    यदि (t->nb[1] == row_आकार &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *पङ्क्तिकाः_प्रत्येकम्_chunk = (row_आकार > 0) ? (HTP_TENSOR_MDEV_पङ्क्तिः_आकार / hex_gcd_u32(row_आकार, HTP_TENSOR_MDEV_पङ्क्तिः_आकार)) : 1;
        प्रत्यागम true;
    }
    प्रत्यागम false;
}

स्थिर अन्तःस्थ संरचना htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t योग_units, uint32_t units_प्रत्येकम्_chunk, uint32_t mdev_idx, uint32_t mdev_गणनम्, ध्रुव संरचना fastdiv_मानानि * mdev_गणनम्_div) {
    संरचना htp_tensor_mdev_range range = { 0, योग_units };

    यदि (mdev_गणनम् <= 1) {
        प्रत्यागम range;
    }

    यदि (units_प्रत्येकम्_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : योग_units;
        range.गणनम् = (mdev_idx == 0) ? योग_units : 0;
        प्रत्यागम range;
    }

    ध्रुव uint32_t योग_chunks = योग_units / units_प्रत्येकम्_chunk;
    यदि (योग_chunks < mdev_गणनम्) {
        range.start = (mdev_idx == 0) ? 0 : योग_units;
        range.गणनम् = (mdev_idx == 0) ? योग_units : 0;
        प्रत्यागम range;
    }

    ध्रुव uint32_t chunks_प्रत्येकम्_mdev = fastdiv(योग_chunks + mdev_गणनम् - 1, mdev_गणनम्_div);
    range.start = MIN(mdev_idx * chunks_प्रत्येकम्_mdev * units_प्रत्येकम्_chunk, योग_units);
    यदि (mdev_idx == mdev_गणनम् - 1) {
        range.गणनम् = योग_units - range.start;
    } अन्यथा {
        range.गणनम् = MIN(chunks_प्रत्येकम्_mdev * units_प्रत्येकम्_chunk, योग_units - range.start);
    }
    प्रत्यागम range;
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_tile_आकार(पूर्णाङ्क type) {
    विकल्प (type) {
        पक्ष HTP_TYPE_Q4_0:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q4_0;
        पक्ष HTP_TYPE_IQ4_NL: प्रत्यागम HTP_TENSOR_TILE_आकार_IQ4_NL;
        पक्ष HTP_TYPE_Q4_1:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q4_1;
        पक्ष HTP_TYPE_Q4_K:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q4_K;
        पक्ष HTP_TYPE_Q8_0:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q8_0;
        पक्ष HTP_TYPE_Q5_K:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q5_K;
        पक्ष HTP_TYPE_Q6_K:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q6_K;
        पक्ष HTP_TYPE_Q3_K:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q3_K;
        पक्ष HTP_TYPE_Q2_K:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q2_K;
        पक्ष HTP_TYPE_Q1_0:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q1_0;
        पक्ष HTP_TYPE_Q2_0:   प्रत्यागम HTP_TENSOR_TILE_आकार_Q2_0;
        पक्ष HTP_TYPE_MXFP4:  प्रत्यागम HTP_TENSOR_TILE_आकार_MXFP4;
        पक्ष HTP_TYPE_F16:    प्रत्यागम HTP_TENSOR_TILE_आकार_F16;
        स्वभाव:              प्रत्यागम 0;
    }
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_tile_k(पूर्णाङ्क type) {
    विकल्प (type) {
        पक्ष HTP_TYPE_Q1_0: प्रत्यागम HTP_TENSOR_TILE_K_Q1_0;
        पक्ष HTP_TYPE_Q2_0: प्रत्यागम HTP_TENSOR_TILE_K_Q2_0;
        स्वभाव:            प्रत्यागम 32;
    }
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_tile_padded_आकार(पूर्णाङ्क type) {
    प्रत्यागम (htp_tensor_get_tile_आकार(type) + 127u) & ~127u;
}

स्थिर अन्तःस्थ bool htp_tensor_is_tiled(ध्रुव संरचना htp_tensor * t) {
    प्रत्यागम t && (t->flags & HTP_TENSOR_TILED) != 0;
}

स्थिर अन्तःस्थ uint32_t htp_tensor_get_row_आकार(पूर्णाङ्क type, uint32_t ne00) {
    विकल्प (type) {
        पक्ष HTP_TYPE_F32:    प्रत्यागम ne00 * 4;
        पक्ष HTP_TYPE_F16:
        पक्ष HTP_TYPE_BF16:   प्रत्यागम ne00 * 2;
        पक्ष HTP_TYPE_Q8_0:   प्रत्यागम (ne00 / 32) * 34;
        पक्ष HTP_TYPE_Q4_0:
        पक्ष HTP_TYPE_IQ4_NL: प्रत्यागम (ne00 / 32) * 18;
        पक्ष HTP_TYPE_Q4_1:   प्रत्यागम (ne00 / 32) * 20;
        पक्ष HTP_TYPE_MXFP4:  प्रत्यागम (ne00 / 32) * 17;
        पक्ष HTP_TYPE_Q2_K:   प्रत्यागम (ne00 / 256) * 84;
        पक्ष HTP_TYPE_Q2_0:   प्रत्यागम (ne00 / 64) * 18;
        पक्ष HTP_TYPE_Q3_K:   प्रत्यागम (ne00 / 256) * 110;
        पक्ष HTP_TYPE_Q4_K:   प्रत्यागम (ne00 / 256) * 144;
        पक्ष HTP_TYPE_Q5_K:   प्रत्यागम (ne00 / 256) * 176;
        पक्ष HTP_TYPE_Q6_K:   प्रत्यागम (ne00 / 256) * 210;
        पक्ष HTP_TYPE_Q1_0:   प्रत्यागम (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        पक्ष HTP_TYPE_I32:    प्रत्यागम ne00 * 4;
        पक्ष HTP_TYPE_I64:    प्रत्यागम ne00 * 8;
        स्वभाव:              प्रत्यागम 0;
    }
}

#यदि_अन्तः // HTP_TENSOR_H
