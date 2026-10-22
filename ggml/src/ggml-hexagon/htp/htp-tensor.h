#अगर_अघोषित HTP_TENSOR_H
#घोषणा HTP_TENSOR_H

#समावेश <stdint.h>
#समावेश <stdbool.h>
#समावेश "htp-ops.h"
#समावेश "hex-bitmap.h"
#समावेश "hex-common.h"
#समावेश "hex-fastdiv.h"

#घोषणा HTP_TENSOR_MDEV_पंक्ति_आकार   128

// Currently HMX requires 32x32 tile formats (do not change)
#घोषणा HTP_TENSOR_TILE_N_COLS      32
#घोषणा HTP_TENSOR_TILE_N_ROWS      32

#घोषणा HTP_TENSOR_TILE_आकार_F16    2048
#घोषणा HTP_TENSOR_TILE_आकार_Q4_0   576
#घोषणा HTP_TENSOR_TILE_आकार_Q4_1   640
#घोषणा HTP_TENSOR_TILE_आकार_Q8_0   1088
#घोषणा HTP_TENSOR_TILE_आकार_IQ4_NL 576
#घोषणा HTP_TENSOR_TILE_आकार_MXFP4  544
#घोषणा HTP_TENSOR_TILE_आकार_Q4_K   640
#घोषणा HTP_TENSOR_TILE_आकार_Q5_K   768
#घोषणा HTP_TENSOR_TILE_आकार_Q6_K   896
#घोषणा HTP_TENSOR_TILE_आकार_Q3_K   512
#घोषणा HTP_TENSOR_TILE_आकार_Q2_K   336
#घोषणा HTP_TENSOR_TILE_K_Q1_0      128
#घोषणा HTP_TENSOR_TILE_आकार_Q1_0   576
#घोषणा HTP_TENSOR_TILE_K_Q2_0      64
#घोषणा HTP_TENSOR_TILE_आकार_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

काष्ठा htp_tensor_mdev_range {
    uint32_t start;
    uint32_t गिनती;
};

अटल अंतरभूत bool htp_tensor_is_extended(स्थिर काष्ठा htp_tensor * t) {
    वापस t && (t->तथ्य >> 32) != 0;
}

अटल अंतरभूत bool htp_tensor_is_contiguous(स्थिर काष्ठा htp_tensor * t, uint32_t type_आकार) {
    uint32_t अग्र_nb = type_आकार;
    अगर (t->ne[0] != 1 && t->nb[0] != अग्र_nb) {
        वापस false;
    }
    अग्र_nb *= t->ne[0];
    क्रम (पूर्णांक i = 1; i < HTP_OP_MAX_DIMS; i++) {
        अगर (t->ne[i] != 1 && t->nb[i] != अग्र_nb) {
            वापस false;
        }
        अग्र_nb *= t->ne[i];
    }
    वापस true;
}

अटल अंतरभूत bool htp_tensor_is_permuted(स्थिर काष्ठा htp_tensor * t) {
    वापस t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

अटल अंतरभूत bool htp_tensor_mdev_तथ्य_aligned(स्थिर काष्ठा htp_tensor * t) {
    वापस ((uintptr_t) t->तथ्य & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0;
}

अटल अंतरभूत bool htp_tensor_can_row_partition(स्थिर काष्ठा htp_tensor * t, uint32_t elem_आकार) {
    अगर (!htp_tensor_mdev_तथ्य_aligned(t)) {
        वापस false;
    }
    अगर (t->ne[0] != 1 && t->nb[0] != elem_आकार) {
        वापस false;
    }
    अगर (htp_tensor_is_permuted(t)) {
        वापस false;
    }
    अगर (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) != 0) वापस false;
    अगर (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) != 0) वापस false;
    अगर (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) != 0) वापस false;
    वापस true;
}

अटल अंतरभूत bool htp_tensor_mdev_rows_per_chunk(स्थिर काष्ठा htp_tensor * t, uint32_t elem_आकार, uint32_t row_आकार, uint32_t * rows_per_chunk) {
    *rows_per_chunk = 0;

    अगर (!htp_tensor_mdev_तथ्य_aligned(t)) {
        वापस false;
    }
    अगर (t->ne[0] != 1 && t->nb[0] != elem_आकार) {
        वापस false;
    }
    अगर (htp_tensor_is_permuted(t)) {
        वापस false;
    }
    अगर (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0)) {
        *rows_per_chunk = 1;
        वापस true;
    }
    अगर (t->nb[1] == row_आकार &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *rows_per_chunk = (row_आकार > 0) ? (HTP_TENSOR_MDEV_पंक्ति_आकार / hex_gcd_u32(row_आकार, HTP_TENSOR_MDEV_पंक्ति_आकार)) : 1;
        वापस true;
    }
    वापस false;
}

अटल अंतरभूत काष्ठा htp_tensor_mdev_range htp_tensor_mdev_partition(uint32_t योग_units, uint32_t units_per_chunk, uint32_t mdev_idx, uint32_t mdev_गिनती, स्थिर काष्ठा fastdiv_मूल्यावली * mdev_गिनती_div) {
    काष्ठा htp_tensor_mdev_range range = { 0, योग_units };

    अगर (mdev_गिनती <= 1) {
        वापस range;
    }

    अगर (units_per_chunk == 0) {
        range.start = (mdev_idx == 0) ? 0 : योग_units;
        range.गिनती = (mdev_idx == 0) ? योग_units : 0;
        वापस range;
    }

    स्थिर uint32_t योग_chunks = योग_units / units_per_chunk;
    अगर (योग_chunks < mdev_गिनती) {
        range.start = (mdev_idx == 0) ? 0 : योग_units;
        range.गिनती = (mdev_idx == 0) ? योग_units : 0;
        वापस range;
    }

    स्थिर uint32_t chunks_per_mdev = fastdiv(योग_chunks + mdev_गिनती - 1, mdev_गिनती_div);
    range.start = MIN(mdev_idx * chunks_per_mdev * units_per_chunk, योग_units);
    अगर (mdev_idx == mdev_गिनती - 1) {
        range.गिनती = योग_units - range.start;
    } अन्यथा {
        range.गिनती = MIN(chunks_per_mdev * units_per_chunk, योग_units - range.start);
    }
    वापस range;
}

अटल अंतरभूत uint32_t htp_tensor_get_tile_आकार(पूर्णांक type) {
    चयन (type) {
        हाल HTP_TYPE_Q4_0:   वापस HTP_TENSOR_TILE_आकार_Q4_0;
        हाल HTP_TYPE_IQ4_NL: वापस HTP_TENSOR_TILE_आकार_IQ4_NL;
        हाल HTP_TYPE_Q4_1:   वापस HTP_TENSOR_TILE_आकार_Q4_1;
        हाल HTP_TYPE_Q4_K:   वापस HTP_TENSOR_TILE_आकार_Q4_K;
        हाल HTP_TYPE_Q8_0:   वापस HTP_TENSOR_TILE_आकार_Q8_0;
        हाल HTP_TYPE_Q5_K:   वापस HTP_TENSOR_TILE_आकार_Q5_K;
        हाल HTP_TYPE_Q6_K:   वापस HTP_TENSOR_TILE_आकार_Q6_K;
        हाल HTP_TYPE_Q3_K:   वापस HTP_TENSOR_TILE_आकार_Q3_K;
        हाल HTP_TYPE_Q2_K:   वापस HTP_TENSOR_TILE_आकार_Q2_K;
        हाल HTP_TYPE_Q1_0:   वापस HTP_TENSOR_TILE_आकार_Q1_0;
        हाल HTP_TYPE_Q2_0:   वापस HTP_TENSOR_TILE_आकार_Q2_0;
        हाल HTP_TYPE_MXFP4:  वापस HTP_TENSOR_TILE_आकार_MXFP4;
        हाल HTP_TYPE_F16:    वापस HTP_TENSOR_TILE_आकार_F16;
        शेष:              वापस 0;
    }
}

अटल अंतरभूत uint32_t htp_tensor_get_tile_k(पूर्णांक type) {
    चयन (type) {
        हाल HTP_TYPE_Q1_0: वापस HTP_TENSOR_TILE_K_Q1_0;
        हाल HTP_TYPE_Q2_0: वापस HTP_TENSOR_TILE_K_Q2_0;
        शेष:            वापस 32;
    }
}

अटल अंतरभूत uint32_t htp_tensor_get_tile_padded_आकार(पूर्णांक type) {
    वापस (htp_tensor_get_tile_आकार(type) + 127u) & ~127u;
}

अटल अंतरभूत bool htp_tensor_is_tiled(स्थिर काष्ठा htp_tensor * t) {
    वापस t && (t->flags & HTP_TENSOR_TILED) != 0;
}

अटल अंतरभूत uint32_t htp_tensor_get_row_आकार(पूर्णांक type, uint32_t ne00) {
    चयन (type) {
        हाल HTP_TYPE_F32:    वापस ne00 * 4;
        हाल HTP_TYPE_F16:
        हाल HTP_TYPE_BF16:   वापस ne00 * 2;
        हाल HTP_TYPE_Q8_0:   वापस (ne00 / 32) * 34;
        हाल HTP_TYPE_Q4_0:
        हाल HTP_TYPE_IQ4_NL: वापस (ne00 / 32) * 18;
        हाल HTP_TYPE_Q4_1:   वापस (ne00 / 32) * 20;
        हाल HTP_TYPE_MXFP4:  वापस (ne00 / 32) * 17;
        हाल HTP_TYPE_Q2_K:   वापस (ne00 / 256) * 84;
        हाल HTP_TYPE_Q2_0:   वापस (ne00 / 64) * 18;
        हाल HTP_TYPE_Q3_K:   वापस (ne00 / 256) * 110;
        हाल HTP_TYPE_Q4_K:   वापस (ne00 / 256) * 144;
        हाल HTP_TYPE_Q5_K:   वापस (ne00 / 256) * 176;
        हाल HTP_TYPE_Q6_K:   वापस (ne00 / 256) * 210;
        हाल HTP_TYPE_Q1_0:   वापस (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        हाल HTP_TYPE_I32:    वापस ne00 * 4;
        हाल HTP_TYPE_I64:    वापस ne00 * 8;
        शेष:              वापस 0;
    }
}

#पूर्ण_अगर // HTP_TENSOR_H
