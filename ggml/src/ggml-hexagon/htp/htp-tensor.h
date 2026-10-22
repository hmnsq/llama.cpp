#నిర్వచితం_కాకపోతే HTP_TENSOR_H
#నిర్వచించు HTP_TENSOR_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "hex-bitmap.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-fastdiv.h"

#నిర్వచించు HTP_TENSOR_MDEV_పంక్తి_కొలత   128

// Currently HMX requires 32x32 tile formats (do not change)
#నిర్వచించు HTP_TENSOR_TILE_N_నిలువుల_సంఖ్య      32
#నిర్వచించు HTP_TENSOR_TILE_N_అడ్డువరుసలు      32

#నిర్వచించు HTP_TENSOR_TILE_కొలత_F16    2048
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q4_0   576
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q4_1   640
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q8_0   1088
#నిర్వచించు HTP_TENSOR_TILE_కొలత_IQ4_NL 576
#నిర్వచించు HTP_TENSOR_TILE_కొలత_MXFP4  544
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q4_K   640
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q5_K   768
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q6_K   896
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q3_K   512
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q2_K   336
#నిర్వచించు HTP_TENSOR_TILE_K_Q1_0      128
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q1_0   576
#నిర్వచించు HTP_TENSOR_TILE_K_Q2_0      64
#నిర్వచించు HTP_TENSOR_TILE_కొలత_Q2_0   576

// Tiled weight tensor layouts (Q4_0, Q8_0, IQ4_NL, MXFP4, Q1_0, Q2_0, K-quants):
// See docs/backend/snapdragon/developer.md ("Tiled Weight Formats").

నిర్మాణం htp_tensor_mdev_range {
    uint32_t ప్రారంభం;
    uint32_t లెక్క;
};

స్థిర అంతఃవరుస bool htp_tensor_is_extended(స్థిరరాశి నిర్మాణం htp_tensor * t) {
    తిరిగి t && (t->దత్తాంశం >> 32) != 0;
}

స్థిర అంతఃవరుస bool htp_tensor_is_contiguous(స్థిరరాశి నిర్మాణం htp_tensor * t, uint32_t type_కొలత) {
    uint32_t తదుపరి_nb = type_కొలత;
    ఒకవేళ (t->ne[0] != 1 && t->nb[0] != తదుపరి_nb) {
        తిరిగి false;
    }
    తదుపరి_nb *= t->ne[0];
    కొరకు (పూర్ణాంకం i = 1; i < HTP_OP_MAX_DIMS; i++) {
        ఒకవేళ (t->ne[i] != 1 && t->nb[i] != తదుపరి_nb) {
            తిరిగి false;
        }
        తదుపరి_nb *= t->ne[i];
    }
    తిరిగి true;
}

స్థిర అంతఃవరుస bool htp_tensor_is_permuted(స్థిరరాశి నిర్మాణం htp_tensor * t) {
    తిరిగి t->nb[0] > t->nb[1] || t->nb[1] > t->nb[2] || t->nb[2] > t->nb[3];
}

స్థిర అంతఃవరుస bool htp_tensor_mdev_దత్తాంశం_aligned(స్థిరరాశి నిర్మాణం htp_tensor * t) {
    తిరిగి ((uintptr_t) t->దత్తాంశం & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0;
}

స్థిర అంతఃవరుస bool htp_tensor_can_అడ్డువరుస_విభజన(స్థిరరాశి నిర్మాణం htp_tensor * t, uint32_t elem_కొలత) {
    ఒకవేళ (!htp_tensor_mdev_దత్తాంశం_aligned(t)) {
        తిరిగి false;
    }
    ఒకవేళ (t->ne[0] != 1 && t->nb[0] != elem_కొలత) {
        తిరిగి false;
    }
    ఒకవేళ (htp_tensor_is_permuted(t)) {
        తిరిగి false;
    }
    ఒకవేళ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) != 0) తిరిగి false;
    ఒకవేళ (t->ne[2] > 1 && (t->nb[2] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) != 0) తిరిగి false;
    ఒకవేళ (t->ne[3] > 1 && (t->nb[3] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) != 0) తిరిగి false;
    తిరిగి true;
}

స్థిర అంతఃవరుస bool htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(స్థిరరాశి నిర్మాణం htp_tensor * t, uint32_t elem_కొలత, uint32_t అడ్డువరుస_కొలత, uint32_t * అడ్డువరుసలు_చొప్పున_chunk) {
    *అడ్డువరుసలు_చొప్పున_chunk = 0;

    ఒకవేళ (!htp_tensor_mdev_దత్తాంశం_aligned(t)) {
        తిరిగి false;
    }
    ఒకవేళ (t->ne[0] != 1 && t->nb[0] != elem_కొలత) {
        తిరిగి false;
    }
    ఒకవేళ (htp_tensor_is_permuted(t)) {
        తిరిగి false;
    }
    ఒకవేళ (t->ne[1] > 1 && (t->nb[1] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0 &&
        (t->ne[2] <= 1 || (t->nb[2] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0) &&
        (t->ne[3] <= 1 || (t->nb[3] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0)) {
        *అడ్డువరుసలు_చొప్పున_chunk = 1;
        తిరిగి true;
    }
    ఒకవేళ (t->nb[1] == అడ్డువరుస_కొలత &&
        (t->ne[2] <= 1 || t->nb[2] == t->nb[1] * t->ne[1]) &&
        (t->ne[3] <= 1 || t->nb[3] == t->nb[2] * t->ne[2])) {
        *అడ్డువరుసలు_చొప్పున_chunk = (అడ్డువరుస_కొలత > 0) ? (HTP_TENSOR_MDEV_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(అడ్డువరుస_కొలత, HTP_TENSOR_MDEV_పంక్తి_కొలత)) : 1;
        తిరిగి true;
    }
    తిరిగి false;
}

స్థిర అంతఃవరుస నిర్మాణం htp_tensor_mdev_range htp_tensor_mdev_విభజన(uint32_t మొత్తం_units, uint32_t units_చొప్పున_chunk, uint32_t mdev_idx, uint32_t mdev_లెక్క, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * mdev_లెక్క_div) {
    నిర్మాణం htp_tensor_mdev_range range = { 0, మొత్తం_units };

    ఒకవేళ (mdev_లెక్క <= 1) {
        తిరిగి range;
    }

    ఒకవేళ (units_చొప్పున_chunk == 0) {
        range.ప్రారంభం = (mdev_idx == 0) ? 0 : మొత్తం_units;
        range.లెక్క = (mdev_idx == 0) ? మొత్తం_units : 0;
        తిరిగి range;
    }

    స్థిరరాశి uint32_t మొత్తం_chunks = మొత్తం_units / units_చొప్పున_chunk;
    ఒకవేళ (మొత్తం_chunks < mdev_లెక్క) {
        range.ప్రారంభం = (mdev_idx == 0) ? 0 : మొత్తం_units;
        range.లెక్క = (mdev_idx == 0) ? మొత్తం_units : 0;
        తిరిగి range;
    }

    స్థిరరాశి uint32_t chunks_చొప్పున_mdev = fastdiv(మొత్తం_chunks + mdev_లెక్క - 1, mdev_లెక్క_div);
    range.ప్రారంభం = MIN(mdev_idx * chunks_చొప్పున_mdev * units_చొప్పున_chunk, మొత్తం_units);
    ఒకవేళ (mdev_idx == mdev_లెక్క - 1) {
        range.లెక్క = మొత్తం_units - range.ప్రారంభం;
    } లేకపోతే {
        range.లెక్క = MIN(chunks_చొప్పున_mdev * units_చొప్పున_chunk, మొత్తం_units - range.ప్రారంభం);
    }
    తిరిగి range;
}

స్థిర అంతఃవరుస uint32_t htp_tensor_get_tile_కొలత(పూర్ణాంకం type) {
    ఎంపిక (type) {
        స్థితి HTP_TYPE_Q4_0:   తిరిగి HTP_TENSOR_TILE_కొలత_Q4_0;
        స్థితి HTP_TYPE_IQ4_NL: తిరిగి HTP_TENSOR_TILE_కొలత_IQ4_NL;
        స్థితి HTP_TYPE_Q4_1:   తిరిగి HTP_TENSOR_TILE_కొలత_Q4_1;
        స్థితి HTP_TYPE_Q4_K:   తిరిగి HTP_TENSOR_TILE_కొలత_Q4_K;
        స్థితి HTP_TYPE_Q8_0:   తిరిగి HTP_TENSOR_TILE_కొలత_Q8_0;
        స్థితి HTP_TYPE_Q5_K:   తిరిగి HTP_TENSOR_TILE_కొలత_Q5_K;
        స్థితి HTP_TYPE_Q6_K:   తిరిగి HTP_TENSOR_TILE_కొలత_Q6_K;
        స్థితి HTP_TYPE_Q3_K:   తిరిగి HTP_TENSOR_TILE_కొలత_Q3_K;
        స్థితి HTP_TYPE_Q2_K:   తిరిగి HTP_TENSOR_TILE_కొలత_Q2_K;
        స్థితి HTP_TYPE_Q1_0:   తిరిగి HTP_TENSOR_TILE_కొలత_Q1_0;
        స్థితి HTP_TYPE_Q2_0:   తిరిగి HTP_TENSOR_TILE_కొలత_Q2_0;
        స్థితి HTP_TYPE_MXFP4:  తిరిగి HTP_TENSOR_TILE_కొలత_MXFP4;
        స్థితి HTP_TYPE_F16:    తిరిగి HTP_TENSOR_TILE_కొలత_F16;
        అప్రమేయ:              తిరిగి 0;
    }
}

స్థిర అంతఃవరుస uint32_t htp_tensor_get_tile_k(పూర్ణాంకం type) {
    ఎంపిక (type) {
        స్థితి HTP_TYPE_Q1_0: తిరిగి HTP_TENSOR_TILE_K_Q1_0;
        స్థితి HTP_TYPE_Q2_0: తిరిగి HTP_TENSOR_TILE_K_Q2_0;
        అప్రమేయ:            తిరిగి 32;
    }
}

స్థిర అంతఃవరుస uint32_t htp_tensor_get_tile_padded_కొలత(పూర్ణాంకం type) {
    తిరిగి (htp_tensor_get_tile_కొలత(type) + 127u) & ~127u;
}

స్థిర అంతఃవరుస bool htp_tensor_is_tiled(స్థిరరాశి నిర్మాణం htp_tensor * t) {
    తిరిగి t && (t->flags & HTP_TENSOR_TILED) != 0;
}

స్థిర అంతఃవరుస uint32_t htp_tensor_get_అడ్డువరుస_కొలత(పూర్ణాంకం type, uint32_t ne00) {
    ఎంపిక (type) {
        స్థితి HTP_TYPE_F32:    తిరిగి ne00 * 4;
        స్థితి HTP_TYPE_F16:
        స్థితి HTP_TYPE_BF16:   తిరిగి ne00 * 2;
        స్థితి HTP_TYPE_Q8_0:   తిరిగి (ne00 / 32) * 34;
        స్థితి HTP_TYPE_Q4_0:
        స్థితి HTP_TYPE_IQ4_NL: తిరిగి (ne00 / 32) * 18;
        స్థితి HTP_TYPE_Q4_1:   తిరిగి (ne00 / 32) * 20;
        స్థితి HTP_TYPE_MXFP4:  తిరిగి (ne00 / 32) * 17;
        స్థితి HTP_TYPE_Q2_K:   తిరిగి (ne00 / 256) * 84;
        స్థితి HTP_TYPE_Q2_0:   తిరిగి (ne00 / 64) * 18;
        స్థితి HTP_TYPE_Q3_K:   తిరిగి (ne00 / 256) * 110;
        స్థితి HTP_TYPE_Q4_K:   తిరిగి (ne00 / 256) * 144;
        స్థితి HTP_TYPE_Q5_K:   తిరిగి (ne00 / 256) * 176;
        స్థితి HTP_TYPE_Q6_K:   తిరిగి (ne00 / 256) * 210;
        స్థితి HTP_TYPE_Q1_0:   తిరిగి (ne00 / HTP_TENSOR_TILE_K_Q1_0) * 18;
        స్థితి HTP_TYPE_I32:    తిరిగి ne00 * 4;
        స్థితి HTP_TYPE_I64:    తిరిగి ne00 * 8;
        అప్రమేయ:              తిరిగి 0;
    }
}

#అయితే_ముగింపు // HTP_TENSOR_H
