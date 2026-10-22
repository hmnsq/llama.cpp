#నిర్వచితం_కాకపోతే HTP_REPACK_H
#నిర్వచించు HTP_REPACK_H

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "htp/htp-tensor.h"

స్థిర అంతఃవరుస bool htp_is_tiled_బరువు_type(గణన ggml_type type) {
    తిరిగి type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

స్థిర అంతఃవరుస bool htp_is_tiled_బరువు_tensor(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        తిరిగి false;
    }
    ఒకవేళ (!htp_is_tiled_బరువు_type(t->type)) {
        తిరిగి false;
    }
    ఒకవేళ (t->ne[0] < 32) {
        తిరిగి false;
    }
    ఒకవేళ (!ggml_is_quantized(t->type)) {
        తిరిగి (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    తిరిగి t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
స్థిర అంతఃవరుస size_t htp_tiled_అడ్డువరుస_కొలత(గణన ggml_type type, int64_t ne0) {
    ఒకవేళ (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        తిరిగి (size_t) ne0 * పరిమాణం(ggml_fp16_t);
    }
    ఒకవేళ (type == GGML_TYPE_Q6_K) {
        తిరిగి (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_కొలత_Q6_K / 32);
    }
    ఒకవేళ (type == GGML_TYPE_Q2_0) {
        తిరిగి (size_t) (ne0 / 64) * 18;
    }
    ఒకవేళ (type == GGML_TYPE_Q4_K) {
        తిరిగి (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_కొలత_Q4_K / 32);
    }
    ఒకవేళ (type == GGML_TYPE_Q5_K) {
        తిరిగి (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_కొలత_Q5_K / 32);
    }
    ఒకవేళ (type == GGML_TYPE_Q3_K) {
        తిరిగి (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_కొలత_Q3_K / 32);
    }
    తిరిగి ggml_అడ్డువరుస_కొలత(type, ne0);
}

స్థిర అంతఃవరుస bool htp_is_hmx_బరువు_type(గణన ggml_type type) {
    తిరిగి htp_is_tiled_బరువు_type(type);
}

శూన్య htp_repack_tensor_tiled(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t కొలత);
శూన్య htp_unpack_tensor_tiled(శూన్య * దత్తాంశం, స్థిరరాశి నిర్మాణం ggml_tensor * t, size_t స్థానభ్రంశం, size_t కొలత);

#అయితే_ముగింపు // HTP_REPACK_H
