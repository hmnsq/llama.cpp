#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_REPACK_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_REPACK_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/htp-tensor.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_is_tiled_ਭਾਰ_type(ਗਿਣਤੀ ggml_type type) {
    ਵਾਪਸ type == GGML_TYPE_Q4_0  || type == GGML_TYPE_Q4_1   ||
           type == GGML_TYPE_Q8_0  || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q1_0   ||
           type == GGML_TYPE_Q6_K   ||
           type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q5_K   ||
           type == GGML_TYPE_Q3_K  || type == GGML_TYPE_Q2_K   ||
           type == GGML_TYPE_Q2_0  ||
           type == GGML_TYPE_F16   || type == GGML_TYPE_F32    ||
           type == GGML_TYPE_BF16;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_is_tiled_ਭਾਰ_tensor(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (!ggml_is_contiguous(t) || ggml_is_view(t)) {
        ਵਾਪਸ false;
    }
    ਜੇ (!htp_is_tiled_ਭਾਰ_type(t->type)) {
        ਵਾਪਸ false;
    }
    ਜੇ (t->ne[0] < 32) {
        ਵਾਪਸ false;
    }
    ਜੇ (!ggml_is_quantized(t->type)) {
        ਵਾਪਸ (t->ne[0] % 32 == 0) && (t->ne[1] % 32 == 0) && (t->ne[2] == 1) && (t->ne[3] == 1);
    }
    ਵਾਪਸ t->ne[3] == 1;
}

// Size of one tiled row in the DSP tiled layout. The Q3_K..Q6_K tiles store uncompressed scales/mins,
// so they are larger than the ggml blocks. For the other tiled types (Q2_K included) the tile has the same size.
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_tiled_ਪੰਕਤੀ_ਮਾਪ(ਗਿਣਤੀ ggml_type type, int64_t ne0) {
    ਜੇ (type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16) {
        ਵਾਪਸ (size_t) ne0 * ਆਕਾਰ(ggml_fp16_t);
    }
    ਜੇ (type == GGML_TYPE_Q6_K) {
        ਵਾਪਸ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ਮਾਪ_Q6_K / 32);
    }
    ਜੇ (type == GGML_TYPE_Q2_0) {
        ਵਾਪਸ (size_t) (ne0 / 64) * 18;
    }
    ਜੇ (type == GGML_TYPE_Q4_K) {
        ਵਾਪਸ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ਮਾਪ_Q4_K / 32);
    }
    ਜੇ (type == GGML_TYPE_Q5_K) {
        ਵਾਪਸ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ਮਾਪ_Q5_K / 32);
    }
    ਜੇ (type == GGML_TYPE_Q3_K) {
        ਵਾਪਸ (size_t) (ne0 / 32) * (HTP_TENSOR_TILE_ਮਾਪ_Q3_K / 32);
    }
    ਵਾਪਸ ggml_ਪੰਕਤੀ_ਮਾਪ(type, ne0);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_is_hmx_ਭਾਰ_type(ਗਿਣਤੀ ggml_type type) {
    ਵਾਪਸ htp_is_tiled_ਭਾਰ_type(type);
}

ਖਾਲੀ htp_repack_tensor_tiled(ਬਣਤਰ ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);
ਖਾਲੀ htp_unpack_tensor_tiled(ਖਾਲੀ * ਅੰਕੜੇ, ਅਚਲ ਬਣਤਰ ggml_tensor * t, size_t ਵਿੱਥ, size_t ਮਾਪ);

#ਜੇ_ਅੰਤ // HTP_REPACK_H
