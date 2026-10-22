#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstring>

#ವ್ಯಾಖ್ಯಾನಿಸು likely(x)   __builtin_expect(!!(x), 1)
#ವ್ಯಾಖ್ಯಾನಿಸು unlikely(x) __builtin_expect(!!(x), 0)

ರಚನೆ apir_encoder {
    ಅಕ್ಷರ *       cur;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೊನೆ;
    bool         fatal;
};

ರಚನೆ apir_decoder {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cur;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಕೊನೆ;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ಸ್ಥಿರ apir_decoder apir_new_decoder(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ptr, size_t ಗಾತ್ರ_2) {
    apir_decoder dec = {
        .cur   = ptr,
        .ಕೊನೆ   = ptr + ಗಾತ್ರ_2,
        .fatal = false,
    };

    ಹಿಂತಿರುಗಿ dec;
}

ಸ್ಥಿರ apir_encoder apir_new_encoder(ಅಕ್ಷರ * ptr, size_t ಗಾತ್ರ_2) {
    apir_encoder enc = {
        .cur   = ptr,
        .ಆರಂಭ = ptr,
        .ಕೊನೆ   = ptr + ಗಾತ್ರ_2,
        .fatal = false,
    };

    ಹಿಂತಿರುಗಿ enc;
}

/*
 * fatal flag handling
 */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encoder_ಗಣ_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool apir_encoder_get_fatal(ಸ್ಥಿರಾಂಕ apir_encoder * enc) {
    ಹಿಂತಿರುಗಿ enc->fatal;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decoder_ಗಣ_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool apir_decoder_get_fatal(ಸ್ಥಿರಾಂಕ apir_decoder * dec) {
    ಹಿಂತಿರುಗಿ dec->fatal;
}

/*
 * encode peek
 */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool apir_decoder_peek_ಆಂತರಿಕ(apir_decoder * dec, size_t ಗಾತ್ರ_2, ಶೂನ್ಯ * val, size_t val_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ(val_ಗಾತ್ರ <= ಗಾತ್ರ_2);

    ಆದರೆ (unlikely(ಗಾತ್ರ_2 > (size_t) (dec->ಕೊನೆ - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
        memset(val, 0, val_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    ಸ್ಮರಣೆ_ನಕಲು(val, dec->cur, val_ಗಾತ್ರ);
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decoder_peek(apir_decoder * dec, size_t ಗಾತ್ರ_2, ಶೂನ್ಯ * val, size_t val_ಗಾತ್ರ) {
    apir_decoder_peek_ಆಂತರಿಕ(dec, ಗಾತ್ರ_2, val, val_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * apir_decoder_use_inplace(apir_decoder * dec, size_t ಗಾತ್ರ_2) {
    ಆದರೆ (unlikely(ಗಾತ್ರ_2 > (size_t) (dec->ಕೊನೆ - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr = dec->cur;
    dec->cur += ಗಾತ್ರ_2;

    ಹಿಂತಿರುಗಿ addr;
}

/*
 * read/write
 */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decoder_read(apir_decoder * dec, size_t ಗಾತ್ರ_2, ಶೂನ್ಯ * val, size_t val_ಗಾತ್ರ) {
    ಆದರೆ (apir_decoder_peek_ಆಂತರಿಕ(dec, ಗಾತ್ರ_2, val, val_ಗಾತ್ರ)) {
        dec->cur += ಗಾತ್ರ_2;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಅಕ್ಷರ * apir_encoder_write(apir_encoder * enc, size_t ಗಾತ್ರ_2, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * val, size_t val_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ(val_ಗಾತ್ರ <= ಗಾತ್ರ_2);
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 <= ((size_t) (enc->ಕೊನೆ - enc->cur)));

    ಅಕ್ಷರ * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    ಸ್ಮರಣೆ_ನಕಲು(write_addr, val, val_ಗಾತ್ರ);
    enc->cur += ಗಾತ್ರ_2;

    ಹಿಂತಿರುಗಿ write_addr;
}

/*
 * encode/decode
 */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode(apir_decoder * dec, size_t ಗಾತ್ರ_2, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 % 4 == 0);
    apir_decoder_read(dec, ಗಾತ್ರ_2, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode(apir_encoder * enc, size_t ಗಾತ್ರ_2, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 % 4 == 0);
    apir_encoder_write(enc, ಗಾತ್ರ_2, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

/*
 * typed encode/decode
 */

/* uint8_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_uint8_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ uint8_t * val) {
    apir_encode(enc, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), val, ಗಾತ್ರ(*val));
}

/* uint64_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_uint64_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ uint64_t * val) {
    apir_encode(enc, 8, val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_uint64_t_array(apir_encoder * enc, ಸ್ಥಿರಾಂಕ uint64_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_2 = ಗಾತ್ರ(*val) * ಎಣಿಕೆ_2;
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 >= ಎಣಿಕೆ_2);
    apir_encode(enc, ಗಾತ್ರ_2, val, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_2 = ಗಾತ್ರ(*val) * ಎಣಿಕೆ_2;
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 >= ಎಣಿಕೆ_2);
    apir_decode(dec, ಗಾತ್ರ_2, val, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t ಎಣಿಕೆ_2) {
    ಹಿಂತಿರುಗಿ (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, ಎಣಿಕೆ_2 * ಗಾತ್ರ(uint64_t));
}

/* int32_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_int32_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ int32_t * val) {
    apir_encode(enc, 4, val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_int32_t_array(apir_encoder * enc, ಸ್ಥಿರಾಂಕ int32_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_2 = ಗಾತ್ರ(*val) * ಎಣಿಕೆ_2;
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 >= ಎಣಿಕೆ_2);
    apir_encode(enc, ಗಾತ್ರ_2, val, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_2 = ಗಾತ್ರ(*val) * ಎಣಿಕೆ_2;
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 >= ಎಣಿಕೆ_2);
    apir_decode(dec, ಗಾತ್ರ_2, val, ಗಾತ್ರ_2);
}

/* array size (uint64_t) */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_array_ಗಾತ್ರ(apir_encoder * enc, uint64_t ಗಾತ್ರ_2) {
    apir_encode_uint64_t(enc, &ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t apir_decode_array_ಗಾತ್ರ(apir_decoder * dec, uint64_t expected_ಗಾತ್ರ) {
    uint64_t ಗಾತ್ರ_2;
    apir_decode_uint64_t(dec, &ಗಾತ್ರ_2);
    ಆದರೆ (ಗಾತ್ರ_2 != expected_ಗಾತ್ರ) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
        ಗಾತ್ರ_2 = 0;
    }
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ_2;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t apir_decode_array_ಗಾತ್ರ_unchecked(apir_decoder * dec) {
    uint64_t ಗಾತ್ರ_2;
    apir_decode_uint64_t(dec, &ಗಾತ್ರ_2);
    ಹಿಂತಿರುಗಿ ಗಾತ್ರ_2;
}

/* non-array pointer */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool apir_encode_simple_ಸೂಚಕ(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * val) {
    apir_encode_array_ಗಾತ್ರ(enc, val ? 1 : 0);
    ಹಿಂತಿರುಗಿ val;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool apir_decode_simple_ಸೂಚಕ(apir_decoder * dec) {
    ಹಿಂತಿರುಗಿ apir_decode_array_ಗಾತ್ರ_unchecked(dec);
}

/* uint32_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_uint32_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ uint32_t * val) {
    apir_encode(enc, 4, val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_uint32_t_array(apir_encoder * enc, ಸ್ಥಿರಾಂಕ uint32_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_2 = ಗಾತ್ರ(*val) * ಎಣಿಕೆ_2;
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 >= ಎಣಿಕೆ_2);
    apir_encode(enc, ಗಾತ್ರ_2, val, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_2 = ಗಾತ್ರ(*val) * ಎಣಿಕೆ_2;
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 >= ಎಣಿಕೆ_2);
    apir_decode(dec, ಗಾತ್ರ_2, val, ಗಾತ್ರ_2);
}

/* size_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ಗಾತ್ರ_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ size_t * val) {
    ಸ್ಥಿರಾಂಕ uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_ಗಾತ್ರ_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_ಗಾತ್ರ_t_array(apir_encoder * enc, ಸ್ಥಿರಾಂಕ size_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಆದರೆ (ಗಾತ್ರ(size_t) == ಗಾತ್ರ(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ಸ್ಥಿರಾಂಕ uint64_t *) val, ಎಣಿಕೆ_2);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (uint32_t i = 0; i < ಎಣಿಕೆ_2; i++) {
            apir_encode_ಗಾತ್ರ_t(enc, &val[i]);
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_ಗಾತ್ರ_t_array(apir_decoder * dec, size_t * val, uint32_t ಎಣಿಕೆ_2) {
    ಆದರೆ (ಗಾತ್ರ(size_t) == ಗಾತ್ರ(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, ಎಣಿಕೆ_2);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (uint32_t i = 0; i < ಎಣಿಕೆ_2; i++) {
            apir_decode_ಗಾತ್ರ_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_blob_array(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * val, size_t ಗಾತ್ರ_2) {
    apir_encode(enc, (ಗಾತ್ರ_2 + 3) & ~3, val, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_blob_array(apir_decoder * dec, ಶೂನ್ಯ * val, size_t ಗಾತ್ರ_2) {
    apir_decode(dec, (ಗಾತ್ರ_2 + 3) & ~3, val, ಗಾತ್ರ_2);
}

/* string */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_char_array(apir_encoder * enc, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * val, size_t ಗಾತ್ರ_2) {
    ಪ್ರತಿಜ್ಞಾ(ಗಾತ್ರ_2 && ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(val) < ಗಾತ್ರ_2);
    apir_encode_blob_array(enc, val, ಗಾತ್ರ_2);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_char_array(apir_decoder * dec, ಅಕ್ಷರ * val, size_t ಗಾತ್ರ_2) {
    apir_decode_blob_array(dec, val, ಗಾತ್ರ_2);
    ಆದರೆ (ಗಾತ್ರ_2) {
        val[ಗಾತ್ರ_2 - 1] = '\0';
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_ಗಣ_fatal(dec);
    }
}

/* (temp) buffer allocation */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * apir_decoder_alloc_array(size_t ಗಾತ್ರ_2, size_t ಎಣಿಕೆ_2) {
    size_t alloc_ಗಾತ್ರ;
    ಆದರೆ (unlikely(__builtin_mul_overflow(ಗಾತ್ರ_2, ಎಣಿಕೆ_2, &alloc_ಗಾತ್ರ))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, ಗಾತ್ರ_2, ಎಣಿಕೆ_2);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ ಸ್ಮರಣೆ_ಕೊಡು(alloc_ಗಾತ್ರ);
}

/* bool */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_bool_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ bool * val) {
    apir_encode(enc, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), val, ಗಾತ್ರ(bool));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), val, ಗಾತ್ರ(bool));
}

/* apir_buffer_type_host_handle_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_apir_ಬಫರ್_type_host_handle_t(apir_encoder *                         enc,
                                                              ಸ್ಥಿರಾಂಕ apir_ಬಫರ್_type_host_handle_t * val) {
    apir_encode(enc, ಗಾತ್ರ(apir_ಬಫರ್_type_host_handle_t), val, ಗಾತ್ರ(apir_ಬಫರ್_type_host_handle_t));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_apir_ಬಫರ್_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_ಬಫರ್_type_host_handle_t * val) {
    apir_decode(dec, ಗಾತ್ರ(apir_ಬಫರ್_type_host_handle_t), val, ಗಾತ್ರ(apir_ಬಫರ್_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_apir_ಬಫರ್_host_handle_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ apir_ಬಫರ್_host_handle_t * val) {
    apir_encode(enc, ಗಾತ್ರ(apir_ಬಫರ್_host_handle_t), val, ಗಾತ್ರ(apir_ಬಫರ್_host_handle_t));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_apir_ಬಫರ್_host_handle_t(apir_decoder * dec, apir_ಬಫರ್_host_handle_t * val) {
    apir_decode(dec, ಗಾತ್ರ(apir_ಬಫರ್_host_handle_t), val, ಗಾತ್ರ(apir_ಬಫರ್_host_handle_t));
}

/* uintptr_t */

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_encode_uintptr_t(apir_encoder * enc, ಸ್ಥಿರಾಂಕ uintptr_t * val) {
    apir_encode(enc, ಗಾತ್ರ(*val), val, ಗಾತ್ರ(*val));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, ಗಾತ್ರ(*val), val, ಗಾತ್ರ(*val));
}
