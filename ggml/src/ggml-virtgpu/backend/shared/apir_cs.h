#هدايت once

#ضمن "ggml-impl.h"

#ضمن <cassert>
#ضمن <cstring>

#وصف_ڪريو likely(x)   __builtin_expect(!!(x), 1)
#وصف_ڪريو unlikely(x) __builtin_expect(!!(x), 0)

بنية apir_encoder {
    حرف *       cur;
    ثابت حرف * شروعات;
    ثابت حرف * پڇاڙي;
    bool         fatal;
};

بنية apir_decoder {
    ثابت حرف * cur;
    ثابت حرف * پڇاڙي;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ساكن apir_decoder apir_new_decoder(ثابت حرف * ptr, size_t سائيز) {
    apir_decoder dec = {
        .cur   = ptr,
        .پڇاڙي   = ptr + سائيز,
        .fatal = false,
    };

    أرجع dec;
}

ساكن apir_encoder apir_new_encoder(حرف * ptr, size_t سائيز) {
    apir_encoder enc = {
        .cur   = ptr,
        .شروعات = ptr,
        .پڇاڙي   = ptr + سائيز,
        .fatal = false,
    };

    أرجع enc;
}

/*
 * fatal flag handling
 */

ساكن مضمن فراغ apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

ساكن مضمن فراغ apir_encoder_سيٽ_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ساكن مضمن bool apir_encoder_get_fatal(ثابت apir_encoder * enc) {
    أرجع enc->fatal;
}

ساكن مضمن فراغ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ساكن مضمن فراغ apir_decoder_سيٽ_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ساكن مضمن bool apir_decoder_get_fatal(ثابت apir_decoder * dec) {
    أرجع dec->fatal;
}

/*
 * encode peek
 */

ساكن مضمن bool apir_decoder_peek_اندروني(apir_decoder * dec, size_t سائيز, فراغ * val, size_t val_سائيز) {
    أكد(val_سائيز <= سائيز);

    إذا (unlikely(سائيز > (size_t) (dec->پڇاڙي - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سيٽ_fatal(dec);
        memset(val, 0, val_سائيز);
        أرجع false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    يادگيري_نقل(val, dec->cur, val_سائيز);
    أرجع true;
}

ساكن مضمن فراغ apir_decoder_peek(apir_decoder * dec, size_t سائيز, فراغ * val, size_t val_سائيز) {
    apir_decoder_peek_اندروني(dec, سائيز, val, val_سائيز);
}

ساكن مضمن ثابت فراغ * apir_decoder_use_inplace(apir_decoder * dec, size_t سائيز) {
    إذا (unlikely(سائيز > (size_t) (dec->پڇاڙي - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سيٽ_fatal(dec);
        أرجع عدم;
    }
    ثابت فراغ * addr = dec->cur;
    dec->cur += سائيز;

    أرجع addr;
}

/*
 * read/write
 */

ساكن مضمن فراغ apir_decoder_read(apir_decoder * dec, size_t سائيز, فراغ * val, size_t val_سائيز) {
    إذا (apir_decoder_peek_اندروني(dec, سائيز, val, val_سائيز)) {
        dec->cur += سائيز;
    }
}

ساكن مضمن حرف * apir_encoder_write(apir_encoder * enc, size_t سائيز, ثابت فراغ * val, size_t val_سائيز) {
    أكد(val_سائيز <= سائيز);
    أكد(سائيز <= ((size_t) (enc->پڇاڙي - enc->cur)));

    حرف * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    يادگيري_نقل(write_addr, val, val_سائيز);
    enc->cur += سائيز;

    أرجع write_addr;
}

/*
 * encode/decode
 */

ساكن مضمن فراغ apir_decode(apir_decoder * dec, size_t سائيز, فراغ * معلومات, size_t معلومات_سائيز) {
    أكد(سائيز % 4 == 0);
    apir_decoder_read(dec, سائيز, معلومات, معلومات_سائيز);
}

ساكن مضمن فراغ apir_encode(apir_encoder * enc, size_t سائيز, ثابت فراغ * معلومات, size_t معلومات_سائيز) {
    أكد(سائيز % 4 == 0);
    apir_encoder_write(enc, سائيز, معلومات, معلومات_سائيز);
}

/*
 * typed encode/decode
 */

/* uint8_t */

ساكن مضمن فراغ apir_encode_uint8_t(apir_encoder * enc, ثابت uint8_t * val) {
    apir_encode(enc, حجم(صحيح), val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, حجم(صحيح), val, حجم(*val));
}

/* uint64_t */

ساكن مضمن فراغ apir_encode_uint64_t(apir_encoder * enc, ثابت uint64_t * val) {
    apir_encode(enc, 8, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_uint64_t_array(apir_encoder * enc, ثابت uint64_t * val, uint32_t ڳڻپ) {
    ثابت size_t سائيز = حجم(*val) * ڳڻپ;
    أكد(سائيز >= ڳڻپ);
    apir_encode(enc, سائيز, val, سائيز);
}

ساكن مضمن فراغ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t ڳڻپ) {
    ثابت size_t سائيز = حجم(*val) * ڳڻپ;
    أكد(سائيز >= ڳڻپ);
    apir_decode(dec, سائيز, val, سائيز);
}

ساكن مضمن ثابت uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t ڳڻپ) {
    أرجع (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, ڳڻپ * حجم(uint64_t));
}

/* int32_t */

ساكن مضمن فراغ apir_encode_int32_t(apir_encoder * enc, ثابت int32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_int32_t_array(apir_encoder * enc, ثابت int32_t * val, uint32_t ڳڻپ) {
    ثابت size_t سائيز = حجم(*val) * ڳڻپ;
    أكد(سائيز >= ڳڻپ);
    apir_encode(enc, سائيز, val, سائيز);
}

ساكن مضمن فراغ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t ڳڻپ) {
    ثابت size_t سائيز = حجم(*val) * ڳڻپ;
    أكد(سائيز >= ڳڻپ);
    apir_decode(dec, سائيز, val, سائيز);
}

/* array size (uint64_t) */

ساكن مضمن فراغ apir_encode_array_سائيز(apir_encoder * enc, uint64_t سائيز) {
    apir_encode_uint64_t(enc, &سائيز);
}

ساكن مضمن uint64_t apir_decode_array_سائيز(apir_decoder * dec, uint64_t expected_سائيز) {
    uint64_t سائيز;
    apir_decode_uint64_t(dec, &سائيز);
    إذا (سائيز != expected_سائيز) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_سيٽ_fatal(dec);
        سائيز = 0;
    }
    أرجع سائيز;
}

ساكن مضمن uint64_t apir_decode_array_سائيز_unchecked(apir_decoder * dec) {
    uint64_t سائيز;
    apir_decode_uint64_t(dec, &سائيز);
    أرجع سائيز;
}

/* non-array pointer */

ساكن مضمن bool apir_encode_simple_اشارو(apir_encoder * enc, ثابت فراغ * val) {
    apir_encode_array_سائيز(enc, val ? 1 : 0);
    أرجع val;
}

ساكن مضمن bool apir_decode_simple_اشارو(apir_decoder * dec) {
    أرجع apir_decode_array_سائيز_unchecked(dec);
}

/* uint32_t */

ساكن مضمن فراغ apir_encode_uint32_t(apir_encoder * enc, ثابت uint32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_uint32_t_array(apir_encoder * enc, ثابت uint32_t * val, uint32_t ڳڻپ) {
    ثابت size_t سائيز = حجم(*val) * ڳڻپ;
    أكد(سائيز >= ڳڻپ);
    apir_encode(enc, سائيز, val, سائيز);
}

ساكن مضمن فراغ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t ڳڻپ) {
    ثابت size_t سائيز = حجم(*val) * ڳڻپ;
    أكد(سائيز >= ڳڻپ);
    apir_decode(dec, سائيز, val, سائيز);
}

/* size_t */

ساكن مضمن فراغ apir_encode_سائيز_t(apir_encoder * enc, ثابت size_t * val) {
    ثابت uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ساكن مضمن فراغ apir_decode_سائيز_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ساكن مضمن فراغ apir_encode_سائيز_t_array(apir_encoder * enc, ثابت size_t * val, uint32_t ڳڻپ) {
    إذا (حجم(size_t) == حجم(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ثابت uint64_t *) val, ڳڻپ);
    } وإلا {
        لكل (uint32_t i = 0; i < ڳڻپ; i++) {
            apir_encode_سائيز_t(enc, &val[i]);
        }
    }
}

ساكن مضمن فراغ apir_decode_سائيز_t_array(apir_decoder * dec, size_t * val, uint32_t ڳڻپ) {
    إذا (حجم(size_t) == حجم(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, ڳڻپ);
    } وإلا {
        لكل (uint32_t i = 0; i < ڳڻپ; i++) {
            apir_decode_سائيز_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ساكن مضمن فراغ apir_encode_blob_array(apir_encoder * enc, ثابت فراغ * val, size_t سائيز) {
    apir_encode(enc, (سائيز + 3) & ~3, val, سائيز);
}

ساكن مضمن فراغ apir_decode_blob_array(apir_decoder * dec, فراغ * val, size_t سائيز) {
    apir_decode(dec, (سائيز + 3) & ~3, val, سائيز);
}

/* string */

ساكن مضمن فراغ apir_encode_char_array(apir_encoder * enc, ثابت حرف * val, size_t سائيز) {
    أكد(سائيز && لڙي_ڊيگهه(val) < سائيز);
    apir_encode_blob_array(enc, val, سائيز);
}

ساكن مضمن فراغ apir_decode_char_array(apir_decoder * dec, حرف * val, size_t سائيز) {
    apir_decode_blob_array(dec, val, سائيز);
    إذا (سائيز) {
        val[سائيز - 1] = '\0';
    } وإلا {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_سيٽ_fatal(dec);
    }
}

/* (temp) buffer allocation */

ساكن مضمن فراغ * apir_decoder_alloc_array(size_t سائيز, size_t ڳڻپ) {
    size_t alloc_سائيز;
    إذا (unlikely(__builtin_mul_overflow(سائيز, ڳڻپ, &alloc_سائيز))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, سائيز, ڳڻپ);
        أرجع عدم;
    }

    أرجع يادگيري_ڏيو(alloc_سائيز);
}

/* bool */

ساكن مضمن فراغ apir_encode_bool_t(apir_encoder * enc, ثابت bool * val) {
    apir_encode(enc, حجم(صحيح), val, حجم(bool));
}

ساكن مضمن فراغ apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, حجم(صحيح), val, حجم(bool));
}

/* apir_buffer_type_host_handle_t */

ساكن مضمن فراغ apir_encode_apir_بفر_type_host_handle_t(apir_encoder *                         enc,
                                                              ثابت apir_بفر_type_host_handle_t * val) {
    apir_encode(enc, حجم(apir_بفر_type_host_handle_t), val, حجم(apir_بفر_type_host_handle_t));
}

ساكن مضمن فراغ apir_decode_apir_بفر_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_بفر_type_host_handle_t * val) {
    apir_decode(dec, حجم(apir_بفر_type_host_handle_t), val, حجم(apir_بفر_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

ساكن مضمن فراغ apir_encode_apir_بفر_host_handle_t(apir_encoder * enc, ثابت apir_بفر_host_handle_t * val) {
    apir_encode(enc, حجم(apir_بفر_host_handle_t), val, حجم(apir_بفر_host_handle_t));
}

ساكن مضمن فراغ apir_decode_apir_بفر_host_handle_t(apir_decoder * dec, apir_بفر_host_handle_t * val) {
    apir_decode(dec, حجم(apir_بفر_host_handle_t), val, حجم(apir_بفر_host_handle_t));
}

/* uintptr_t */

ساكن مضمن فراغ apir_encode_uintptr_t(apir_encoder * enc, ثابت uintptr_t * val) {
    apir_encode(enc, حجم(*val), val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, حجم(*val), val, حجم(*val));
}
