#ہدایت once

#ضمن "ggml-impl.h"

#ضمن <cassert>
#ضمن <cstring>

#تعریف_کرو likely(x)   __builtin_expect(!!(x), 1)
#تعریف_کرو unlikely(x) __builtin_expect(!!(x), 0)

بنية apir_encoder {
    حرف *       cur;
    ثابت حرف * شروع;
    ثابت حرف * سرا;
    bool         fatal;
};

بنية apir_decoder {
    ثابت حرف * cur;
    ثابت حرف * سرا;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ساكن apir_decoder apir_new_decoder(ثابت حرف * ptr, size_t ناپ) {
    apir_decoder dec = {
        .cur   = ptr,
        .سرا   = ptr + ناپ,
        .fatal = false,
    };

    أرجع dec;
}

ساكن apir_encoder apir_new_encoder(حرف * ptr, size_t ناپ) {
    apir_encoder enc = {
        .cur   = ptr,
        .شروع = ptr,
        .سرا   = ptr + ناپ,
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

ساكن مضمن فراغ apir_encoder_سیٹ_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ساكن مضمن bool apir_encoder_get_fatal(ثابت apir_encoder * enc) {
    أرجع enc->fatal;
}

ساكن مضمن فراغ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ساكن مضمن فراغ apir_decoder_سیٹ_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ساكن مضمن bool apir_decoder_get_fatal(ثابت apir_decoder * dec) {
    أرجع dec->fatal;
}

/*
 * encode peek
 */

ساكن مضمن bool apir_decoder_peek_اندرونی(apir_decoder * dec, size_t ناپ, فراغ * val, size_t val_ناپ) {
    أكد(val_ناپ <= ناپ);

    إذا (unlikely(ناپ > (size_t) (dec->سرا - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        memset(val, 0, val_ناپ);
        أرجع false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    حافظہ_نقل(val, dec->cur, val_ناپ);
    أرجع true;
}

ساكن مضمن فراغ apir_decoder_peek(apir_decoder * dec, size_t ناپ, فراغ * val, size_t val_ناپ) {
    apir_decoder_peek_اندرونی(dec, ناپ, val, val_ناپ);
}

ساكن مضمن ثابت فراغ * apir_decoder_use_inplace(apir_decoder * dec, size_t ناپ) {
    إذا (unlikely(ناپ > (size_t) (dec->سرا - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        أرجع عدم;
    }
    ثابت فراغ * addr = dec->cur;
    dec->cur += ناپ;

    أرجع addr;
}

/*
 * read/write
 */

ساكن مضمن فراغ apir_decoder_read(apir_decoder * dec, size_t ناپ, فراغ * val, size_t val_ناپ) {
    إذا (apir_decoder_peek_اندرونی(dec, ناپ, val, val_ناپ)) {
        dec->cur += ناپ;
    }
}

ساكن مضمن حرف * apir_encoder_write(apir_encoder * enc, size_t ناپ, ثابت فراغ * val, size_t val_ناپ) {
    أكد(val_ناپ <= ناپ);
    أكد(ناپ <= ((size_t) (enc->سرا - enc->cur)));

    حرف * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    حافظہ_نقل(write_addr, val, val_ناپ);
    enc->cur += ناپ;

    أرجع write_addr;
}

/*
 * encode/decode
 */

ساكن مضمن فراغ apir_decode(apir_decoder * dec, size_t ناپ, فراغ * انکڑے, size_t انکڑے_ناپ) {
    أكد(ناپ % 4 == 0);
    apir_decoder_read(dec, ناپ, انکڑے, انکڑے_ناپ);
}

ساكن مضمن فراغ apir_encode(apir_encoder * enc, size_t ناپ, ثابت فراغ * انکڑے, size_t انکڑے_ناپ) {
    أكد(ناپ % 4 == 0);
    apir_encoder_write(enc, ناپ, انکڑے, انکڑے_ناپ);
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

ساكن مضمن فراغ apir_encode_uint64_t_array(apir_encoder * enc, ثابت uint64_t * val, uint32_t گنتی) {
    ثابت size_t ناپ = حجم(*val) * گنتی;
    أكد(ناپ >= گنتی);
    apir_encode(enc, ناپ, val, ناپ);
}

ساكن مضمن فراغ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t گنتی) {
    ثابت size_t ناپ = حجم(*val) * گنتی;
    أكد(ناپ >= گنتی);
    apir_decode(dec, ناپ, val, ناپ);
}

ساكن مضمن ثابت uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t گنتی) {
    أرجع (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, گنتی * حجم(uint64_t));
}

/* int32_t */

ساكن مضمن فراغ apir_encode_int32_t(apir_encoder * enc, ثابت int32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_int32_t_array(apir_encoder * enc, ثابت int32_t * val, uint32_t گنتی) {
    ثابت size_t ناپ = حجم(*val) * گنتی;
    أكد(ناپ >= گنتی);
    apir_encode(enc, ناپ, val, ناپ);
}

ساكن مضمن فراغ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t گنتی) {
    ثابت size_t ناپ = حجم(*val) * گنتی;
    أكد(ناپ >= گنتی);
    apir_decode(dec, ناپ, val, ناپ);
}

/* array size (uint64_t) */

ساكن مضمن فراغ apir_encode_array_ناپ(apir_encoder * enc, uint64_t ناپ) {
    apir_encode_uint64_t(enc, &ناپ);
}

ساكن مضمن uint64_t apir_decode_array_ناپ(apir_decoder * dec, uint64_t expected_ناپ) {
    uint64_t ناپ;
    apir_decode_uint64_t(dec, &ناپ);
    إذا (ناپ != expected_ناپ) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        ناپ = 0;
    }
    أرجع ناپ;
}

ساكن مضمن uint64_t apir_decode_array_ناپ_unchecked(apir_decoder * dec) {
    uint64_t ناپ;
    apir_decode_uint64_t(dec, &ناپ);
    أرجع ناپ;
}

/* non-array pointer */

ساكن مضمن bool apir_encode_simple_اشارہ_گر(apir_encoder * enc, ثابت فراغ * val) {
    apir_encode_array_ناپ(enc, val ? 1 : 0);
    أرجع val;
}

ساكن مضمن bool apir_decode_simple_اشارہ_گر(apir_decoder * dec) {
    أرجع apir_decode_array_ناپ_unchecked(dec);
}

/* uint32_t */

ساكن مضمن فراغ apir_encode_uint32_t(apir_encoder * enc, ثابت uint32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_uint32_t_array(apir_encoder * enc, ثابت uint32_t * val, uint32_t گنتی) {
    ثابت size_t ناپ = حجم(*val) * گنتی;
    أكد(ناپ >= گنتی);
    apir_encode(enc, ناپ, val, ناپ);
}

ساكن مضمن فراغ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t گنتی) {
    ثابت size_t ناپ = حجم(*val) * گنتی;
    أكد(ناپ >= گنتی);
    apir_decode(dec, ناپ, val, ناپ);
}

/* size_t */

ساكن مضمن فراغ apir_encode_ناپ_t(apir_encoder * enc, ثابت size_t * val) {
    ثابت uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ساكن مضمن فراغ apir_decode_ناپ_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ساكن مضمن فراغ apir_encode_ناپ_t_array(apir_encoder * enc, ثابت size_t * val, uint32_t گنتی) {
    إذا (حجم(size_t) == حجم(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ثابت uint64_t *) val, گنتی);
    } وإلا {
        لكل (uint32_t i = 0; i < گنتی; i++) {
            apir_encode_ناپ_t(enc, &val[i]);
        }
    }
}

ساكن مضمن فراغ apir_decode_ناپ_t_array(apir_decoder * dec, size_t * val, uint32_t گنتی) {
    إذا (حجم(size_t) == حجم(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, گنتی);
    } وإلا {
        لكل (uint32_t i = 0; i < گنتی; i++) {
            apir_decode_ناپ_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ساكن مضمن فراغ apir_encode_blob_array(apir_encoder * enc, ثابت فراغ * val, size_t ناپ) {
    apir_encode(enc, (ناپ + 3) & ~3, val, ناپ);
}

ساكن مضمن فراغ apir_decode_blob_array(apir_decoder * dec, فراغ * val, size_t ناپ) {
    apir_decode(dec, (ناپ + 3) & ~3, val, ناپ);
}

/* string */

ساكن مضمن فراغ apir_encode_char_array(apir_encoder * enc, ثابت حرف * val, size_t ناپ) {
    أكد(ناپ && لڑی_لمبائی(val) < ناپ);
    apir_encode_blob_array(enc, val, ناپ);
}

ساكن مضمن فراغ apir_decode_char_array(apir_decoder * dec, حرف * val, size_t ناپ) {
    apir_decode_blob_array(dec, val, ناپ);
    إذا (ناپ) {
        val[ناپ - 1] = '\0';
    } وإلا {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
    }
}

/* (temp) buffer allocation */

ساكن مضمن فراغ * apir_decoder_alloc_array(size_t ناپ, size_t گنتی) {
    size_t alloc_ناپ;
    إذا (unlikely(__builtin_mul_overflow(ناپ, گنتی, &alloc_ناپ))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, ناپ, گنتی);
        أرجع عدم;
    }

    أرجع حافظہ_دیو(alloc_ناپ);
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
