#ہدایت once

#شامل "ggml-impl.h"

#شامل <cassert>
#شامل <cstring>

#تعریف_کرو likely(x)   __builtin_expect(!!(x), 1)
#تعریف_کرو unlikely(x) __builtin_expect(!!(x), 0)

ساخت apir_encoder {
    حرف *       cur;
    ثابت حرف * آغاز;
    ثابت حرف * انتہا;
    bool         fatal;
};

ساخت apir_decoder {
    ثابت حرف * cur;
    ثابت حرف * انتہا;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ساکن apir_decoder apir_new_decoder(ثابت حرف * ptr, size_t سائز) {
    apir_decoder dec = {
        .cur   = ptr,
        .انتہا   = ptr + سائز,
        .fatal = false,
    };

    واپس dec;
}

ساکن apir_encoder apir_new_encoder(حرف * ptr, size_t سائز) {
    apir_encoder enc = {
        .cur   = ptr,
        .آغاز = ptr,
        .انتہا   = ptr + سائز,
        .fatal = false,
    };

    واپس enc;
}

/*
 * fatal flag handling
 */

ساکن اندرونی خالی apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

ساکن اندرونی خالی apir_encoder_سیٹ_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ساکن اندرونی bool apir_encoder_get_fatal(ثابت apir_encoder * enc) {
    واپس enc->fatal;
}

ساکن اندرونی خالی apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ساکن اندرونی خالی apir_decoder_سیٹ_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ساکن اندرونی bool apir_decoder_get_fatal(ثابت apir_decoder * dec) {
    واپس dec->fatal;
}

/*
 * encode peek
 */

ساکن اندرونی bool apir_decoder_peek_اندرونی(apir_decoder * dec, size_t سائز, خالی * val, size_t val_سائز) {
    أكد(val_سائز <= سائز);

    اگر (unlikely(سائز > (size_t) (dec->انتہا - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        memset(val, 0, val_سائز);
        واپس false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    حافظہ_نقل(val, dec->cur, val_سائز);
    واپس true;
}

ساکن اندرونی خالی apir_decoder_peek(apir_decoder * dec, size_t سائز, خالی * val, size_t val_سائز) {
    apir_decoder_peek_اندرونی(dec, سائز, val, val_سائز);
}

ساکن اندرونی ثابت خالی * apir_decoder_use_inplace(apir_decoder * dec, size_t سائز) {
    اگر (unlikely(سائز > (size_t) (dec->انتہا - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        واپس تہی;
    }
    ثابت خالی * addr = dec->cur;
    dec->cur += سائز;

    واپس addr;
}

/*
 * read/write
 */

ساکن اندرونی خالی apir_decoder_read(apir_decoder * dec, size_t سائز, خالی * val, size_t val_سائز) {
    اگر (apir_decoder_peek_اندرونی(dec, سائز, val, val_سائز)) {
        dec->cur += سائز;
    }
}

ساکن اندرونی حرف * apir_encoder_write(apir_encoder * enc, size_t سائز, ثابت خالی * val, size_t val_سائز) {
    أكد(val_سائز <= سائز);
    أكد(سائز <= ((size_t) (enc->انتہا - enc->cur)));

    حرف * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    حافظہ_نقل(write_addr, val, val_سائز);
    enc->cur += سائز;

    واپس write_addr;
}

/*
 * encode/decode
 */

ساکن اندرونی خالی apir_decode(apir_decoder * dec, size_t سائز, خالی * ڈیٹا, size_t ڈیٹا_سائز) {
    أكد(سائز % 4 == 0);
    apir_decoder_read(dec, سائز, ڈیٹا, ڈیٹا_سائز);
}

ساکن اندرونی خالی apir_encode(apir_encoder * enc, size_t سائز, ثابت خالی * ڈیٹا, size_t ڈیٹا_سائز) {
    أكد(سائز % 4 == 0);
    apir_encoder_write(enc, سائز, ڈیٹا, ڈیٹا_سائز);
}

/*
 * typed encode/decode
 */

/* uint8_t */

ساکن اندرونی خالی apir_encode_uint8_t(apir_encoder * enc, ثابت uint8_t * val) {
    apir_encode(enc, حجم(صحیح), val, حجم(*val));
}

ساکن اندرونی خالی apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, حجم(صحیح), val, حجم(*val));
}

/* uint64_t */

ساکن اندرونی خالی apir_encode_uint64_t(apir_encoder * enc, ثابت uint64_t * val) {
    apir_encode(enc, 8, val, حجم(*val));
}

ساکن اندرونی خالی apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, حجم(*val));
}

ساکن اندرونی خالی apir_encode_uint64_t_array(apir_encoder * enc, ثابت uint64_t * val, uint32_t گنتی) {
    ثابت size_t سائز = حجم(*val) * گنتی;
    أكد(سائز >= گنتی);
    apir_encode(enc, سائز, val, سائز);
}

ساکن اندرونی خالی apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t گنتی) {
    ثابت size_t سائز = حجم(*val) * گنتی;
    أكد(سائز >= گنتی);
    apir_decode(dec, سائز, val, سائز);
}

ساکن اندرونی ثابت uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t گنتی) {
    واپس (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, گنتی * حجم(uint64_t));
}

/* int32_t */

ساکن اندرونی خالی apir_encode_int32_t(apir_encoder * enc, ثابت int32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساکن اندرونی خالی apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساکن اندرونی خالی apir_encode_int32_t_array(apir_encoder * enc, ثابت int32_t * val, uint32_t گنتی) {
    ثابت size_t سائز = حجم(*val) * گنتی;
    أكد(سائز >= گنتی);
    apir_encode(enc, سائز, val, سائز);
}

ساکن اندرونی خالی apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t گنتی) {
    ثابت size_t سائز = حجم(*val) * گنتی;
    أكد(سائز >= گنتی);
    apir_decode(dec, سائز, val, سائز);
}

/* array size (uint64_t) */

ساکن اندرونی خالی apir_encode_array_سائز(apir_encoder * enc, uint64_t سائز) {
    apir_encode_uint64_t(enc, &سائز);
}

ساکن اندرونی uint64_t apir_decode_array_سائز(apir_decoder * dec, uint64_t expected_سائز) {
    uint64_t سائز;
    apir_decode_uint64_t(dec, &سائز);
    اگر (سائز != expected_سائز) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
        سائز = 0;
    }
    واپس سائز;
}

ساکن اندرونی uint64_t apir_decode_array_سائز_unchecked(apir_decoder * dec) {
    uint64_t سائز;
    apir_decode_uint64_t(dec, &سائز);
    واپس سائز;
}

/* non-array pointer */

ساکن اندرونی bool apir_encode_simple_اشارہ_گر(apir_encoder * enc, ثابت خالی * val) {
    apir_encode_array_سائز(enc, val ? 1 : 0);
    واپس val;
}

ساکن اندرونی bool apir_decode_simple_اشارہ_گر(apir_decoder * dec) {
    واپس apir_decode_array_سائز_unchecked(dec);
}

/* uint32_t */

ساکن اندرونی خالی apir_encode_uint32_t(apir_encoder * enc, ثابت uint32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساکن اندرونی خالی apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساکن اندرونی خالی apir_encode_uint32_t_array(apir_encoder * enc, ثابت uint32_t * val, uint32_t گنتی) {
    ثابت size_t سائز = حجم(*val) * گنتی;
    أكد(سائز >= گنتی);
    apir_encode(enc, سائز, val, سائز);
}

ساکن اندرونی خالی apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t گنتی) {
    ثابت size_t سائز = حجم(*val) * گنتی;
    أكد(سائز >= گنتی);
    apir_decode(dec, سائز, val, سائز);
}

/* size_t */

ساکن اندرونی خالی apir_encode_سائز_t(apir_encoder * enc, ثابت size_t * val) {
    ثابت uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ساکن اندرونی خالی apir_decode_سائز_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ساکن اندرونی خالی apir_encode_سائز_t_array(apir_encoder * enc, ثابت size_t * val, uint32_t گنتی) {
    اگر (حجم(size_t) == حجم(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ثابت uint64_t *) val, گنتی);
    } ورنہ {
        برائے (uint32_t i = 0; i < گنتی; i++) {
            apir_encode_سائز_t(enc, &val[i]);
        }
    }
}

ساکن اندرونی خالی apir_decode_سائز_t_array(apir_decoder * dec, size_t * val, uint32_t گنتی) {
    اگر (حجم(size_t) == حجم(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, گنتی);
    } ورنہ {
        برائے (uint32_t i = 0; i < گنتی; i++) {
            apir_decode_سائز_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ساکن اندرونی خالی apir_encode_blob_array(apir_encoder * enc, ثابت خالی * val, size_t سائز) {
    apir_encode(enc, (سائز + 3) & ~3, val, سائز);
}

ساکن اندرونی خالی apir_decode_blob_array(apir_decoder * dec, خالی * val, size_t سائز) {
    apir_decode(dec, (سائز + 3) & ~3, val, سائز);
}

/* string */

ساکن اندرونی خالی apir_encode_char_array(apir_encoder * enc, ثابت حرف * val, size_t سائز) {
    أكد(سائز && لڑی_لمبائی(val) < سائز);
    apir_encode_blob_array(enc, val, سائز);
}

ساکن اندرونی خالی apir_decode_char_array(apir_decoder * dec, حرف * val, size_t سائز) {
    apir_decode_blob_array(dec, val, سائز);
    اگر (سائز) {
        val[سائز - 1] = '\0';
    } ورنہ {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_سیٹ_fatal(dec);
    }
}

/* (temp) buffer allocation */

ساکن اندرونی خالی * apir_decoder_alloc_array(size_t سائز, size_t گنتی) {
    size_t alloc_سائز;
    اگر (unlikely(__builtin_mul_overflow(سائز, گنتی, &alloc_سائز))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, سائز, گنتی);
        واپس تہی;
    }

    واپس حافظہ_دو(alloc_سائز);
}

/* bool */

ساکن اندرونی خالی apir_encode_bool_t(apir_encoder * enc, ثابت bool * val) {
    apir_encode(enc, حجم(صحیح), val, حجم(bool));
}

ساکن اندرونی خالی apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, حجم(صحیح), val, حجم(bool));
}

/* apir_buffer_type_host_handle_t */

ساکن اندرونی خالی apir_encode_apir_بفر_type_host_handle_t(apir_encoder *                         enc,
                                                              ثابت apir_بفر_type_host_handle_t * val) {
    apir_encode(enc, حجم(apir_بفر_type_host_handle_t), val, حجم(apir_بفر_type_host_handle_t));
}

ساکن اندرونی خالی apir_decode_apir_بفر_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_بفر_type_host_handle_t * val) {
    apir_decode(dec, حجم(apir_بفر_type_host_handle_t), val, حجم(apir_بفر_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

ساکن اندرونی خالی apir_encode_apir_بفر_host_handle_t(apir_encoder * enc, ثابت apir_بفر_host_handle_t * val) {
    apir_encode(enc, حجم(apir_بفر_host_handle_t), val, حجم(apir_بفر_host_handle_t));
}

ساکن اندرونی خالی apir_decode_apir_بفر_host_handle_t(apir_decoder * dec, apir_بفر_host_handle_t * val) {
    apir_decode(dec, حجم(apir_بفر_host_handle_t), val, حجم(apir_بفر_host_handle_t));
}

/* uintptr_t */

ساکن اندرونی خالی apir_encode_uintptr_t(apir_encoder * enc, ثابت uintptr_t * val) {
    apir_encode(enc, حجم(*val), val, حجم(*val));
}

ساکن اندرونی خالی apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, حجم(*val), val, حجم(*val));
}
