#لارښوونه once

#ضمن "ggml-impl.h"

#ضمن <cassert>
#ضمن <cstring>

#تعریف_کړه likely(x)   __builtin_expect(!!(x), 1)
#تعریف_کړه unlikely(x) __builtin_expect(!!(x), 0)

بنية apir_encoder {
    حرف *       cur;
    ثابت حرف * پیل;
    ثابت حرف * پای_ټکی;
    bool         fatal;
};

بنية apir_decoder {
    ثابت حرف * cur;
    ثابت حرف * پای_ټکی;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ساكن apir_decoder apir_new_decoder(ثابت حرف * ptr, size_t اندازه) {
    apir_decoder dec = {
        .cur   = ptr,
        .پای_ټکی   = ptr + اندازه,
        .fatal = false,
    };

    أرجع dec;
}

ساكن apir_encoder apir_new_encoder(حرف * ptr, size_t اندازه) {
    apir_encoder enc = {
        .cur   = ptr,
        .پیل = ptr,
        .پای_ټکی   = ptr + اندازه,
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

ساكن مضمن فراغ apir_encoder_سېټ_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ساكن مضمن bool apir_encoder_get_fatal(ثابت apir_encoder * enc) {
    أرجع enc->fatal;
}

ساكن مضمن فراغ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ساكن مضمن فراغ apir_decoder_سېټ_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ساكن مضمن bool apir_decoder_get_fatal(ثابت apir_decoder * dec) {
    أرجع dec->fatal;
}

/*
 * encode peek
 */

ساكن مضمن bool apir_decoder_peek_دننی(apir_decoder * dec, size_t اندازه, فراغ * val, size_t val_اندازه) {
    أكد(val_اندازه <= اندازه);

    إذا (unlikely(اندازه > (size_t) (dec->پای_ټکی - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سېټ_fatal(dec);
        memset(val, 0, val_اندازه);
        أرجع false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    حافظه_نقل(val, dec->cur, val_اندازه);
    أرجع true;
}

ساكن مضمن فراغ apir_decoder_peek(apir_decoder * dec, size_t اندازه, فراغ * val, size_t val_اندازه) {
    apir_decoder_peek_دننی(dec, اندازه, val, val_اندازه);
}

ساكن مضمن ثابت فراغ * apir_decoder_use_inplace(apir_decoder * dec, size_t اندازه) {
    إذا (unlikely(اندازه > (size_t) (dec->پای_ټکی - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_سېټ_fatal(dec);
        أرجع عدم;
    }
    ثابت فراغ * addr = dec->cur;
    dec->cur += اندازه;

    أرجع addr;
}

/*
 * read/write
 */

ساكن مضمن فراغ apir_decoder_read(apir_decoder * dec, size_t اندازه, فراغ * val, size_t val_اندازه) {
    إذا (apir_decoder_peek_دننی(dec, اندازه, val, val_اندازه)) {
        dec->cur += اندازه;
    }
}

ساكن مضمن حرف * apir_encoder_write(apir_encoder * enc, size_t اندازه, ثابت فراغ * val, size_t val_اندازه) {
    أكد(val_اندازه <= اندازه);
    أكد(اندازه <= ((size_t) (enc->پای_ټکی - enc->cur)));

    حرف * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    حافظه_نقل(write_addr, val, val_اندازه);
    enc->cur += اندازه;

    أرجع write_addr;
}

/*
 * encode/decode
 */

ساكن مضمن فراغ apir_decode(apir_decoder * dec, size_t اندازه, فراغ * معلومات, size_t معلومات_اندازه) {
    أكد(اندازه % 4 == 0);
    apir_decoder_read(dec, اندازه, معلومات, معلومات_اندازه);
}

ساكن مضمن فراغ apir_encode(apir_encoder * enc, size_t اندازه, ثابت فراغ * معلومات, size_t معلومات_اندازه) {
    أكد(اندازه % 4 == 0);
    apir_encoder_write(enc, اندازه, معلومات, معلومات_اندازه);
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

ساكن مضمن فراغ apir_encode_uint64_t_array(apir_encoder * enc, ثابت uint64_t * val, uint32_t شمېر) {
    ثابت size_t اندازه = حجم(*val) * شمېر;
    أكد(اندازه >= شمېر);
    apir_encode(enc, اندازه, val, اندازه);
}

ساكن مضمن فراغ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t شمېر) {
    ثابت size_t اندازه = حجم(*val) * شمېر;
    أكد(اندازه >= شمېر);
    apir_decode(dec, اندازه, val, اندازه);
}

ساكن مضمن ثابت uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t شمېر) {
    أرجع (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, شمېر * حجم(uint64_t));
}

/* int32_t */

ساكن مضمن فراغ apir_encode_int32_t(apir_encoder * enc, ثابت int32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_int32_t_array(apir_encoder * enc, ثابت int32_t * val, uint32_t شمېر) {
    ثابت size_t اندازه = حجم(*val) * شمېر;
    أكد(اندازه >= شمېر);
    apir_encode(enc, اندازه, val, اندازه);
}

ساكن مضمن فراغ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t شمېر) {
    ثابت size_t اندازه = حجم(*val) * شمېر;
    أكد(اندازه >= شمېر);
    apir_decode(dec, اندازه, val, اندازه);
}

/* array size (uint64_t) */

ساكن مضمن فراغ apir_encode_array_اندازه(apir_encoder * enc, uint64_t اندازه) {
    apir_encode_uint64_t(enc, &اندازه);
}

ساكن مضمن uint64_t apir_decode_array_اندازه(apir_decoder * dec, uint64_t expected_اندازه) {
    uint64_t اندازه;
    apir_decode_uint64_t(dec, &اندازه);
    إذا (اندازه != expected_اندازه) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_سېټ_fatal(dec);
        اندازه = 0;
    }
    أرجع اندازه;
}

ساكن مضمن uint64_t apir_decode_array_اندازه_unchecked(apir_decoder * dec) {
    uint64_t اندازه;
    apir_decode_uint64_t(dec, &اندازه);
    أرجع اندازه;
}

/* non-array pointer */

ساكن مضمن bool apir_encode_simple_اشاره_ګر(apir_encoder * enc, ثابت فراغ * val) {
    apir_encode_array_اندازه(enc, val ? 1 : 0);
    أرجع val;
}

ساكن مضمن bool apir_decode_simple_اشاره_ګر(apir_decoder * dec) {
    أرجع apir_decode_array_اندازه_unchecked(dec);
}

/* uint32_t */

ساكن مضمن فراغ apir_encode_uint32_t(apir_encoder * enc, ثابت uint32_t * val) {
    apir_encode(enc, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, حجم(*val));
}

ساكن مضمن فراغ apir_encode_uint32_t_array(apir_encoder * enc, ثابت uint32_t * val, uint32_t شمېر) {
    ثابت size_t اندازه = حجم(*val) * شمېر;
    أكد(اندازه >= شمېر);
    apir_encode(enc, اندازه, val, اندازه);
}

ساكن مضمن فراغ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t شمېر) {
    ثابت size_t اندازه = حجم(*val) * شمېر;
    أكد(اندازه >= شمېر);
    apir_decode(dec, اندازه, val, اندازه);
}

/* size_t */

ساكن مضمن فراغ apir_encode_اندازه_t(apir_encoder * enc, ثابت size_t * val) {
    ثابت uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ساكن مضمن فراغ apir_decode_اندازه_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ساكن مضمن فراغ apir_encode_اندازه_t_array(apir_encoder * enc, ثابت size_t * val, uint32_t شمېر) {
    إذا (حجم(size_t) == حجم(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ثابت uint64_t *) val, شمېر);
    } وإلا {
        لكل (uint32_t i = 0; i < شمېر; i++) {
            apir_encode_اندازه_t(enc, &val[i]);
        }
    }
}

ساكن مضمن فراغ apir_decode_اندازه_t_array(apir_decoder * dec, size_t * val, uint32_t شمېر) {
    إذا (حجم(size_t) == حجم(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, شمېر);
    } وإلا {
        لكل (uint32_t i = 0; i < شمېر; i++) {
            apir_decode_اندازه_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ساكن مضمن فراغ apir_encode_blob_array(apir_encoder * enc, ثابت فراغ * val, size_t اندازه) {
    apir_encode(enc, (اندازه + 3) & ~3, val, اندازه);
}

ساكن مضمن فراغ apir_decode_blob_array(apir_decoder * dec, فراغ * val, size_t اندازه) {
    apir_decode(dec, (اندازه + 3) & ~3, val, اندازه);
}

/* string */

ساكن مضمن فراغ apir_encode_char_array(apir_encoder * enc, ثابت حرف * val, size_t اندازه) {
    أكد(اندازه && مزي_اوږدوالی(val) < اندازه);
    apir_encode_blob_array(enc, val, اندازه);
}

ساكن مضمن فراغ apir_decode_char_array(apir_decoder * dec, حرف * val, size_t اندازه) {
    apir_decode_blob_array(dec, val, اندازه);
    إذا (اندازه) {
        val[اندازه - 1] = '\0';
    } وإلا {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_سېټ_fatal(dec);
    }
}

/* (temp) buffer allocation */

ساكن مضمن فراغ * apir_decoder_alloc_array(size_t اندازه, size_t شمېر) {
    size_t alloc_اندازه;
    إذا (unlikely(__builtin_mul_overflow(اندازه, شمېر, &alloc_اندازه))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, اندازه, شمېر);
        أرجع عدم;
    }

    أرجع حافظه_ورکړه(alloc_اندازه);
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
