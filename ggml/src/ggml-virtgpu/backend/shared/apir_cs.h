#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ likely(x)   __builtin_expect(!!(x), 1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ unlikely(x) __builtin_expect(!!(x), 0)

ਬਣਤਰ apir_encoder {
    ਅੱਖਰ *       cur;
    ਅਚਲ ਅੱਖਰ * ਸ਼ੁਰੂ;
    ਅਚਲ ਅੱਖਰ * ਸਿਰਾ;
    bool         fatal;
};

ਬਣਤਰ apir_decoder {
    ਅਚਲ ਅੱਖਰ * cur;
    ਅਚਲ ਅੱਖਰ * ਸਿਰਾ;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ਸਥਿਰ apir_decoder apir_new_decoder(ਅਚਲ ਅੱਖਰ * ptr, size_t ਮਾਪ) {
    apir_decoder dec = {
        .cur   = ptr,
        .ਸਿਰਾ   = ptr + ਮਾਪ,
        .fatal = false,
    };

    ਵਾਪਸ dec;
}

ਸਥਿਰ apir_encoder apir_new_encoder(ਅੱਖਰ * ptr, size_t ਮਾਪ) {
    apir_encoder enc = {
        .cur   = ptr,
        .ਸ਼ੁਰੂ = ptr,
        .ਸਿਰਾ   = ptr + ਮਾਪ,
        .fatal = false,
    };

    ਵਾਪਸ enc;
}

/*
 * fatal flag handling
 */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encoder_ਸੈੱਟ_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool apir_encoder_get_fatal(ਅਚਲ apir_encoder * enc) {
    ਵਾਪਸ enc->fatal;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decoder_ਸੈੱਟ_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool apir_decoder_get_fatal(ਅਚਲ apir_decoder * dec) {
    ਵਾਪਸ dec->fatal;
}

/*
 * encode peek
 */

ਸਥਿਰ ਸਤਰਅੰਦਰ bool apir_decoder_peek_ਅੰਦਰੂਨੀ(apir_decoder * dec, size_t ਮਾਪ, ਖਾਲੀ * val, size_t val_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(val_ਮਾਪ <= ਮਾਪ);

    ਜੇ (unlikely(ਮਾਪ > (size_t) (dec->ਸਿਰਾ - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
        memset(val, 0, val_ਮਾਪ);
        ਵਾਪਸ false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(val, dec->cur, val_ਮਾਪ);
    ਵਾਪਸ true;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decoder_peek(apir_decoder * dec, size_t ਮਾਪ, ਖਾਲੀ * val, size_t val_ਮਾਪ) {
    apir_decoder_peek_ਅੰਦਰੂਨੀ(dec, ਮਾਪ, val, val_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਖਾਲੀ * apir_decoder_use_inplace(apir_decoder * dec, size_t ਮਾਪ) {
    ਜੇ (unlikely(ਮਾਪ > (size_t) (dec->ਸਿਰਾ - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਅਚਲ ਖਾਲੀ * addr = dec->cur;
    dec->cur += ਮਾਪ;

    ਵਾਪਸ addr;
}

/*
 * read/write
 */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decoder_read(apir_decoder * dec, size_t ਮਾਪ, ਖਾਲੀ * val, size_t val_ਮਾਪ) {
    ਜੇ (apir_decoder_peek_ਅੰਦਰੂਨੀ(dec, ਮਾਪ, val, val_ਮਾਪ)) {
        dec->cur += ਮਾਪ;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅੱਖਰ * apir_encoder_write(apir_encoder * enc, size_t ਮਾਪ, ਅਚਲ ਖਾਲੀ * val, size_t val_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(val_ਮਾਪ <= ਮਾਪ);
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ <= ((size_t) (enc->ਸਿਰਾ - enc->cur)));

    ਅੱਖਰ * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(write_addr, val, val_ਮਾਪ);
    enc->cur += ਮਾਪ;

    ਵਾਪਸ write_addr;
}

/*
 * encode/decode
 */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode(apir_decoder * dec, size_t ਮਾਪ, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ % 4 == 0);
    apir_decoder_read(dec, ਮਾਪ, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode(apir_encoder * enc, size_t ਮਾਪ, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ % 4 == 0);
    apir_encoder_write(enc, ਮਾਪ, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

/*
 * typed encode/decode
 */

/* uint8_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_uint8_t(apir_encoder * enc, ਅਚਲ uint8_t * val) {
    apir_encode(enc, ਆਕਾਰ(ਪੂਰਨ), val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, ਆਕਾਰ(ਪੂਰਨ), val, ਆਕਾਰ(*val));
}

/* uint64_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_uint64_t(apir_encoder * enc, ਅਚਲ uint64_t * val) {
    apir_encode(enc, 8, val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_uint64_t_array(apir_encoder * enc, ਅਚਲ uint64_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਅਚਲ size_t ਮਾਪ = ਆਕਾਰ(*val) * ਗਿਣਤੀ_2;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ >= ਗਿਣਤੀ_2);
    apir_encode(enc, ਮਾਪ, val, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਅਚਲ size_t ਮਾਪ = ਆਕਾਰ(*val) * ਗਿਣਤੀ_2;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ >= ਗਿਣਤੀ_2);
    apir_decode(dec, ਮਾਪ, val, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t ਗਿਣਤੀ_2) {
    ਵਾਪਸ (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, ਗਿਣਤੀ_2 * ਆਕਾਰ(uint64_t));
}

/* int32_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_int32_t(apir_encoder * enc, ਅਚਲ int32_t * val) {
    apir_encode(enc, 4, val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_int32_t_array(apir_encoder * enc, ਅਚਲ int32_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਅਚਲ size_t ਮਾਪ = ਆਕਾਰ(*val) * ਗਿਣਤੀ_2;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ >= ਗਿਣਤੀ_2);
    apir_encode(enc, ਮਾਪ, val, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਅਚਲ size_t ਮਾਪ = ਆਕਾਰ(*val) * ਗਿਣਤੀ_2;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ >= ਗਿਣਤੀ_2);
    apir_decode(dec, ਮਾਪ, val, ਮਾਪ);
}

/* array size (uint64_t) */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_array_ਮਾਪ(apir_encoder * enc, uint64_t ਮਾਪ) {
    apir_encode_uint64_t(enc, &ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t apir_decode_array_ਮਾਪ(apir_decoder * dec, uint64_t expected_ਮਾਪ) {
    uint64_t ਮਾਪ;
    apir_decode_uint64_t(dec, &ਮਾਪ);
    ਜੇ (ਮਾਪ != expected_ਮਾਪ) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
        ਮਾਪ = 0;
    }
    ਵਾਪਸ ਮਾਪ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t apir_decode_array_ਮਾਪ_unchecked(apir_decoder * dec) {
    uint64_t ਮਾਪ;
    apir_decode_uint64_t(dec, &ਮਾਪ);
    ਵਾਪਸ ਮਾਪ;
}

/* non-array pointer */

ਸਥਿਰ ਸਤਰਅੰਦਰ bool apir_encode_simple_ਸੂਚਕ(apir_encoder * enc, ਅਚਲ ਖਾਲੀ * val) {
    apir_encode_array_ਮਾਪ(enc, val ? 1 : 0);
    ਵਾਪਸ val;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool apir_decode_simple_ਸੂਚਕ(apir_decoder * dec) {
    ਵਾਪਸ apir_decode_array_ਮਾਪ_unchecked(dec);
}

/* uint32_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_uint32_t(apir_encoder * enc, ਅਚਲ uint32_t * val) {
    apir_encode(enc, 4, val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_uint32_t_array(apir_encoder * enc, ਅਚਲ uint32_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਅਚਲ size_t ਮਾਪ = ਆਕਾਰ(*val) * ਗਿਣਤੀ_2;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ >= ਗਿਣਤੀ_2);
    apir_encode(enc, ਮਾਪ, val, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਅਚਲ size_t ਮਾਪ = ਆਕਾਰ(*val) * ਗਿਣਤੀ_2;
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ >= ਗਿਣਤੀ_2);
    apir_decode(dec, ਮਾਪ, val, ਮਾਪ);
}

/* size_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ਮਾਪ_t(apir_encoder * enc, ਅਚਲ size_t * val) {
    ਅਚਲ uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_ਮਾਪ_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_ਮਾਪ_t_array(apir_encoder * enc, ਅਚਲ size_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਜੇ (ਆਕਾਰ(size_t) == ਆਕਾਰ(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ਅਚਲ uint64_t *) val, ਗਿਣਤੀ_2);
    } ਨਹੀਂਤਾਂ {
        ਲਈ (uint32_t i = 0; i < ਗਿਣਤੀ_2; i++) {
            apir_encode_ਮਾਪ_t(enc, &val[i]);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_ਮਾਪ_t_array(apir_decoder * dec, size_t * val, uint32_t ਗਿਣਤੀ_2) {
    ਜੇ (ਆਕਾਰ(size_t) == ਆਕਾਰ(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, ਗਿਣਤੀ_2);
    } ਨਹੀਂਤਾਂ {
        ਲਈ (uint32_t i = 0; i < ਗਿਣਤੀ_2; i++) {
            apir_decode_ਮਾਪ_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_blob_array(apir_encoder * enc, ਅਚਲ ਖਾਲੀ * val, size_t ਮਾਪ) {
    apir_encode(enc, (ਮਾਪ + 3) & ~3, val, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_blob_array(apir_decoder * dec, ਖਾਲੀ * val, size_t ਮਾਪ) {
    apir_decode(dec, (ਮਾਪ + 3) & ~3, val, ਮਾਪ);
}

/* string */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_char_array(apir_encoder * enc, ਅਚਲ ਅੱਖਰ * val, size_t ਮਾਪ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ਮਾਪ && ਲੜੀ_ਲੰਬਾਈ(val) < ਮਾਪ);
    apir_encode_blob_array(enc, val, ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_char_array(apir_decoder * dec, ਅੱਖਰ * val, size_t ਮਾਪ) {
    apir_decode_blob_array(dec, val, ਮਾਪ);
    ਜੇ (ਮਾਪ) {
        val[ਮਾਪ - 1] = '\0';
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
    }
}

/* (temp) buffer allocation */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * apir_decoder_alloc_array(size_t ਮਾਪ, size_t ਗਿਣਤੀ_2) {
    size_t alloc_ਮਾਪ;
    ਜੇ (unlikely(__builtin_mul_overflow(ਮਾਪ, ਗਿਣਤੀ_2, &alloc_ਮਾਪ))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, ਮਾਪ, ਗਿਣਤੀ_2);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ ਸਮ੍ਰਿਤੀ_ਦਿਓ(alloc_ਮਾਪ);
}

/* bool */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_bool_t(apir_encoder * enc, ਅਚਲ bool * val) {
    apir_encode(enc, ਆਕਾਰ(ਪੂਰਨ), val, ਆਕਾਰ(bool));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, ਆਕਾਰ(ਪੂਰਨ), val, ਆਕਾਰ(bool));
}

/* apir_buffer_type_host_handle_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_apir_ਬਫਰ_type_host_handle_t(apir_encoder *                         enc,
                                                              ਅਚਲ apir_ਬਫਰ_type_host_handle_t * val) {
    apir_encode(enc, ਆਕਾਰ(apir_ਬਫਰ_type_host_handle_t), val, ਆਕਾਰ(apir_ਬਫਰ_type_host_handle_t));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_apir_ਬਫਰ_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_ਬਫਰ_type_host_handle_t * val) {
    apir_decode(dec, ਆਕਾਰ(apir_ਬਫਰ_type_host_handle_t), val, ਆਕਾਰ(apir_ਬਫਰ_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_apir_ਬਫਰ_host_handle_t(apir_encoder * enc, ਅਚਲ apir_ਬਫਰ_host_handle_t * val) {
    apir_encode(enc, ਆਕਾਰ(apir_ਬਫਰ_host_handle_t), val, ਆਕਾਰ(apir_ਬਫਰ_host_handle_t));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_apir_ਬਫਰ_host_handle_t(apir_decoder * dec, apir_ਬਫਰ_host_handle_t * val) {
    apir_decode(dec, ਆਕਾਰ(apir_ਬਫਰ_host_handle_t), val, ਆਕਾਰ(apir_ਬਫਰ_host_handle_t));
}

/* uintptr_t */

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_encode_uintptr_t(apir_encoder * enc, ਅਚਲ uintptr_t * val) {
    apir_encode(enc, ਆਕਾਰ(*val), val, ਆਕਾਰ(*val));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, ਆਕਾਰ(*val), val, ਆਕਾਰ(*val));
}
