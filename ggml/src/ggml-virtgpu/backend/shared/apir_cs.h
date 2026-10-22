#pragma once

#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cassert>
#അന്തര്ഭാവ <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

ഘടന apir_encoder {
    അക്ഷരം *       cur;
    സ്ഥിരം അക്ഷരം * start;
    സ്ഥിരം അക്ഷരം * end;
    bool         fatal;
};

ഘടന apir_decoder {
    സ്ഥിരം അക്ഷരം * cur;
    സ്ഥിരം അക്ഷരം * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

സ്ഥിതം apir_decoder apir_new_decoder(സ്ഥിരം അക്ഷരം * ptr, size_t size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    തിരികെ dec;
}

സ്ഥിതം apir_encoder apir_new_encoder(അക്ഷരം * ptr, size_t size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    തിരികെ enc;
}

/*
 * fatal flag handling
 */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

സ്ഥിതം വരിയിൽ bool apir_encoder_get_fatal(സ്ഥിരം apir_encoder * enc) {
    തിരികെ enc->fatal;
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

സ്ഥിതം വരിയിൽ bool apir_decoder_get_fatal(സ്ഥിരം apir_decoder * dec) {
    തിരികെ dec->fatal;
}

/*
 * encode peek
 */

സ്ഥിതം വരിയിൽ bool apir_decoder_peek_internal(apir_decoder * dec, size_t size, ശൂന്യം * val, size_t val_size) {
    പ്രതിജ്ഞാ(val_size <= size);

    എങ്കിൽ (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_size);
        തിരികെ false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_size);
    തിരികെ true;
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decoder_peek(apir_decoder * dec, size_t size, ശൂന്യം * val, size_t val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

സ്ഥിതം വരിയിൽ സ്ഥിരം ശൂന്യം * apir_decoder_use_inplace(apir_decoder * dec, size_t size) {
    എങ്കിൽ (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        തിരികെ ശൂന്യ;
    }
    സ്ഥിരം ശൂന്യം * addr = dec->cur;
    dec->cur += size;

    തിരികെ addr;
}

/*
 * read/write
 */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decoder_read(apir_decoder * dec, size_t size, ശൂന്യം * val, size_t val_size) {
    എങ്കിൽ (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

സ്ഥിതം വരിയിൽ അക്ഷരം * apir_encoder_write(apir_encoder * enc, size_t size, സ്ഥിരം ശൂന്യം * val, size_t val_size) {
    പ്രതിജ്ഞാ(val_size <= size);
    പ്രതിജ്ഞാ(size <= ((size_t) (enc->end - enc->cur)));

    അക്ഷരം * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_size);
    enc->cur += size;

    തിരികെ write_addr;
}

/*
 * encode/decode
 */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode(apir_decoder * dec, size_t size, ശൂന്യം * data, size_t data_size) {
    പ്രതിജ്ഞാ(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode(apir_encoder * enc, size_t size, സ്ഥിരം ശൂന്യം * data, size_t data_size) {
    പ്രതിജ്ഞാ(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_uint8_t(apir_encoder * enc, സ്ഥിരം uint8_t * val) {
    apir_encode(enc, വലുപ്പം(പൂർണ്ണസംഖ്യ), val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, വലുപ്പം(പൂർണ്ണസംഖ്യ), val, വലുപ്പം(*val));
}

/* uint64_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_uint64_t(apir_encoder * enc, സ്ഥിരം uint64_t * val) {
    apir_encode(enc, 8, val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_uint64_t_array(apir_encoder * enc, സ്ഥിരം uint64_t * val, uint32_t എണ്ണം) {
    സ്ഥിരം size_t size = വലുപ്പം(*val) * എണ്ണം;
    പ്രതിജ്ഞാ(size >= എണ്ണം);
    apir_encode(enc, size, val, size);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t എണ്ണം) {
    സ്ഥിരം size_t size = വലുപ്പം(*val) * എണ്ണം;
    പ്രതിജ്ഞാ(size >= എണ്ണം);
    apir_decode(dec, size, val, size);
}

സ്ഥിതം വരിയിൽ സ്ഥിരം uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t എണ്ണം) {
    തിരികെ (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, എണ്ണം * വലുപ്പം(uint64_t));
}

/* int32_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_int32_t(apir_encoder * enc, സ്ഥിരം int32_t * val) {
    apir_encode(enc, 4, val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_int32_t_array(apir_encoder * enc, സ്ഥിരം int32_t * val, uint32_t എണ്ണം) {
    സ്ഥിരം size_t size = വലുപ്പം(*val) * എണ്ണം;
    പ്രതിജ്ഞാ(size >= എണ്ണം);
    apir_encode(enc, size, val, size);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t എണ്ണം) {
    സ്ഥിരം size_t size = വലുപ്പം(*val) * എണ്ണം;
    പ്രതിജ്ഞാ(size >= എണ്ണം);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

സ്ഥിതം വരിയിൽ uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    എങ്കിൽ (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    തിരികെ size;
}

സ്ഥിതം വരിയിൽ uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    തിരികെ size;
}

/* non-array pointer */

സ്ഥിതം വരിയിൽ bool apir_encode_simple_pointer(apir_encoder * enc, സ്ഥിരം ശൂന്യം * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    തിരികെ val;
}

സ്ഥിതം വരിയിൽ bool apir_decode_simple_pointer(apir_decoder * dec) {
    തിരികെ apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_uint32_t(apir_encoder * enc, സ്ഥിരം uint32_t * val) {
    apir_encode(enc, 4, val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_uint32_t_array(apir_encoder * enc, സ്ഥിരം uint32_t * val, uint32_t എണ്ണം) {
    സ്ഥിരം size_t size = വലുപ്പം(*val) * എണ്ണം;
    പ്രതിജ്ഞാ(size >= എണ്ണം);
    apir_encode(enc, size, val, size);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t എണ്ണം) {
    സ്ഥിരം size_t size = വലുപ്പം(*val) * എണ്ണം;
    പ്രതിജ്ഞാ(size >= എണ്ണം);
    apir_decode(dec, size, val, size);
}

/* size_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_size_t(apir_encoder * enc, സ്ഥിരം size_t * val) {
    സ്ഥിരം uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_size_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_size_t_array(apir_encoder * enc, സ്ഥിരം size_t * val, uint32_t എണ്ണം) {
    എങ്കിൽ (വലുപ്പം(size_t) == വലുപ്പം(uint64_t)) {
        apir_encode_uint64_t_array(enc, (സ്ഥിരം uint64_t *) val, എണ്ണം);
    } അല്ലെങ്കിൽ {
        വേണ്ടി (uint32_t i = 0; i < എണ്ണം; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_size_t_array(apir_decoder * dec, size_t * val, uint32_t എണ്ണം) {
    എങ്കിൽ (വലുപ്പം(size_t) == വലുപ്പം(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, എണ്ണം);
    } അല്ലെങ്കിൽ {
        വേണ്ടി (uint32_t i = 0; i < എണ്ണം; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_blob_array(apir_encoder * enc, സ്ഥിരം ശൂന്യം * val, size_t size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_blob_array(apir_decoder * dec, ശൂന്യം * val, size_t size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_char_array(apir_encoder * enc, സ്ഥിരം അക്ഷരം * val, size_t size) {
    പ്രതിജ്ഞാ(size && strlen(val) < size);
    apir_encode_blob_array(enc, val, size);
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_char_array(apir_decoder * dec, അക്ഷരം * val, size_t size) {
    apir_decode_blob_array(dec, val, size);
    എങ്കിൽ (size) {
        val[size - 1] = '\0';
    } അല്ലെങ്കിൽ {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

സ്ഥിതം വരിയിൽ ശൂന്യം * apir_decoder_alloc_array(size_t size, size_t എണ്ണം) {
    size_t alloc_size;
    എങ്കിൽ (unlikely(__builtin_mul_overflow(size, എണ്ണം, &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, എണ്ണം);
        തിരികെ ശൂന്യ;
    }

    തിരികെ malloc(alloc_size);
}

/* bool */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_bool_t(apir_encoder * enc, സ്ഥിരം bool * val) {
    apir_encode(enc, വലുപ്പം(പൂർണ്ണസംഖ്യ), val, വലുപ്പം(bool));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, വലുപ്പം(പൂർണ്ണസംഖ്യ), val, വലുപ്പം(bool));
}

/* apir_buffer_type_host_handle_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              സ്ഥിരം apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, വലുപ്പം(apir_buffer_type_host_handle_t), val, വലുപ്പം(apir_buffer_type_host_handle_t));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, വലുപ്പം(apir_buffer_type_host_handle_t), val, വലുപ്പം(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, സ്ഥിരം apir_buffer_host_handle_t * val) {
    apir_encode(enc, വലുപ്പം(apir_buffer_host_handle_t), val, വലുപ്പം(apir_buffer_host_handle_t));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, വലുപ്പം(apir_buffer_host_handle_t), val, വലുപ്പം(apir_buffer_host_handle_t));
}

/* uintptr_t */

സ്ഥിതം വരിയിൽ ശൂന്യം apir_encode_uintptr_t(apir_encoder * enc, സ്ഥിരം uintptr_t * val) {
    apir_encode(enc, വലുപ്പം(*val), val, വലുപ്പം(*val));
}

സ്ഥിതം വരിയിൽ ശൂന്യം apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, വലുപ്പം(*val), val, വലുപ്പം(*val));
}
