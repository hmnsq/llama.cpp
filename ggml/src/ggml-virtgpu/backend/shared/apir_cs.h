#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

ಸಂರಚನಾ apir_encoder {
    ವರ್ಣ *       cur;
    ಧ್ರುವ ವರ್ಣ * start;
    ಧ್ರುವ ವರ್ಣ * end;
    bool         fatal;
};

ಸಂರಚನಾ apir_decoder {
    ಧ್ರುವ ವರ್ಣ * cur;
    ಧ್ರುವ ವರ್ಣ * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ಸ್ಥಿರ apir_decoder apir_new_decoder(ಧ್ರುವ ವರ್ಣ * ptr, size_t size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    ಪ್ರತ್ಯಾಗಮ dec;
}

ಸ್ಥಿರ apir_encoder apir_new_encoder(ವರ್ಣ * ptr, size_t size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    ಪ್ರತ್ಯಾಗಮ enc;
}

/*
 * fatal flag handling
 */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool apir_encoder_get_fatal(ಧ್ರುವ apir_encoder * enc) {
    ಪ್ರತ್ಯಾಗಮ enc->fatal;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool apir_decoder_get_fatal(ಧ್ರುವ apir_decoder * dec) {
    ಪ್ರತ್ಯಾಗಮ dec->fatal;
}

/*
 * encode peek
 */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool apir_decoder_peek_internal(apir_decoder * dec, size_t size, ಶೂನ್ಯ * val, size_t val_size) {
    ಪ್ರತಿಜ್ಞಾ(val_size <= size);

    ಯದಿ (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_size);
        ಪ್ರತ್ಯಾಗಮ false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_size);
    ಪ್ರತ್ಯಾಗಮ true;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decoder_peek(apir_decoder * dec, size_t size, ಶೂನ್ಯ * val, size_t val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ಶೂನ್ಯ * apir_decoder_use_inplace(apir_decoder * dec, size_t size) {
    ಯದಿ (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಧ್ರುವ ಶೂನ್ಯ * addr = dec->cur;
    dec->cur += size;

    ಪ್ರತ್ಯಾಗಮ addr;
}

/*
 * read/write
 */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decoder_read(apir_decoder * dec, size_t size, ಶೂನ್ಯ * val, size_t val_size) {
    ಯದಿ (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ವರ್ಣ * apir_encoder_write(apir_encoder * enc, size_t size, ಧ್ರುವ ಶೂನ್ಯ * val, size_t val_size) {
    ಪ್ರತಿಜ್ಞಾ(val_size <= size);
    ಪ್ರತಿಜ್ಞಾ(size <= ((size_t) (enc->end - enc->cur)));

    ವರ್ಣ * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_size);
    enc->cur += size;

    ಪ್ರತ್ಯಾಗಮ write_addr;
}

/*
 * encode/decode
 */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode(apir_decoder * dec, size_t size, ಶೂನ್ಯ * data, size_t data_size) {
    ಪ್ರತಿಜ್ಞಾ(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode(apir_encoder * enc, size_t size, ಧ್ರುವ ಶೂನ್ಯ * data, size_t data_size) {
    ಪ್ರತಿಜ್ಞಾ(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_uint8_t(apir_encoder * enc, ಧ್ರುವ uint8_t * val) {
    apir_encode(enc, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), val, ಪರಿಮಾಣ(*val));
}

/* uint64_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_uint64_t(apir_encoder * enc, ಧ್ರುವ uint64_t * val) {
    apir_encode(enc, 8, val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_uint64_t_array(apir_encoder * enc, ಧ್ರುವ uint64_t * val, uint32_t ಎಣಿಕೆ) {
    ಧ್ರುವ size_t size = ಪರಿಮಾಣ(*val) * ಎಣಿಕೆ;
    ಪ್ರತಿಜ್ಞಾ(size >= ಎಣಿಕೆ);
    apir_encode(enc, size, val, size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t ಎಣಿಕೆ) {
    ಧ್ರುವ size_t size = ಪರಿಮಾಣ(*val) * ಎಣಿಕೆ;
    ಪ್ರತಿಜ್ಞಾ(size >= ಎಣಿಕೆ);
    apir_decode(dec, size, val, size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t ಎಣಿಕೆ) {
    ಪ್ರತ್ಯಾಗಮ (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, ಎಣಿಕೆ * ಪರಿಮಾಣ(uint64_t));
}

/* int32_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_int32_t(apir_encoder * enc, ಧ್ರುವ int32_t * val) {
    apir_encode(enc, 4, val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_int32_t_array(apir_encoder * enc, ಧ್ರುವ int32_t * val, uint32_t ಎಣಿಕೆ) {
    ಧ್ರುವ size_t size = ಪರಿಮಾಣ(*val) * ಎಣಿಕೆ;
    ಪ್ರತಿಜ್ಞಾ(size >= ಎಣಿಕೆ);
    apir_encode(enc, size, val, size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t ಎಣಿಕೆ) {
    ಧ್ರುವ size_t size = ಪರಿಮಾಣ(*val) * ಎಣಿಕೆ;
    ಪ್ರತಿಜ್ಞಾ(size >= ಎಣಿಕೆ);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    ಯದಿ (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    ಪ್ರತ್ಯಾಗಮ size;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    ಪ್ರತ್ಯಾಗಮ size;
}

/* non-array pointer */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool apir_encode_simple_pointer(apir_encoder * enc, ಧ್ರುವ ಶೂನ್ಯ * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    ಪ್ರತ್ಯಾಗಮ val;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool apir_decode_simple_pointer(apir_decoder * dec) {
    ಪ್ರತ್ಯಾಗಮ apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_uint32_t(apir_encoder * enc, ಧ್ರುವ uint32_t * val) {
    apir_encode(enc, 4, val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_uint32_t_array(apir_encoder * enc, ಧ್ರುವ uint32_t * val, uint32_t ಎಣಿಕೆ) {
    ಧ್ರುವ size_t size = ಪರಿಮಾಣ(*val) * ಎಣಿಕೆ;
    ಪ್ರತಿಜ್ಞಾ(size >= ಎಣಿಕೆ);
    apir_encode(enc, size, val, size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t ಎಣಿಕೆ) {
    ಧ್ರುವ size_t size = ಪರಿಮಾಣ(*val) * ಎಣಿಕೆ;
    ಪ್ರತಿಜ್ಞಾ(size >= ಎಣಿಕೆ);
    apir_decode(dec, size, val, size);
}

/* size_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_size_t(apir_encoder * enc, ಧ್ರುವ size_t * val) {
    ಧ್ರುವ uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_size_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_size_t_array(apir_encoder * enc, ಧ್ರುವ size_t * val, uint32_t ಎಣಿಕೆ) {
    ಯದಿ (ಪರಿಮಾಣ(size_t) == ಪರಿಮಾಣ(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ಧ್ರುವ uint64_t *) val, ಎಣಿಕೆ);
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (uint32_t i = 0; i < ಎಣಿಕೆ; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_size_t_array(apir_decoder * dec, size_t * val, uint32_t ಎಣಿಕೆ) {
    ಯದಿ (ಪರಿಮಾಣ(size_t) == ಪರಿಮಾಣ(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, ಎಣಿಕೆ);
    } ಅನ್ಯಥಾ {
        ಪ್ರತಿ (uint32_t i = 0; i < ಎಣಿಕೆ; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_blob_array(apir_encoder * enc, ಧ್ರುವ ಶೂನ್ಯ * val, size_t size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_blob_array(apir_decoder * dec, ಶೂನ್ಯ * val, size_t size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_char_array(apir_encoder * enc, ಧ್ರುವ ವರ್ಣ * val, size_t size) {
    ಪ್ರತಿಜ್ಞಾ(size && strlen(val) < size);
    apir_encode_blob_array(enc, val, size);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_char_array(apir_decoder * dec, ವರ್ಣ * val, size_t size) {
    apir_decode_blob_array(dec, val, size);
    ಯದಿ (size) {
        val[size - 1] = '\0';
    } ಅನ್ಯಥಾ {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * apir_decoder_alloc_array(size_t size, size_t ಎಣಿಕೆ) {
    size_t alloc_size;
    ಯದಿ (unlikely(__builtin_mul_overflow(size, ಎಣಿಕೆ, &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, ಎಣಿಕೆ);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ malloc(alloc_size);
}

/* bool */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_bool_t(apir_encoder * enc, ಧ್ರುವ bool * val) {
    apir_encode(enc, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), val, ಪರಿಮಾಣ(bool));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), val, ಪರಿಮಾಣ(bool));
}

/* apir_buffer_type_host_handle_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              ಧ್ರುವ apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, ಪರಿಮಾಣ(apir_buffer_type_host_handle_t), val, ಪರಿಮಾಣ(apir_buffer_type_host_handle_t));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, ಪರಿಮಾಣ(apir_buffer_type_host_handle_t), val, ಪರಿಮಾಣ(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, ಧ್ರುವ apir_buffer_host_handle_t * val) {
    apir_encode(enc, ಪರಿಮಾಣ(apir_buffer_host_handle_t), val, ಪರಿಮಾಣ(apir_buffer_host_handle_t));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, ಪರಿಮಾಣ(apir_buffer_host_handle_t), val, ಪರಿಮಾಣ(apir_buffer_host_handle_t));
}

/* uintptr_t */

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_encode_uintptr_t(apir_encoder * enc, ಧ್ರುವ uintptr_t * val) {
    apir_encode(enc, ಪರಿಮಾಣ(*val), val, ಪರಿಮಾಣ(*val));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, ಪರಿಮಾಣ(*val), val, ಪರಿಮಾಣ(*val));
}
