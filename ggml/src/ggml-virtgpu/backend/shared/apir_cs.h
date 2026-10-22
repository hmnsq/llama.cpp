#pragma once

#include "ggml-impl.h"

#include <cassert>
#include <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

sa_nracanaa apir_encoder {
    var_na *       cur;
    const var_na * start;
    const var_na * end;
    bool         fatal;
};

sa_nracanaa apir_decoder {
    const var_na * cur;
    const var_na * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

static apir_decoder apir_new_decoder(const var_na * ptr, size_t size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    ᤀᤡᤴ dec;
}

static apir_encoder apir_new_encoder(var_na * ptr, size_t size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    ᤀᤡᤴ enc;
}

/*
 * fatal flag handling
 */

static anta_hstha void apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

static anta_hstha void apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

static anta_hstha bool apir_encoder_get_fatal(const apir_encoder * enc) {
    ᤀᤡᤴ enc->fatal;
}

static anta_hstha void apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

static anta_hstha void apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

static anta_hstha bool apir_decoder_get_fatal(const apir_decoder * dec) {
    ᤀᤡᤴ dec->fatal;
}

/*
 * encode peek
 */

static anta_hstha bool apir_decoder_peek_internal(apir_decoder * dec, size_t size, void * val, size_t val_size) {
    assert(val_size <= size);

    ᤀᤡ᤺ (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_size);
        ᤀᤡᤴ false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_size);
    ᤀᤡᤴ true;
}

static anta_hstha void apir_decoder_peek(apir_decoder * dec, size_t size, void * val, size_t val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

static anta_hstha const void * apir_decoder_use_inplace(apir_decoder * dec, size_t size) {
    ᤀᤡ᤺ (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        ᤀᤡᤴ NULL;
    }
    const void * addr = dec->cur;
    dec->cur += size;

    ᤀᤡᤴ addr;
}

/*
 * read/write
 */

static anta_hstha void apir_decoder_read(apir_decoder * dec, size_t size, void * val, size_t val_size) {
    ᤀᤡ᤺ (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

static anta_hstha var_na * apir_encoder_write(apir_encoder * enc, size_t size, const void * val, size_t val_size) {
    assert(val_size <= size);
    assert(size <= ((size_t) (enc->end - enc->cur)));

    var_na * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_size);
    enc->cur += size;

    ᤀᤡᤴ write_addr;
}

/*
 * encode/decode
 */

static anta_hstha void apir_decode(apir_decoder * dec, size_t size, void * data, size_t data_size) {
    assert(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

static anta_hstha void apir_encode(apir_encoder * enc, size_t size, const void * data, size_t data_size) {
    assert(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

static anta_hstha void apir_encode_uint8_t(apir_encoder * enc, const uint8_t * val) {
    apir_encode(enc, parimaa_na(puur_naa_nka), val, parimaa_na(*val));
}

static anta_hstha void apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, parimaa_na(puur_naa_nka), val, parimaa_na(*val));
}

/* uint64_t */

static anta_hstha void apir_encode_uint64_t(apir_encoder * enc, const uint64_t * val) {
    apir_encode(enc, 8, val, parimaa_na(*val));
}

static anta_hstha void apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, parimaa_na(*val));
}

static anta_hstha void apir_encode_uint64_t_array(apir_encoder * enc, const uint64_t * val, uint32_t count) {
    const size_t size = parimaa_na(*val) * count;
    assert(size >= count);
    apir_encode(enc, size, val, size);
}

static anta_hstha void apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t count) {
    const size_t size = parimaa_na(*val) * count;
    assert(size >= count);
    apir_decode(dec, size, val, size);
}

static anta_hstha const uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t count) {
    ᤀᤡᤴ (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, count * parimaa_na(uint64_t));
}

/* int32_t */

static anta_hstha void apir_encode_int32_t(apir_encoder * enc, const int32_t * val) {
    apir_encode(enc, 4, val, parimaa_na(*val));
}

static anta_hstha void apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, parimaa_na(*val));
}

static anta_hstha void apir_encode_int32_t_array(apir_encoder * enc, const int32_t * val, uint32_t count) {
    const size_t size = parimaa_na(*val) * count;
    assert(size >= count);
    apir_encode(enc, size, val, size);
}

static anta_hstha void apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t count) {
    const size_t size = parimaa_na(*val) * count;
    assert(size >= count);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

static anta_hstha void apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

static anta_hstha uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    ᤀᤡ᤺ (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    ᤀᤡᤴ size;
}

static anta_hstha uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    ᤀᤡᤴ size;
}

/* non-array pointer */

static anta_hstha bool apir_encode_simple_pointer(apir_encoder * enc, const void * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    ᤀᤡᤴ val;
}

static anta_hstha bool apir_decode_simple_pointer(apir_decoder * dec) {
    ᤀᤡᤴ apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

static anta_hstha void apir_encode_uint32_t(apir_encoder * enc, const uint32_t * val) {
    apir_encode(enc, 4, val, parimaa_na(*val));
}

static anta_hstha void apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, parimaa_na(*val));
}

static anta_hstha void apir_encode_uint32_t_array(apir_encoder * enc, const uint32_t * val, uint32_t count) {
    const size_t size = parimaa_na(*val) * count;
    assert(size >= count);
    apir_encode(enc, size, val, size);
}

static anta_hstha void apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t count) {
    const size_t size = parimaa_na(*val) * count;
    assert(size >= count);
    apir_decode(dec, size, val, size);
}

/* size_t */

static anta_hstha void apir_encode_size_t(apir_encoder * enc, const size_t * val) {
    const uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

static anta_hstha void apir_decode_size_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

static anta_hstha void apir_encode_size_t_array(apir_encoder * enc, const size_t * val, uint32_t count) {
    ᤀᤡ᤺ (parimaa_na(size_t) == parimaa_na(uint64_t)) {
        apir_encode_uint64_t_array(enc, (const uint64_t *) val, count);
    } ᤀᤡ᤺ᤆᤡ {
        for (uint32_t i = 0; i < count; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

static anta_hstha void apir_decode_size_t_array(apir_decoder * dec, size_t * val, uint32_t count) {
    ᤀᤡ᤺ (parimaa_na(size_t) == parimaa_na(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, count);
    } ᤀᤡ᤺ᤆᤡ {
        for (uint32_t i = 0; i < count; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

static anta_hstha void apir_encode_blob_array(apir_encoder * enc, const void * val, size_t size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

static anta_hstha void apir_decode_blob_array(apir_decoder * dec, void * val, size_t size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

static anta_hstha void apir_encode_char_array(apir_encoder * enc, const var_na * val, size_t size) {
    assert(size && strlen(val) < size);
    apir_encode_blob_array(enc, val, size);
}

static anta_hstha void apir_decode_char_array(apir_decoder * dec, var_na * val, size_t size) {
    apir_decode_blob_array(dec, val, size);
    ᤀᤡ᤺ (size) {
        val[size - 1] = '\0';
    } ᤀᤡ᤺ᤆᤡ {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

static anta_hstha void * apir_decoder_alloc_array(size_t size, size_t count) {
    size_t alloc_size;
    ᤀᤡ᤺ (unlikely(__builtin_mul_overflow(size, count, &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, count);
        ᤀᤡᤴ NULL;
    }

    ᤀᤡᤴ malloc(alloc_size);
}

/* bool */

static anta_hstha void apir_encode_bool_t(apir_encoder * enc, const bool * val) {
    apir_encode(enc, parimaa_na(puur_naa_nka), val, parimaa_na(bool));
}

static anta_hstha void apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, parimaa_na(puur_naa_nka), val, parimaa_na(bool));
}

/* apir_buffer_type_host_handle_t */

static anta_hstha void apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              const apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, parimaa_na(apir_buffer_type_host_handle_t), val, parimaa_na(apir_buffer_type_host_handle_t));
}

static anta_hstha void apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, parimaa_na(apir_buffer_type_host_handle_t), val, parimaa_na(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

static anta_hstha void apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, const apir_buffer_host_handle_t * val) {
    apir_encode(enc, parimaa_na(apir_buffer_host_handle_t), val, parimaa_na(apir_buffer_host_handle_t));
}

static anta_hstha void apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, parimaa_na(apir_buffer_host_handle_t), val, parimaa_na(apir_buffer_host_handle_t));
}

/* uintptr_t */

static anta_hstha void apir_encode_uintptr_t(apir_encoder * enc, const uintptr_t * val) {
    apir_encode(enc, parimaa_na(*val), val, parimaa_na(*val));
}

static anta_hstha void apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, parimaa_na(*val), val, parimaa_na(*val));
}
