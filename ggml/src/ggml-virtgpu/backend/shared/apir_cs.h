#pragma once

#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

સંરચના apir_encoder {
    અક્ષર *       cur;
    અચલ અક્ષર * start;
    અચલ અક્ષર * end;
    bool         fatal;
};

સંરચના apir_decoder {
    અચલ અક્ષર * cur;
    અચલ અક્ષર * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

સ્થિર apir_decoder apir_new_decoder(અચલ અક્ષર * ptr, size_t size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    પાછા dec;
}

સ્થિર apir_encoder apir_new_encoder(અક્ષર * ptr, size_t size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    પાછા enc;
}

/*
 * fatal flag handling
 */

સ્થિર અંતઃરેખ શૂન્ય apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

સ્થિર અંતઃરેખ શૂન્ય apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

સ્થિર અંતઃરેખ bool apir_encoder_get_fatal(અચલ apir_encoder * enc) {
    પાછા enc->fatal;
}

સ્થિર અંતઃરેખ શૂન્ય apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

સ્થિર અંતઃરેખ શૂન્ય apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

સ્થિર અંતઃરેખ bool apir_decoder_get_fatal(અચલ apir_decoder * dec) {
    પાછા dec->fatal;
}

/*
 * encode peek
 */

સ્થિર અંતઃરેખ bool apir_decoder_peek_internal(apir_decoder * dec, size_t size, શૂન્ય * val, size_t val_size) {
    પ્રતિજ્ઞા(val_size <= size);

    જો (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_size);
        પાછા false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_size);
    પાછા true;
}

સ્થિર અંતઃરેખ શૂન્ય apir_decoder_peek(apir_decoder * dec, size_t size, શૂન્ય * val, size_t val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

સ્થિર અંતઃરેખ અચલ શૂન્ય * apir_decoder_use_inplace(apir_decoder * dec, size_t size) {
    જો (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        પાછા ખાલી;
    }
    અચલ શૂન્ય * addr = dec->cur;
    dec->cur += size;

    પાછા addr;
}

/*
 * read/write
 */

સ્થિર અંતઃરેખ શૂન્ય apir_decoder_read(apir_decoder * dec, size_t size, શૂન્ય * val, size_t val_size) {
    જો (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

સ્થિર અંતઃરેખ અક્ષર * apir_encoder_write(apir_encoder * enc, size_t size, અચલ શૂન્ય * val, size_t val_size) {
    પ્રતિજ્ઞા(val_size <= size);
    પ્રતિજ્ઞા(size <= ((size_t) (enc->end - enc->cur)));

    અક્ષર * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_size);
    enc->cur += size;

    પાછા write_addr;
}

/*
 * encode/decode
 */

સ્થિર અંતઃરેખ શૂન્ય apir_decode(apir_decoder * dec, size_t size, શૂન્ય * data, size_t data_size) {
    પ્રતિજ્ઞા(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

સ્થિર અંતઃરેખ શૂન્ય apir_encode(apir_encoder * enc, size_t size, અચલ શૂન્ય * data, size_t data_size) {
    પ્રતિજ્ઞા(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_uint8_t(apir_encoder * enc, અચલ uint8_t * val) {
    apir_encode(enc, કદ(પૂર્ણાંક), val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, કદ(પૂર્ણાંક), val, કદ(*val));
}

/* uint64_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_uint64_t(apir_encoder * enc, અચલ uint64_t * val) {
    apir_encode(enc, 8, val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_encode_uint64_t_array(apir_encoder * enc, અચલ uint64_t * val, uint32_t ગણતરી) {
    અચલ size_t size = કદ(*val) * ગણતરી;
    પ્રતિજ્ઞા(size >= ગણતરી);
    apir_encode(enc, size, val, size);
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t ગણતરી) {
    અચલ size_t size = કદ(*val) * ગણતરી;
    પ્રતિજ્ઞા(size >= ગણતરી);
    apir_decode(dec, size, val, size);
}

સ્થિર અંતઃરેખ અચલ uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t ગણતરી) {
    પાછા (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, ગણતરી * કદ(uint64_t));
}

/* int32_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_int32_t(apir_encoder * enc, અચલ int32_t * val) {
    apir_encode(enc, 4, val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_encode_int32_t_array(apir_encoder * enc, અચલ int32_t * val, uint32_t ગણતરી) {
    અચલ size_t size = કદ(*val) * ગણતરી;
    પ્રતિજ્ઞા(size >= ગણતરી);
    apir_encode(enc, size, val, size);
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t ગણતરી) {
    અચલ size_t size = કદ(*val) * ગણતરી;
    પ્રતિજ્ઞા(size >= ગણતરી);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

સ્થિર અંતઃરેખ uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    જો (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    પાછા size;
}

સ્થિર અંતઃરેખ uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    પાછા size;
}

/* non-array pointer */

સ્થિર અંતઃરેખ bool apir_encode_simple_pointer(apir_encoder * enc, અચલ શૂન્ય * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    પાછા val;
}

સ્થિર અંતઃરેખ bool apir_decode_simple_pointer(apir_decoder * dec) {
    પાછા apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_uint32_t(apir_encoder * enc, અચલ uint32_t * val) {
    apir_encode(enc, 4, val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_encode_uint32_t_array(apir_encoder * enc, અચલ uint32_t * val, uint32_t ગણતરી) {
    અચલ size_t size = કદ(*val) * ગણતરી;
    પ્રતિજ્ઞા(size >= ગણતરી);
    apir_encode(enc, size, val, size);
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t ગણતરી) {
    અચલ size_t size = કદ(*val) * ગણતરી;
    પ્રતિજ્ઞા(size >= ગણતરી);
    apir_decode(dec, size, val, size);
}

/* size_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_size_t(apir_encoder * enc, અચલ size_t * val) {
    અચલ uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_size_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

સ્થિર અંતઃરેખ શૂન્ય apir_encode_size_t_array(apir_encoder * enc, અચલ size_t * val, uint32_t ગણતરી) {
    જો (કદ(size_t) == કદ(uint64_t)) {
        apir_encode_uint64_t_array(enc, (અચલ uint64_t *) val, ગણતરી);
    } નહીંતર {
        માટે (uint32_t i = 0; i < ગણતરી; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_size_t_array(apir_decoder * dec, size_t * val, uint32_t ગણતરી) {
    જો (કદ(size_t) == કદ(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, ગણતરી);
    } નહીંતર {
        માટે (uint32_t i = 0; i < ગણતરી; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_blob_array(apir_encoder * enc, અચલ શૂન્ય * val, size_t size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_blob_array(apir_decoder * dec, શૂન્ય * val, size_t size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_char_array(apir_encoder * enc, અચલ અક્ષર * val, size_t size) {
    પ્રતિજ્ઞા(size && strlen(val) < size);
    apir_encode_blob_array(enc, val, size);
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_char_array(apir_decoder * dec, અક્ષર * val, size_t size) {
    apir_decode_blob_array(dec, val, size);
    જો (size) {
        val[size - 1] = '\0';
    } નહીંતર {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

સ્થિર અંતઃરેખ શૂન્ય * apir_decoder_alloc_array(size_t size, size_t ગણતરી) {
    size_t alloc_size;
    જો (unlikely(__builtin_mul_overflow(size, ગણતરી, &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, ગણતરી);
        પાછા ખાલી;
    }

    પાછા malloc(alloc_size);
}

/* bool */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_bool_t(apir_encoder * enc, અચલ bool * val) {
    apir_encode(enc, કદ(પૂર્ણાંક), val, કદ(bool));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, કદ(પૂર્ણાંક), val, કદ(bool));
}

/* apir_buffer_type_host_handle_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              અચલ apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, કદ(apir_buffer_type_host_handle_t), val, કદ(apir_buffer_type_host_handle_t));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, કદ(apir_buffer_type_host_handle_t), val, કદ(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, અચલ apir_buffer_host_handle_t * val) {
    apir_encode(enc, કદ(apir_buffer_host_handle_t), val, કદ(apir_buffer_host_handle_t));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, કદ(apir_buffer_host_handle_t), val, કદ(apir_buffer_host_handle_t));
}

/* uintptr_t */

સ્થિર અંતઃરેખ શૂન્ય apir_encode_uintptr_t(apir_encoder * enc, અચલ uintptr_t * val) {
    apir_encode(enc, કદ(*val), val, કદ(*val));
}

સ્થિર અંતઃરેખ શૂન્ય apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, કદ(*val), val, કદ(*val));
}
