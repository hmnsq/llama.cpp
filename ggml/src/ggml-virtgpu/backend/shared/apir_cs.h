#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <cassert>
#ଅନ୍ତର୍ଭାଵ <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

ସଂରଚନା apir_encoder {
    ଅକ୍ଷର *       cur;
    ଅଚଳ ଅକ୍ଷର * start;
    ଅଚଳ ଅକ୍ଷର * end;
    bool         fatal;
};

ସଂରଚନା apir_decoder {
    ଅଚଳ ଅକ୍ଷର * cur;
    ଅଚଳ ଅକ୍ଷର * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

ସ୍ଥିର apir_decoder apir_new_decoder(ଅଚଳ ଅକ୍ଷର * ptr, size_t size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    ଫେରନ୍ତୁ dec;
}

ସ୍ଥିର apir_encoder apir_new_encoder(ଅକ୍ଷର * ptr, size_t size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    ଫେରନ୍ତୁ enc;
}

/*
 * fatal flag handling
 */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool apir_encoder_get_fatal(ଅଚଳ apir_encoder * enc) {
    ଫେରନ୍ତୁ enc->fatal;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool apir_decoder_get_fatal(ଅଚଳ apir_decoder * dec) {
    ଫେରନ୍ତୁ dec->fatal;
}

/*
 * encode peek
 */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool apir_decoder_peek_internal(apir_decoder * dec, size_t size, ଶୂନ୍ୟ * val, size_t val_size) {
    ପ୍ରତିଜ୍ଞା(val_size <= size);

    ଯଦି (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_size);
        ଫେରନ୍ତୁ false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_size);
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decoder_peek(apir_decoder * dec, size_t size, ଶୂନ୍ୟ * val, size_t val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଶୂନ୍ୟ * apir_decoder_use_inplace(apir_decoder * dec, size_t size) {
    ଯଦି (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଅଚଳ ଶୂନ୍ୟ * addr = dec->cur;
    dec->cur += size;

    ଫେରନ୍ତୁ addr;
}

/*
 * read/write
 */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decoder_read(apir_decoder * dec, size_t size, ଶୂନ୍ୟ * val, size_t val_size) {
    ଯଦି (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅକ୍ଷର * apir_encoder_write(apir_encoder * enc, size_t size, ଅଚଳ ଶୂନ୍ୟ * val, size_t val_size) {
    ପ୍ରତିଜ୍ଞା(val_size <= size);
    ପ୍ରତିଜ୍ଞା(size <= ((size_t) (enc->end - enc->cur)));

    ଅକ୍ଷର * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_size);
    enc->cur += size;

    ଫେରନ୍ତୁ write_addr;
}

/*
 * encode/decode
 */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode(apir_decoder * dec, size_t size, ଶୂନ୍ୟ * data, size_t data_size) {
    ପ୍ରତିଜ୍ଞା(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode(apir_encoder * enc, size_t size, ଅଚଳ ଶୂନ୍ୟ * data, size_t data_size) {
    ପ୍ରତିଜ୍ଞା(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_uint8_t(apir_encoder * enc, ଅଚଳ uint8_t * val) {
    apir_encode(enc, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), val, ଆକାର(*val));
}

/* uint64_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_uint64_t(apir_encoder * enc, ଅଚଳ uint64_t * val) {
    apir_encode(enc, 8, val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_uint64_t_array(apir_encoder * enc, ଅଚଳ uint64_t * val, uint32_t ଗଣନା_2) {
    ଅଚଳ size_t size = ଆକାର(*val) * ଗଣନା_2;
    ପ୍ରତିଜ୍ଞା(size >= ଗଣନା_2);
    apir_encode(enc, size, val, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t ଗଣନା_2) {
    ଅଚଳ size_t size = ଆକାର(*val) * ଗଣନା_2;
    ପ୍ରତିଜ୍ଞା(size >= ଗଣନା_2);
    apir_decode(dec, size, val, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t ଗଣନା_2) {
    ଫେରନ୍ତୁ (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, ଗଣନା_2 * ଆକାର(uint64_t));
}

/* int32_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_int32_t(apir_encoder * enc, ଅଚଳ int32_t * val) {
    apir_encode(enc, 4, val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_int32_t_array(apir_encoder * enc, ଅଚଳ int32_t * val, uint32_t ଗଣନା_2) {
    ଅଚଳ size_t size = ଆକାର(*val) * ଗଣନା_2;
    ପ୍ରତିଜ୍ଞା(size >= ଗଣନା_2);
    apir_encode(enc, size, val, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t ଗଣନା_2) {
    ଅଚଳ size_t size = ଆକାର(*val) * ଗଣନା_2;
    ପ୍ରତିଜ୍ଞା(size >= ଗଣନା_2);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    ଯଦି (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    ଫେରନ୍ତୁ size;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    ଫେରନ୍ତୁ size;
}

/* non-array pointer */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool apir_encode_simple_pointer(apir_encoder * enc, ଅଚଳ ଶୂନ୍ୟ * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    ଫେରନ୍ତୁ val;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool apir_decode_simple_pointer(apir_decoder * dec) {
    ଫେରନ୍ତୁ apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_uint32_t(apir_encoder * enc, ଅଚଳ uint32_t * val) {
    apir_encode(enc, 4, val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_uint32_t_array(apir_encoder * enc, ଅଚଳ uint32_t * val, uint32_t ଗଣନା_2) {
    ଅଚଳ size_t size = ଆକାର(*val) * ଗଣନା_2;
    ପ୍ରତିଜ୍ଞା(size >= ଗଣନା_2);
    apir_encode(enc, size, val, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t ଗଣନା_2) {
    ଅଚଳ size_t size = ଆକାର(*val) * ଗଣନା_2;
    ପ୍ରତିଜ୍ଞା(size >= ଗଣନା_2);
    apir_decode(dec, size, val, size);
}

/* size_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_size_t(apir_encoder * enc, ଅଚଳ size_t * val) {
    ଅଚଳ uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_size_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_size_t_array(apir_encoder * enc, ଅଚଳ size_t * val, uint32_t ଗଣନା_2) {
    ଯଦି (ଆକାର(size_t) == ଆକାର(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ଅଚଳ uint64_t *) val, ଗଣନା_2);
    } ଅନ୍ୟଥା {
        ପାଇଁ (uint32_t i = 0; i < ଗଣନା_2; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_size_t_array(apir_decoder * dec, size_t * val, uint32_t ଗଣନା_2) {
    ଯଦି (ଆକାର(size_t) == ଆକାର(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, ଗଣନା_2);
    } ଅନ୍ୟଥା {
        ପାଇଁ (uint32_t i = 0; i < ଗଣନା_2; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_blob_array(apir_encoder * enc, ଅଚଳ ଶୂନ୍ୟ * val, size_t size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_blob_array(apir_decoder * dec, ଶୂନ୍ୟ * val, size_t size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_char_array(apir_encoder * enc, ଅଚଳ ଅକ୍ଷର * val, size_t size) {
    ପ୍ରତିଜ୍ଞା(size && strlen(val) < size);
    apir_encode_blob_array(enc, val, size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_char_array(apir_decoder * dec, ଅକ୍ଷର * val, size_t size) {
    apir_decode_blob_array(dec, val, size);
    ଯଦି (size) {
        val[size - 1] = '\0';
    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * apir_decoder_alloc_array(size_t size, size_t ଗଣନା_2) {
    size_t alloc_size;
    ଯଦି (unlikely(__builtin_mul_overflow(size, ଗଣନା_2, &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, ଗଣନା_2);
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଫେରନ୍ତୁ malloc(alloc_size);
}

/* bool */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_bool_t(apir_encoder * enc, ଅଚଳ bool * val) {
    apir_encode(enc, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), val, ଆକାର(bool));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), val, ଆକାର(bool));
}

/* apir_buffer_type_host_handle_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              ଅଚଳ apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, ଆକାର(apir_buffer_type_host_handle_t), val, ଆକାର(apir_buffer_type_host_handle_t));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, ଆକାର(apir_buffer_type_host_handle_t), val, ଆକାର(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, ଅଚଳ apir_buffer_host_handle_t * val) {
    apir_encode(enc, ଆକାର(apir_buffer_host_handle_t), val, ଆକାର(apir_buffer_host_handle_t));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, ଆକାର(apir_buffer_host_handle_t), val, ଆକାର(apir_buffer_host_handle_t));
}

/* uintptr_t */

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_encode_uintptr_t(apir_encoder * enc, ଅଚଳ uintptr_t * val) {
    apir_encode(enc, ଆକାର(*val), val, ଆକାର(*val));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, ଆକାର(*val), val, ଆକାର(*val));
}
