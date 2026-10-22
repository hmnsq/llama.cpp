#pragma once

#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

கட்டமைப்பு apir_encoder {
    எழுத்து *       cur;
    மாறிலி எழுத்து * start;
    மாறிலி எழுத்து * end;
    bool         fatal;
};

கட்டமைப்பு apir_decoder {
    மாறிலி எழுத்து * cur;
    மாறிலி எழுத்து * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

நிலையான apir_decoder apir_new_decoder(மாறிலி எழுத்து * ptr, size_t size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    திரும்பு dec;
}

நிலையான apir_encoder apir_new_encoder(எழுத்து * ptr, size_t size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    திரும்பு enc;
}

/*
 * fatal flag handling
 */

நிலையான உள்வரி வெறுமை apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

நிலையான உள்வரி வெறுமை apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

நிலையான உள்வரி bool apir_encoder_get_fatal(மாறிலி apir_encoder * enc) {
    திரும்பு enc->fatal;
}

நிலையான உள்வரி வெறுமை apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

நிலையான உள்வரி வெறுமை apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

நிலையான உள்வரி bool apir_decoder_get_fatal(மாறிலி apir_decoder * dec) {
    திரும்பு dec->fatal;
}

/*
 * encode peek
 */

நிலையான உள்வரி bool apir_decoder_peek_internal(apir_decoder * dec, size_t size, வெறுமை * val, size_t val_size) {
    ப்ரதிஜ்ஞா(val_size <= size);

    எனில் (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_size);
        திரும்பு false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_size);
    திரும்பு true;
}

நிலையான உள்வரி வெறுமை apir_decoder_peek(apir_decoder * dec, size_t size, வெறுமை * val, size_t val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

நிலையான உள்வரி மாறிலி வெறுமை * apir_decoder_use_inplace(apir_decoder * dec, size_t size) {
    எனில் (unlikely(size > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        திரும்பு வெற்று;
    }
    மாறிலி வெறுமை * addr = dec->cur;
    dec->cur += size;

    திரும்பு addr;
}

/*
 * read/write
 */

நிலையான உள்வரி வெறுமை apir_decoder_read(apir_decoder * dec, size_t size, வெறுமை * val, size_t val_size) {
    எனில் (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

நிலையான உள்வரி எழுத்து * apir_encoder_write(apir_encoder * enc, size_t size, மாறிலி வெறுமை * val, size_t val_size) {
    ப்ரதிஜ்ஞா(val_size <= size);
    ப்ரதிஜ்ஞா(size <= ((size_t) (enc->end - enc->cur)));

    எழுத்து * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_size);
    enc->cur += size;

    திரும்பு write_addr;
}

/*
 * encode/decode
 */

நிலையான உள்வரி வெறுமை apir_decode(apir_decoder * dec, size_t size, வெறுமை * data, size_t data_size) {
    ப்ரதிஜ்ஞா(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

நிலையான உள்வரி வெறுமை apir_encode(apir_encoder * enc, size_t size, மாறிலி வெறுமை * data, size_t data_size) {
    ப்ரதிஜ்ஞா(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

நிலையான உள்வரி வெறுமை apir_encode_uint8_t(apir_encoder * enc, மாறிலி uint8_t * val) {
    apir_encode(enc, அளவு(முழு), val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, அளவு(முழு), val, அளவு(*val));
}

/* uint64_t */

நிலையான உள்வரி வெறுமை apir_encode_uint64_t(apir_encoder * enc, மாறிலி uint64_t * val) {
    apir_encode(enc, 8, val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_encode_uint64_t_array(apir_encoder * enc, மாறிலி uint64_t * val, uint32_t மொத்தம்) {
    மாறிலி size_t size = அளவு(*val) * மொத்தம்;
    ப்ரதிஜ்ஞா(size >= மொத்தம்);
    apir_encode(enc, size, val, size);
}

நிலையான உள்வரி வெறுமை apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t மொத்தம்) {
    மாறிலி size_t size = அளவு(*val) * மொத்தம்;
    ப்ரதிஜ்ஞா(size >= மொத்தம்);
    apir_decode(dec, size, val, size);
}

நிலையான உள்வரி மாறிலி uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t மொத்தம்) {
    திரும்பு (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, மொத்தம் * அளவு(uint64_t));
}

/* int32_t */

நிலையான உள்வரி வெறுமை apir_encode_int32_t(apir_encoder * enc, மாறிலி int32_t * val) {
    apir_encode(enc, 4, val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_encode_int32_t_array(apir_encoder * enc, மாறிலி int32_t * val, uint32_t மொத்தம்) {
    மாறிலி size_t size = அளவு(*val) * மொத்தம்;
    ப்ரதிஜ்ஞா(size >= மொத்தம்);
    apir_encode(enc, size, val, size);
}

நிலையான உள்வரி வெறுமை apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t மொத்தம்) {
    மாறிலி size_t size = அளவு(*val) * மொத்தம்;
    ப்ரதிஜ்ஞா(size >= மொத்தம்);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

நிலையான உள்வரி வெறுமை apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

நிலையான உள்வரி uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    எனில் (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    திரும்பு size;
}

நிலையான உள்வரி uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    திரும்பு size;
}

/* non-array pointer */

நிலையான உள்வரி bool apir_encode_simple_pointer(apir_encoder * enc, மாறிலி வெறுமை * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    திரும்பு val;
}

நிலையான உள்வரி bool apir_decode_simple_pointer(apir_decoder * dec) {
    திரும்பு apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

நிலையான உள்வரி வெறுமை apir_encode_uint32_t(apir_encoder * enc, மாறிலி uint32_t * val) {
    apir_encode(enc, 4, val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_encode_uint32_t_array(apir_encoder * enc, மாறிலி uint32_t * val, uint32_t மொத்தம்) {
    மாறிலி size_t size = அளவு(*val) * மொத்தம்;
    ப்ரதிஜ்ஞா(size >= மொத்தம்);
    apir_encode(enc, size, val, size);
}

நிலையான உள்வரி வெறுமை apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t மொத்தம்) {
    மாறிலி size_t size = அளவு(*val) * மொத்தம்;
    ப்ரதிஜ்ஞா(size >= மொத்தம்);
    apir_decode(dec, size, val, size);
}

/* size_t */

நிலையான உள்வரி வெறுமை apir_encode_size_t(apir_encoder * enc, மாறிலி size_t * val) {
    மாறிலி uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

நிலையான உள்வரி வெறுமை apir_decode_size_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

நிலையான உள்வரி வெறுமை apir_encode_size_t_array(apir_encoder * enc, மாறிலி size_t * val, uint32_t மொத்தம்) {
    எனில் (அளவு(size_t) == அளவு(uint64_t)) {
        apir_encode_uint64_t_array(enc, (மாறிலி uint64_t *) val, மொத்தம்);
    } இல்லையெனில் {
        க்கு (uint32_t i = 0; i < மொத்தம்; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

நிலையான உள்வரி வெறுமை apir_decode_size_t_array(apir_decoder * dec, size_t * val, uint32_t மொத்தம்) {
    எனில் (அளவு(size_t) == அளவு(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, மொத்தம்);
    } இல்லையெனில் {
        க்கு (uint32_t i = 0; i < மொத்தம்; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

நிலையான உள்வரி வெறுமை apir_encode_blob_array(apir_encoder * enc, மாறிலி வெறுமை * val, size_t size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

நிலையான உள்வரி வெறுமை apir_decode_blob_array(apir_decoder * dec, வெறுமை * val, size_t size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

நிலையான உள்வரி வெறுமை apir_encode_char_array(apir_encoder * enc, மாறிலி எழுத்து * val, size_t size) {
    ப்ரதிஜ்ஞா(size && strlen(val) < size);
    apir_encode_blob_array(enc, val, size);
}

நிலையான உள்வரி வெறுமை apir_decode_char_array(apir_decoder * dec, எழுத்து * val, size_t size) {
    apir_decode_blob_array(dec, val, size);
    எனில் (size) {
        val[size - 1] = '\0';
    } இல்லையெனில் {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

நிலையான உள்வரி வெறுமை * apir_decoder_alloc_array(size_t size, size_t மொத்தம்) {
    size_t alloc_size;
    எனில் (unlikely(__builtin_mul_overflow(size, மொத்தம், &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, மொத்தம்);
        திரும்பு வெற்று;
    }

    திரும்பு malloc(alloc_size);
}

/* bool */

நிலையான உள்வரி வெறுமை apir_encode_bool_t(apir_encoder * enc, மாறிலி bool * val) {
    apir_encode(enc, அளவு(முழு), val, அளவு(bool));
}

நிலையான உள்வரி வெறுமை apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, அளவு(முழு), val, அளவு(bool));
}

/* apir_buffer_type_host_handle_t */

நிலையான உள்வரி வெறுமை apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              மாறிலி apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, அளவு(apir_buffer_type_host_handle_t), val, அளவு(apir_buffer_type_host_handle_t));
}

நிலையான உள்வரி வெறுமை apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, அளவு(apir_buffer_type_host_handle_t), val, அளவு(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

நிலையான உள்வரி வெறுமை apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, மாறிலி apir_buffer_host_handle_t * val) {
    apir_encode(enc, அளவு(apir_buffer_host_handle_t), val, அளவு(apir_buffer_host_handle_t));
}

நிலையான உள்வரி வெறுமை apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, அளவு(apir_buffer_host_handle_t), val, அளவு(apir_buffer_host_handle_t));
}

/* uintptr_t */

நிலையான உள்வரி வெறுமை apir_encode_uintptr_t(apir_encoder * enc, மாறிலி uintptr_t * val) {
    apir_encode(enc, அளவு(*val), val, அளவு(*val));
}

நிலையான உள்வரி வெறுமை apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, அளவு(*val), val, அளவு(*val));
}
