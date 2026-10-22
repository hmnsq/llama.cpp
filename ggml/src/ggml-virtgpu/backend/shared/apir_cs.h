#pragma once

#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

संरचना apir_encoder {
    अक्षर *       cur;
    स्थिरांक अक्षर * start;
    स्थिरांक अक्षर * end;
    bool         fatal;
};

संरचना apir_decoder {
    स्थिरांक अक्षर * cur;
    स्थिरांक अक्षर * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

स्थिर apir_decoder apir_new_decoder(स्थिरांक अक्षर * ptr, माप_प्रकार size) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    परत dec;
}

स्थिर apir_encoder apir_new_encoder(अक्षर * ptr, माप_प्रकार size) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + size,
        .fatal = false,
    };

    परत enc;
}

/*
 * fatal flag handling
 */

स्थिर ओळीत शून्य apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

स्थिर ओळीत शून्य apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

स्थिर ओळीत bool apir_encoder_get_fatal(स्थिरांक apir_encoder * enc) {
    परत enc->fatal;
}

स्थिर ओळीत शून्य apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

स्थिर ओळीत शून्य apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

स्थिर ओळीत bool apir_decoder_get_fatal(स्थिरांक apir_decoder * dec) {
    परत dec->fatal;
}

/*
 * encode peek
 */

स्थिर ओळीत bool apir_decoder_peek_internal(apir_decoder * dec, माप_प्रकार size, शून्य * val, माप_प्रकार val_size) {
    प्रतिज्ञा(val_size <= size);

    जर (unlikely(size > (माप_प्रकार) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        स_रखो(val, 0, val_size);
        परत false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    स_नकल(val, dec->cur, val_size);
    परत true;
}

स्थिर ओळीत शून्य apir_decoder_peek(apir_decoder * dec, माप_प्रकार size, शून्य * val, माप_प्रकार val_size) {
    apir_decoder_peek_internal(dec, size, val, val_size);
}

स्थिर ओळीत स्थिरांक शून्य * apir_decoder_use_inplace(apir_decoder * dec, माप_प्रकार size) {
    जर (unlikely(size > (माप_प्रकार) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        परत रिक्त;
    }
    स्थिरांक शून्य * addr = dec->cur;
    dec->cur += size;

    परत addr;
}

/*
 * read/write
 */

स्थिर ओळीत शून्य apir_decoder_read(apir_decoder * dec, माप_प्रकार size, शून्य * val, माप_प्रकार val_size) {
    जर (apir_decoder_peek_internal(dec, size, val, val_size)) {
        dec->cur += size;
    }
}

स्थिर ओळीत अक्षर * apir_encoder_write(apir_encoder * enc, माप_प्रकार size, स्थिरांक शून्य * val, माप_प्रकार val_size) {
    प्रतिज्ञा(val_size <= size);
    प्रतिज्ञा(size <= ((माप_प्रकार) (enc->end - enc->cur)));

    अक्षर * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    स_नकल(write_addr, val, val_size);
    enc->cur += size;

    परत write_addr;
}

/*
 * encode/decode
 */

स्थिर ओळीत शून्य apir_decode(apir_decoder * dec, माप_प्रकार size, शून्य * data, माप_प्रकार data_size) {
    प्रतिज्ञा(size % 4 == 0);
    apir_decoder_read(dec, size, data, data_size);
}

स्थिर ओळीत शून्य apir_encode(apir_encoder * enc, माप_प्रकार size, स्थिरांक शून्य * data, माप_प्रकार data_size) {
    प्रतिज्ञा(size % 4 == 0);
    apir_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint8_t */

स्थिर ओळीत शून्य apir_encode_uint8_t(apir_encoder * enc, स्थिरांक uint8_t * val) {
    apir_encode(enc, आकार(पूर्णांक), val, आकार(*val));
}

स्थिर ओळीत शून्य apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, आकार(पूर्णांक), val, आकार(*val));
}

/* uint64_t */

स्थिर ओळीत शून्य apir_encode_uint64_t(apir_encoder * enc, स्थिरांक uint64_t * val) {
    apir_encode(enc, 8, val, आकार(*val));
}

स्थिर ओळीत शून्य apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, आकार(*val));
}

स्थिर ओळीत शून्य apir_encode_uint64_t_array(apir_encoder * enc, स्थिरांक uint64_t * val, uint32_t मोजणी) {
    स्थिरांक माप_प्रकार size = आकार(*val) * मोजणी;
    प्रतिज्ञा(size >= मोजणी);
    apir_encode(enc, size, val, size);
}

स्थिर ओळीत शून्य apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t मोजणी) {
    स्थिरांक माप_प्रकार size = आकार(*val) * मोजणी;
    प्रतिज्ञा(size >= मोजणी);
    apir_decode(dec, size, val, size);
}

स्थिर ओळीत स्थिरांक uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t मोजणी) {
    परत (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, मोजणी * आकार(uint64_t));
}

/* int32_t */

स्थिर ओळीत शून्य apir_encode_int32_t(apir_encoder * enc, स्थिरांक int32_t * val) {
    apir_encode(enc, 4, val, आकार(*val));
}

स्थिर ओळीत शून्य apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, आकार(*val));
}

स्थिर ओळीत शून्य apir_encode_int32_t_array(apir_encoder * enc, स्थिरांक int32_t * val, uint32_t मोजणी) {
    स्थिरांक माप_प्रकार size = आकार(*val) * मोजणी;
    प्रतिज्ञा(size >= मोजणी);
    apir_encode(enc, size, val, size);
}

स्थिर ओळीत शून्य apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t मोजणी) {
    स्थिरांक माप_प्रकार size = आकार(*val) * मोजणी;
    प्रतिज्ञा(size >= मोजणी);
    apir_decode(dec, size, val, size);
}

/* array size (uint64_t) */

स्थिर ओळीत शून्य apir_encode_array_size(apir_encoder * enc, uint64_t size) {
    apir_encode_uint64_t(enc, &size);
}

स्थिर ओळीत uint64_t apir_decode_array_size(apir_decoder * dec, uint64_t expected_size) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    जर (size != expected_size) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        size = 0;
    }
    परत size;
}

स्थिर ओळीत uint64_t apir_decode_array_size_unchecked(apir_decoder * dec) {
    uint64_t size;
    apir_decode_uint64_t(dec, &size);
    परत size;
}

/* non-array pointer */

स्थिर ओळीत bool apir_encode_simple_pointer(apir_encoder * enc, स्थिरांक शून्य * val) {
    apir_encode_array_size(enc, val ? 1 : 0);
    परत val;
}

स्थिर ओळीत bool apir_decode_simple_pointer(apir_decoder * dec) {
    परत apir_decode_array_size_unchecked(dec);
}

/* uint32_t */

स्थिर ओळीत शून्य apir_encode_uint32_t(apir_encoder * enc, स्थिरांक uint32_t * val) {
    apir_encode(enc, 4, val, आकार(*val));
}

स्थिर ओळीत शून्य apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, आकार(*val));
}

स्थिर ओळीत शून्य apir_encode_uint32_t_array(apir_encoder * enc, स्थिरांक uint32_t * val, uint32_t मोजणी) {
    स्थिरांक माप_प्रकार size = आकार(*val) * मोजणी;
    प्रतिज्ञा(size >= मोजणी);
    apir_encode(enc, size, val, size);
}

स्थिर ओळीत शून्य apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t मोजणी) {
    स्थिरांक माप_प्रकार size = आकार(*val) * मोजणी;
    प्रतिज्ञा(size >= मोजणी);
    apir_decode(dec, size, val, size);
}

/* size_t */

स्थिर ओळीत शून्य apir_encode_size_t(apir_encoder * enc, स्थिरांक माप_प्रकार * val) {
    स्थिरांक uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

स्थिर ओळीत शून्य apir_decode_size_t(apir_decoder * dec, माप_प्रकार * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

स्थिर ओळीत शून्य apir_encode_size_t_array(apir_encoder * enc, स्थिरांक माप_प्रकार * val, uint32_t मोजणी) {
    जर (आकार(माप_प्रकार) == आकार(uint64_t)) {
        apir_encode_uint64_t_array(enc, (स्थिरांक uint64_t *) val, मोजणी);
    } अन्यथा {
        साठी (uint32_t i = 0; i < मोजणी; i++) {
            apir_encode_size_t(enc, &val[i]);
        }
    }
}

स्थिर ओळीत शून्य apir_decode_size_t_array(apir_decoder * dec, माप_प्रकार * val, uint32_t मोजणी) {
    जर (आकार(माप_प्रकार) == आकार(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, मोजणी);
    } अन्यथा {
        साठी (uint32_t i = 0; i < मोजणी; i++) {
            apir_decode_size_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

स्थिर ओळीत शून्य apir_encode_blob_array(apir_encoder * enc, स्थिरांक शून्य * val, माप_प्रकार size) {
    apir_encode(enc, (size + 3) & ~3, val, size);
}

स्थिर ओळीत शून्य apir_decode_blob_array(apir_decoder * dec, शून्य * val, माप_प्रकार size) {
    apir_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

स्थिर ओळीत शून्य apir_encode_char_array(apir_encoder * enc, स्थिरांक अक्षर * val, माप_प्रकार size) {
    प्रतिज्ञा(size && म_माप(val) < size);
    apir_encode_blob_array(enc, val, size);
}

स्थिर ओळीत शून्य apir_decode_char_array(apir_decoder * dec, अक्षर * val, माप_प्रकार size) {
    apir_decode_blob_array(dec, val, size);
    जर (size) {
        val[size - 1] = '\0';
    } अन्यथा {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

स्थिर ओळीत शून्य * apir_decoder_alloc_array(माप_प्रकार size, माप_प्रकार मोजणी) {
    माप_प्रकार alloc_size;
    जर (unlikely(__builtin_mul_overflow(size, मोजणी, &alloc_size))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, size, मोजणी);
        परत रिक्त;
    }

    परत दो_स्मृति(alloc_size);
}

/* bool */

स्थिर ओळीत शून्य apir_encode_bool_t(apir_encoder * enc, स्थिरांक bool * val) {
    apir_encode(enc, आकार(पूर्णांक), val, आकार(bool));
}

स्थिर ओळीत शून्य apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, आकार(पूर्णांक), val, आकार(bool));
}

/* apir_buffer_type_host_handle_t */

स्थिर ओळीत शून्य apir_encode_apir_buffer_type_host_handle_t(apir_encoder *                         enc,
                                                              स्थिरांक apir_buffer_type_host_handle_t * val) {
    apir_encode(enc, आकार(apir_buffer_type_host_handle_t), val, आकार(apir_buffer_type_host_handle_t));
}

स्थिर ओळीत शून्य apir_decode_apir_buffer_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_buffer_type_host_handle_t * val) {
    apir_decode(dec, आकार(apir_buffer_type_host_handle_t), val, आकार(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

स्थिर ओळीत शून्य apir_encode_apir_buffer_host_handle_t(apir_encoder * enc, स्थिरांक apir_buffer_host_handle_t * val) {
    apir_encode(enc, आकार(apir_buffer_host_handle_t), val, आकार(apir_buffer_host_handle_t));
}

स्थिर ओळीत शून्य apir_decode_apir_buffer_host_handle_t(apir_decoder * dec, apir_buffer_host_handle_t * val) {
    apir_decode(dec, आकार(apir_buffer_host_handle_t), val, आकार(apir_buffer_host_handle_t));
}

/* uintptr_t */

स्थिर ओळीत शून्य apir_encode_uintptr_t(apir_encoder * enc, स्थिरांक uintptr_t * val) {
    apir_encode(enc, आकार(*val), val, आकार(*val));
}

स्थिर ओळीत शून्य apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, आकार(*val), val, आकार(*val));
}
