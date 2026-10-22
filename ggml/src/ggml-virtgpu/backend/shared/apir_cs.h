#निर्देशः once

#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstring>

#परिभाषय likely(x)   __builtin_expect(!!(x), 1)
#परिभाषय unlikely(x) __builtin_expect(!!(x), 0)

संरचना apir_encoder {
    वर्ण *       cur;
    ध्रुव वर्ण * start;
    ध्रुव वर्ण * अन्तः_2;
    bool         fatal;
};

संरचना apir_decoder {
    ध्रुव वर्ण * cur;
    ध्रुव वर्ण * अन्तः_2;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

स्थिर apir_decoder apir_new_decoder(ध्रुव वर्ण * ptr, size_t आकार) {
    apir_decoder dec = {
        .cur   = ptr,
        .अन्तः_2   = ptr + आकार,
        .fatal = false,
    };

    प्रत्यागम dec;
}

स्थिर apir_encoder apir_new_encoder(वर्ण * ptr, size_t आकार) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .अन्तः_2   = ptr + आकार,
        .fatal = false,
    };

    प्रत्यागम enc;
}

/*
 * fatal flag handling
 */

स्थिर अन्तःस्थ शून्य apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

स्थिर अन्तःस्थ शून्य apir_encoder_समुच्चयः_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

स्थिर अन्तःस्थ bool apir_encoder_get_fatal(ध्रुव apir_encoder * enc) {
    प्रत्यागम enc->fatal;
}

स्थिर अन्तःस्थ शून्य apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

स्थिर अन्तःस्थ शून्य apir_decoder_समुच्चयः_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

स्थिर अन्तःस्थ bool apir_decoder_get_fatal(ध्रुव apir_decoder * dec) {
    प्रत्यागम dec->fatal;
}

/*
 * encode peek
 */

स्थिर अन्तःस्थ bool apir_decoder_peek_आन्तरिकम्(apir_decoder * dec, size_t आकार, शून्य * val, size_t val_आकार) {
    प्रतिज्ञा(val_आकार <= आकार);

    यदि (unlikely(आकार > (size_t) (dec->अन्तः_2 - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_समुच्चयः_fatal(dec);
        स्मृतिं_पूरय(val, 0, val_आकार);
        प्रत्यागम false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    स्मृतिं_प्रतिलिख(val, dec->cur, val_आकार);
    प्रत्यागम true;
}

स्थिर अन्तःस्थ शून्य apir_decoder_peek(apir_decoder * dec, size_t आकार, शून्य * val, size_t val_आकार) {
    apir_decoder_peek_आन्तरिकम्(dec, आकार, val, val_आकार);
}

स्थिर अन्तःस्थ ध्रुव शून्य * apir_decoder_use_inplace(apir_decoder * dec, size_t आकार) {
    यदि (unlikely(आकार > (size_t) (dec->अन्तः_2 - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_समुच्चयः_fatal(dec);
        प्रत्यागम रिक्त;
    }
    ध्रुव शून्य * addr = dec->cur;
    dec->cur += आकार;

    प्रत्यागम addr;
}

/*
 * read/write
 */

स्थिर अन्तःस्थ शून्य apir_decoder_read(apir_decoder * dec, size_t आकार, शून्य * val, size_t val_आकार) {
    यदि (apir_decoder_peek_आन्तरिकम्(dec, आकार, val, val_आकार)) {
        dec->cur += आकार;
    }
}

स्थिर अन्तःस्थ वर्ण * apir_encoder_write(apir_encoder * enc, size_t आकार, ध्रुव शून्य * val, size_t val_आकार) {
    प्रतिज्ञा(val_आकार <= आकार);
    प्रतिज्ञा(आकार <= ((size_t) (enc->अन्तः_2 - enc->cur)));

    वर्ण * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    स्मृतिं_प्रतिलिख(write_addr, val, val_आकार);
    enc->cur += आकार;

    प्रत्यागम write_addr;
}

/*
 * encode/decode
 */

स्थिर अन्तःस्थ शून्य apir_decode(apir_decoder * dec, size_t आकार, शून्य * दत्त, size_t दत्त_आकार) {
    प्रतिज्ञा(आकार % 4 == 0);
    apir_decoder_read(dec, आकार, दत्त, दत्त_आकार);
}

स्थिर अन्तःस्थ शून्य apir_encode(apir_encoder * enc, size_t आकार, ध्रुव शून्य * दत्त, size_t दत्त_आकार) {
    प्रतिज्ञा(आकार % 4 == 0);
    apir_encoder_write(enc, आकार, दत्त, दत्त_आकार);
}

/*
 * typed encode/decode
 */

/* uint8_t */

स्थिर अन्तःस्थ शून्य apir_encode_uint8_t(apir_encoder * enc, ध्रुव uint8_t * val) {
    apir_encode(enc, परिमाण(पूर्णाङ्क), val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, परिमाण(पूर्णाङ्क), val, परिमाण(*val));
}

/* uint64_t */

स्थिर अन्तःस्थ शून्य apir_encode_uint64_t(apir_encoder * enc, ध्रुव uint64_t * val) {
    apir_encode(enc, 8, val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_encode_uint64_t_array(apir_encoder * enc, ध्रुव uint64_t * val, uint32_t गणनम्) {
    ध्रुव size_t आकार = परिमाण(*val) * गणनम्;
    प्रतिज्ञा(आकार >= गणनम्);
    apir_encode(enc, आकार, val, आकार);
}

स्थिर अन्तःस्थ शून्य apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t गणनम्) {
    ध्रुव size_t आकार = परिमाण(*val) * गणनम्;
    प्रतिज्ञा(आकार >= गणनम्);
    apir_decode(dec, आकार, val, आकार);
}

स्थिर अन्तःस्थ ध्रुव uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t गणनम्) {
    प्रत्यागम (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, गणनम् * परिमाण(uint64_t));
}

/* int32_t */

स्थिर अन्तःस्थ शून्य apir_encode_int32_t(apir_encoder * enc, ध्रुव int32_t * val) {
    apir_encode(enc, 4, val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_encode_int32_t_array(apir_encoder * enc, ध्रुव int32_t * val, uint32_t गणनम्) {
    ध्रुव size_t आकार = परिमाण(*val) * गणनम्;
    प्रतिज्ञा(आकार >= गणनम्);
    apir_encode(enc, आकार, val, आकार);
}

स्थिर अन्तःस्थ शून्य apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t गणनम्) {
    ध्रुव size_t आकार = परिमाण(*val) * गणनम्;
    प्रतिज्ञा(आकार >= गणनम्);
    apir_decode(dec, आकार, val, आकार);
}

/* array size (uint64_t) */

स्थिर अन्तःस्थ शून्य apir_encode_array_आकार(apir_encoder * enc, uint64_t आकार) {
    apir_encode_uint64_t(enc, &आकार);
}

स्थिर अन्तःस्थ uint64_t apir_decode_array_आकार(apir_decoder * dec, uint64_t expected_आकार) {
    uint64_t आकार;
    apir_decode_uint64_t(dec, &आकार);
    यदि (आकार != expected_आकार) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_समुच्चयः_fatal(dec);
        आकार = 0;
    }
    प्रत्यागम आकार;
}

स्थिर अन्तःस्थ uint64_t apir_decode_array_आकार_unchecked(apir_decoder * dec) {
    uint64_t आकार;
    apir_decode_uint64_t(dec, &आकार);
    प्रत्यागम आकार;
}

/* non-array pointer */

स्थिर अन्तःस्थ bool apir_encode_simple_pointer(apir_encoder * enc, ध्रुव शून्य * val) {
    apir_encode_array_आकार(enc, val ? 1 : 0);
    प्रत्यागम val;
}

स्थिर अन्तःस्थ bool apir_decode_simple_pointer(apir_decoder * dec) {
    प्रत्यागम apir_decode_array_आकार_unchecked(dec);
}

/* uint32_t */

स्थिर अन्तःस्थ शून्य apir_encode_uint32_t(apir_encoder * enc, ध्रुव uint32_t * val) {
    apir_encode(enc, 4, val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_encode_uint32_t_array(apir_encoder * enc, ध्रुव uint32_t * val, uint32_t गणनम्) {
    ध्रुव size_t आकार = परिमाण(*val) * गणनम्;
    प्रतिज्ञा(आकार >= गणनम्);
    apir_encode(enc, आकार, val, आकार);
}

स्थिर अन्तःस्थ शून्य apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t गणनम्) {
    ध्रुव size_t आकार = परिमाण(*val) * गणनम्;
    प्रतिज्ञा(आकार >= गणनम्);
    apir_decode(dec, आकार, val, आकार);
}

/* size_t */

स्थिर अन्तःस्थ शून्य apir_encode_आकार_t(apir_encoder * enc, ध्रुव size_t * val) {
    ध्रुव uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

स्थिर अन्तःस्थ शून्य apir_decode_आकार_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

स्थिर अन्तःस्थ शून्य apir_encode_आकार_t_array(apir_encoder * enc, ध्रुव size_t * val, uint32_t गणनम्) {
    यदि (परिमाण(size_t) == परिमाण(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ध्रुव uint64_t *) val, गणनम्);
    } अन्यथा {
        प्रति (uint32_t i = 0; i < गणनम्; i++) {
            apir_encode_आकार_t(enc, &val[i]);
        }
    }
}

स्थिर अन्तःस्थ शून्य apir_decode_आकार_t_array(apir_decoder * dec, size_t * val, uint32_t गणनम्) {
    यदि (परिमाण(size_t) == परिमाण(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, गणनम्);
    } अन्यथा {
        प्रति (uint32_t i = 0; i < गणनम्; i++) {
            apir_decode_आकार_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

स्थिर अन्तःस्थ शून्य apir_encode_blob_array(apir_encoder * enc, ध्रुव शून्य * val, size_t आकार) {
    apir_encode(enc, (आकार + 3) & ~3, val, आकार);
}

स्थिर अन्तःस्थ शून्य apir_decode_blob_array(apir_decoder * dec, शून्य * val, size_t आकार) {
    apir_decode(dec, (आकार + 3) & ~3, val, आकार);
}

/* string */

स्थिर अन्तःस्थ शून्य apir_encode_char_array(apir_encoder * enc, ध्रुव वर्ण * val, size_t आकार) {
    प्रतिज्ञा(आकार && वाक्यदैर्घ्यम्(val) < आकार);
    apir_encode_blob_array(enc, val, आकार);
}

स्थिर अन्तःस्थ शून्य apir_decode_char_array(apir_decoder * dec, वर्ण * val, size_t आकार) {
    apir_decode_blob_array(dec, val, आकार);
    यदि (आकार) {
        val[आकार - 1] = '\0';
    } अन्यथा {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_समुच्चयः_fatal(dec);
    }
}

/* (temp) buffer allocation */

स्थिर अन्तःस्थ शून्य * apir_decoder_alloc_array(size_t आकार, size_t गणनम्) {
    size_t alloc_आकार;
    यदि (unlikely(__builtin_mul_overflow(आकार, गणनम्, &alloc_आकार))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, आकार, गणनम्);
        प्रत्यागम रिक्त;
    }

    प्रत्यागम स्मृतिं_देहि(alloc_आकार);
}

/* bool */

स्थिर अन्तःस्थ शून्य apir_encode_bool_t(apir_encoder * enc, ध्रुव bool * val) {
    apir_encode(enc, परिमाण(पूर्णाङ्क), val, परिमाण(bool));
}

स्थिर अन्तःस्थ शून्य apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, परिमाण(पूर्णाङ्क), val, परिमाण(bool));
}

/* apir_buffer_type_host_handle_t */

स्थिर अन्तःस्थ शून्य apir_encode_apir_बफर_type_host_handle_t(apir_encoder *                         enc,
                                                              ध्रुव apir_बफर_type_host_handle_t * val) {
    apir_encode(enc, परिमाण(apir_बफर_type_host_handle_t), val, परिमाण(apir_बफर_type_host_handle_t));
}

स्थिर अन्तःस्थ शून्य apir_decode_apir_बफर_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_बफर_type_host_handle_t * val) {
    apir_decode(dec, परिमाण(apir_बफर_type_host_handle_t), val, परिमाण(apir_बफर_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

स्थिर अन्तःस्थ शून्य apir_encode_apir_बफर_host_handle_t(apir_encoder * enc, ध्रुव apir_बफर_host_handle_t * val) {
    apir_encode(enc, परिमाण(apir_बफर_host_handle_t), val, परिमाण(apir_बफर_host_handle_t));
}

स्थिर अन्तःस्थ शून्य apir_decode_apir_बफर_host_handle_t(apir_decoder * dec, apir_बफर_host_handle_t * val) {
    apir_decode(dec, परिमाण(apir_बफर_host_handle_t), val, परिमाण(apir_बफर_host_handle_t));
}

/* uintptr_t */

स्थिर अन्तःस्थ शून्य apir_encode_uintptr_t(apir_encoder * enc, ध्रुव uintptr_t * val) {
    apir_encode(enc, परिमाण(*val), val, परिमाण(*val));
}

स्थिर अन्तःस्थ शून्य apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, परिमाण(*val), val, परिमाण(*val));
}
