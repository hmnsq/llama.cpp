#आशय once

#समावेश "ggml-impl.h"

#समावेश <cassert>
#समावेश <cstring>

#घोषणा likely(x)   __builtin_expect(!!(x), 1)
#घोषणा unlikely(x) __builtin_expect(!!(x), 0)

काष्ठा apir_encoder {
    अक्षर *       cur;
    स्थिर अक्षर * start;
    स्थिर अक्षर * end;
    bool         fatal;
};

काष्ठा apir_decoder {
    स्थिर अक्षर * cur;
    स्थिर अक्षर * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

अटल apir_decoder apir_new_decoder(स्थिर अक्षर * ptr, माप_प्रकार आकार) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + आकार,
        .fatal = false,
    };

    वापस dec;
}

अटल apir_encoder apir_new_encoder(अक्षर * ptr, माप_प्रकार आकार) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + आकार,
        .fatal = false,
    };

    वापस enc;
}

/*
 * fatal flag handling
 */

अटल अंतरभूत व्योम apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

अटल अंतरभूत व्योम apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

अटल अंतरभूत bool apir_encoder_get_fatal(स्थिर apir_encoder * enc) {
    वापस enc->fatal;
}

अटल अंतरभूत व्योम apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

अटल अंतरभूत व्योम apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

अटल अंतरभूत bool apir_decoder_get_fatal(स्थिर apir_decoder * dec) {
    वापस dec->fatal;
}

/*
 * encode peek
 */

अटल अंतरभूत bool apir_decoder_peek_internal(apir_decoder * dec, माप_प्रकार आकार, व्योम * val, माप_प्रकार val_आकार) {
    निश्चित(val_आकार <= आकार);

    अगर (unlikely(आकार > (माप_प्रकार) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        स_रखो(val, 0, val_आकार);
        वापस false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    स_नकल(val, dec->cur, val_आकार);
    वापस true;
}

अटल अंतरभूत व्योम apir_decoder_peek(apir_decoder * dec, माप_प्रकार आकार, व्योम * val, माप_प्रकार val_आकार) {
    apir_decoder_peek_internal(dec, आकार, val, val_आकार);
}

अटल अंतरभूत स्थिर व्योम * apir_decoder_use_inplace(apir_decoder * dec, माप_प्रकार आकार) {
    अगर (unlikely(आकार > (माप_प्रकार) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        वापस शून्य;
    }
    स्थिर व्योम * addr = dec->cur;
    dec->cur += आकार;

    वापस addr;
}

/*
 * read/write
 */

अटल अंतरभूत व्योम apir_decoder_read(apir_decoder * dec, माप_प्रकार आकार, व्योम * val, माप_प्रकार val_आकार) {
    अगर (apir_decoder_peek_internal(dec, आकार, val, val_आकार)) {
        dec->cur += आकार;
    }
}

अटल अंतरभूत अक्षर * apir_encoder_write(apir_encoder * enc, माप_प्रकार आकार, स्थिर व्योम * val, माप_प्रकार val_आकार) {
    निश्चित(val_आकार <= आकार);
    निश्चित(आकार <= ((माप_प्रकार) (enc->end - enc->cur)));

    अक्षर * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    स_नकल(write_addr, val, val_आकार);
    enc->cur += आकार;

    वापस write_addr;
}

/*
 * encode/decode
 */

अटल अंतरभूत व्योम apir_decode(apir_decoder * dec, माप_प्रकार आकार, व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    निश्चित(आकार % 4 == 0);
    apir_decoder_read(dec, आकार, तथ्य, तथ्य_आकार);
}

अटल अंतरभूत व्योम apir_encode(apir_encoder * enc, माप_प्रकार आकार, स्थिर व्योम * तथ्य, माप_प्रकार तथ्य_आकार) {
    निश्चित(आकार % 4 == 0);
    apir_encoder_write(enc, आकार, तथ्य, तथ्य_आकार);
}

/*
 * typed encode/decode
 */

/* uint8_t */

अटल अंतरभूत व्योम apir_encode_uint8_t(apir_encoder * enc, स्थिर uint8_t * val) {
    apir_encode(enc, माप(पूर्णांक), val, माप(*val));
}

अटल अंतरभूत व्योम apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, माप(पूर्णांक), val, माप(*val));
}

/* uint64_t */

अटल अंतरभूत व्योम apir_encode_uint64_t(apir_encoder * enc, स्थिर uint64_t * val) {
    apir_encode(enc, 8, val, माप(*val));
}

अटल अंतरभूत व्योम apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, माप(*val));
}

अटल अंतरभूत व्योम apir_encode_uint64_t_array(apir_encoder * enc, स्थिर uint64_t * val, uint32_t गिनती) {
    स्थिर माप_प्रकार आकार = माप(*val) * गिनती;
    निश्चित(आकार >= गिनती);
    apir_encode(enc, आकार, val, आकार);
}

अटल अंतरभूत व्योम apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t गिनती) {
    स्थिर माप_प्रकार आकार = माप(*val) * गिनती;
    निश्चित(आकार >= गिनती);
    apir_decode(dec, आकार, val, आकार);
}

अटल अंतरभूत स्थिर uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t गिनती) {
    वापस (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, गिनती * माप(uint64_t));
}

/* int32_t */

अटल अंतरभूत व्योम apir_encode_int32_t(apir_encoder * enc, स्थिर int32_t * val) {
    apir_encode(enc, 4, val, माप(*val));
}

अटल अंतरभूत व्योम apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, माप(*val));
}

अटल अंतरभूत व्योम apir_encode_int32_t_array(apir_encoder * enc, स्थिर int32_t * val, uint32_t गिनती) {
    स्थिर माप_प्रकार आकार = माप(*val) * गिनती;
    निश्चित(आकार >= गिनती);
    apir_encode(enc, आकार, val, आकार);
}

अटल अंतरभूत व्योम apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t गिनती) {
    स्थिर माप_प्रकार आकार = माप(*val) * गिनती;
    निश्चित(आकार >= गिनती);
    apir_decode(dec, आकार, val, आकार);
}

/* array size (uint64_t) */

अटल अंतरभूत व्योम apir_encode_array_आकार(apir_encoder * enc, uint64_t आकार) {
    apir_encode_uint64_t(enc, &आकार);
}

अटल अंतरभूत uint64_t apir_decode_array_आकार(apir_decoder * dec, uint64_t expected_आकार) {
    uint64_t आकार;
    apir_decode_uint64_t(dec, &आकार);
    अगर (आकार != expected_आकार) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        आकार = 0;
    }
    वापस आकार;
}

अटल अंतरभूत uint64_t apir_decode_array_आकार_unchecked(apir_decoder * dec) {
    uint64_t आकार;
    apir_decode_uint64_t(dec, &आकार);
    वापस आकार;
}

/* non-array pointer */

अटल अंतरभूत bool apir_encode_simple_pointer(apir_encoder * enc, स्थिर व्योम * val) {
    apir_encode_array_आकार(enc, val ? 1 : 0);
    वापस val;
}

अटल अंतरभूत bool apir_decode_simple_pointer(apir_decoder * dec) {
    वापस apir_decode_array_आकार_unchecked(dec);
}

/* uint32_t */

अटल अंतरभूत व्योम apir_encode_uint32_t(apir_encoder * enc, स्थिर uint32_t * val) {
    apir_encode(enc, 4, val, माप(*val));
}

अटल अंतरभूत व्योम apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, माप(*val));
}

अटल अंतरभूत व्योम apir_encode_uint32_t_array(apir_encoder * enc, स्थिर uint32_t * val, uint32_t गिनती) {
    स्थिर माप_प्रकार आकार = माप(*val) * गिनती;
    निश्चित(आकार >= गिनती);
    apir_encode(enc, आकार, val, आकार);
}

अटल अंतरभूत व्योम apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t गिनती) {
    स्थिर माप_प्रकार आकार = माप(*val) * गिनती;
    निश्चित(आकार >= गिनती);
    apir_decode(dec, आकार, val, आकार);
}

/* size_t */

अटल अंतरभूत व्योम apir_encode_आकार_t(apir_encoder * enc, स्थिर माप_प्रकार * val) {
    स्थिर uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

अटल अंतरभूत व्योम apir_decode_आकार_t(apir_decoder * dec, माप_प्रकार * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

अटल अंतरभूत व्योम apir_encode_आकार_t_array(apir_encoder * enc, स्थिर माप_प्रकार * val, uint32_t गिनती) {
    अगर (माप(माप_प्रकार) == माप(uint64_t)) {
        apir_encode_uint64_t_array(enc, (स्थिर uint64_t *) val, गिनती);
    } अन्यथा {
        क्रम (uint32_t i = 0; i < गिनती; i++) {
            apir_encode_आकार_t(enc, &val[i]);
        }
    }
}

अटल अंतरभूत व्योम apir_decode_आकार_t_array(apir_decoder * dec, माप_प्रकार * val, uint32_t गिनती) {
    अगर (माप(माप_प्रकार) == माप(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, गिनती);
    } अन्यथा {
        क्रम (uint32_t i = 0; i < गिनती; i++) {
            apir_decode_आकार_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

अटल अंतरभूत व्योम apir_encode_blob_array(apir_encoder * enc, स्थिर व्योम * val, माप_प्रकार आकार) {
    apir_encode(enc, (आकार + 3) & ~3, val, आकार);
}

अटल अंतरभूत व्योम apir_decode_blob_array(apir_decoder * dec, व्योम * val, माप_प्रकार आकार) {
    apir_decode(dec, (आकार + 3) & ~3, val, आकार);
}

/* string */

अटल अंतरभूत व्योम apir_encode_char_array(apir_encoder * enc, स्थिर अक्षर * val, माप_प्रकार आकार) {
    निश्चित(आकार && म_माप(val) < आकार);
    apir_encode_blob_array(enc, val, आकार);
}

अटल अंतरभूत व्योम apir_decode_char_array(apir_decoder * dec, अक्षर * val, माप_प्रकार आकार) {
    apir_decode_blob_array(dec, val, आकार);
    अगर (आकार) {
        val[आकार - 1] = '\0';
    } अन्यथा {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

अटल अंतरभूत व्योम * apir_decoder_alloc_array(माप_प्रकार आकार, माप_प्रकार गिनती) {
    माप_प्रकार alloc_आकार;
    अगर (unlikely(__builtin_mul_overflow(आकार, गिनती, &alloc_आकार))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, आकार, गिनती);
        वापस शून्य;
    }

    वापस दो_स्मृति(alloc_आकार);
}

/* bool */

अटल अंतरभूत व्योम apir_encode_bool_t(apir_encoder * enc, स्थिर bool * val) {
    apir_encode(enc, माप(पूर्णांक), val, माप(bool));
}

अटल अंतरभूत व्योम apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, माप(पूर्णांक), val, माप(bool));
}

/* apir_buffer_type_host_handle_t */

अटल अंतरभूत व्योम apir_encode_apir_बफर_type_host_handle_t(apir_encoder *                         enc,
                                                              स्थिर apir_बफर_type_host_handle_t * val) {
    apir_encode(enc, माप(apir_बफर_type_host_handle_t), val, माप(apir_बफर_type_host_handle_t));
}

अटल अंतरभूत व्योम apir_decode_apir_बफर_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_बफर_type_host_handle_t * val) {
    apir_decode(dec, माप(apir_बफर_type_host_handle_t), val, माप(apir_बफर_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

अटल अंतरभूत व्योम apir_encode_apir_बफर_host_handle_t(apir_encoder * enc, स्थिर apir_बफर_host_handle_t * val) {
    apir_encode(enc, माप(apir_बफर_host_handle_t), val, माप(apir_बफर_host_handle_t));
}

अटल अंतरभूत व्योम apir_decode_apir_बफर_host_handle_t(apir_decoder * dec, apir_बफर_host_handle_t * val) {
    apir_decode(dec, माप(apir_बफर_host_handle_t), val, माप(apir_बफर_host_handle_t));
}

/* uintptr_t */

अटल अंतरभूत व्योम apir_encode_uintptr_t(apir_encoder * enc, स्थिर uintptr_t * val) {
    apir_encode(enc, माप(*val), val, माप(*val));
}

अटल अंतरभूत व्योम apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, माप(*val), val, माप(*val));
}
