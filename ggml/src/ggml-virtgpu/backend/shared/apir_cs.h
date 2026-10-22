#pragma once

#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cassert>
#অন্তর্ভাব <cstring>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

কাঠামো apir_encoder {
    অক্ষর *       cur;
    ধ্রুবক অক্ষর * start;
    ধ্রুবক অক্ষর * end;
    bool         fatal;
};

কাঠামো apir_decoder {
    ধ্রুবক অক্ষর * cur;
    ধ্রুবক অক্ষর * end;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

স্থির apir_decoder apir_new_decoder(ধ্রুবক অক্ষর * ptr, size_t আকার_2) {
    apir_decoder dec = {
        .cur   = ptr,
        .end   = ptr + আকার_2,
        .fatal = false,
    };

    ফেরত dec;
}

স্থির apir_encoder apir_new_encoder(অক্ষর * ptr, size_t আকার_2) {
    apir_encoder enc = {
        .cur   = ptr,
        .start = ptr,
        .end   = ptr + আকার_2,
        .fatal = false,
    };

    ফেরত enc;
}

/*
 * fatal flag handling
 */

স্থির অন্তঃরেখ শূন্য apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

স্থির অন্তঃরেখ শূন্য apir_encoder_set_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

স্থির অন্তঃরেখ bool apir_encoder_get_fatal(ধ্রুবক apir_encoder * enc) {
    ফেরত enc->fatal;
}

স্থির অন্তঃরেখ শূন্য apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

স্থির অন্তঃরেখ শূন্য apir_decoder_set_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

স্থির অন্তঃরেখ bool apir_decoder_get_fatal(ধ্রুবক apir_decoder * dec) {
    ফেরত dec->fatal;
}

/*
 * encode peek
 */

স্থির অন্তঃরেখ bool apir_decoder_peek_internal(apir_decoder * dec, size_t আকার_2, শূন্য * val, size_t val_আকার) {
    প্রতিজ্ঞা(val_আকার <= আকার_2);

    যদি (unlikely(আকার_2 > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        memset(val, 0, val_আকার);
        ফেরত false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(val, dec->cur, val_আকার);
    ফেরত true;
}

স্থির অন্তঃরেখ শূন্য apir_decoder_peek(apir_decoder * dec, size_t আকার_2, শূন্য * val, size_t val_আকার) {
    apir_decoder_peek_internal(dec, আকার_2, val, val_আকার);
}

স্থির অন্তঃরেখ ধ্রুবক শূন্য * apir_decoder_use_inplace(apir_decoder * dec, size_t আকার_2) {
    যদি (unlikely(আকার_2 > (size_t) (dec->end - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_set_fatal(dec);
        ফেরত ফাঁকা;
    }
    ধ্রুবক শূন্য * addr = dec->cur;
    dec->cur += আকার_2;

    ফেরত addr;
}

/*
 * read/write
 */

স্থির অন্তঃরেখ শূন্য apir_decoder_read(apir_decoder * dec, size_t আকার_2, শূন্য * val, size_t val_আকার) {
    যদি (apir_decoder_peek_internal(dec, আকার_2, val, val_আকার)) {
        dec->cur += আকার_2;
    }
}

স্থির অন্তঃরেখ অক্ষর * apir_encoder_write(apir_encoder * enc, size_t আকার_2, ধ্রুবক শূন্য * val, size_t val_আকার) {
    প্রতিজ্ঞা(val_আকার <= আকার_2);
    প্রতিজ্ঞা(আকার_2 <= ((size_t) (enc->end - enc->cur)));

    অক্ষর * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    memcpy(write_addr, val, val_আকার);
    enc->cur += আকার_2;

    ফেরত write_addr;
}

/*
 * encode/decode
 */

স্থির অন্তঃরেখ শূন্য apir_decode(apir_decoder * dec, size_t আকার_2, শূন্য * তথ্য, size_t তথ্য_আকার) {
    প্রতিজ্ঞা(আকার_2 % 4 == 0);
    apir_decoder_read(dec, আকার_2, তথ্য, তথ্য_আকার);
}

স্থির অন্তঃরেখ শূন্য apir_encode(apir_encoder * enc, size_t আকার_2, ধ্রুবক শূন্য * তথ্য, size_t তথ্য_আকার) {
    প্রতিজ্ঞা(আকার_2 % 4 == 0);
    apir_encoder_write(enc, আকার_2, তথ্য, তথ্য_আকার);
}

/*
 * typed encode/decode
 */

/* uint8_t */

স্থির অন্তঃরেখ শূন্য apir_encode_uint8_t(apir_encoder * enc, ধ্রুবক uint8_t * val) {
    apir_encode(enc, আকার(পূর্ণসংখ্যা), val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, আকার(পূর্ণসংখ্যা), val, আকার(*val));
}

/* uint64_t */

স্থির অন্তঃরেখ শূন্য apir_encode_uint64_t(apir_encoder * enc, ধ্রুবক uint64_t * val) {
    apir_encode(enc, 8, val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_encode_uint64_t_array(apir_encoder * enc, ধ্রুবক uint64_t * val, uint32_t গুনতি) {
    ধ্রুবক size_t আকার_2 = আকার(*val) * গুনতি;
    প্রতিজ্ঞা(আকার_2 >= গুনতি);
    apir_encode(enc, আকার_2, val, আকার_2);
}

স্থির অন্তঃরেখ শূন্য apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t গুনতি) {
    ধ্রুবক size_t আকার_2 = আকার(*val) * গুনতি;
    প্রতিজ্ঞা(আকার_2 >= গুনতি);
    apir_decode(dec, আকার_2, val, আকার_2);
}

স্থির অন্তঃরেখ ধ্রুবক uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t গুনতি) {
    ফেরত (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, গুনতি * আকার(uint64_t));
}

/* int32_t */

স্থির অন্তঃরেখ শূন্য apir_encode_int32_t(apir_encoder * enc, ধ্রুবক int32_t * val) {
    apir_encode(enc, 4, val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_encode_int32_t_array(apir_encoder * enc, ধ্রুবক int32_t * val, uint32_t গুনতি) {
    ধ্রুবক size_t আকার_2 = আকার(*val) * গুনতি;
    প্রতিজ্ঞা(আকার_2 >= গুনতি);
    apir_encode(enc, আকার_2, val, আকার_2);
}

স্থির অন্তঃরেখ শূন্য apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t গুনতি) {
    ধ্রুবক size_t আকার_2 = আকার(*val) * গুনতি;
    প্রতিজ্ঞা(আকার_2 >= গুনতি);
    apir_decode(dec, আকার_2, val, আকার_2);
}

/* array size (uint64_t) */

স্থির অন্তঃরেখ শূন্য apir_encode_array_আকার(apir_encoder * enc, uint64_t আকার_2) {
    apir_encode_uint64_t(enc, &আকার_2);
}

স্থির অন্তঃরেখ uint64_t apir_decode_array_আকার(apir_decoder * dec, uint64_t expected_আকার) {
    uint64_t আকার_2;
    apir_decode_uint64_t(dec, &আকার_2);
    যদি (আকার_2 != expected_আকার) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_set_fatal(dec);
        আকার_2 = 0;
    }
    ফেরত আকার_2;
}

স্থির অন্তঃরেখ uint64_t apir_decode_array_আকার_unchecked(apir_decoder * dec) {
    uint64_t আকার_2;
    apir_decode_uint64_t(dec, &আকার_2);
    ফেরত আকার_2;
}

/* non-array pointer */

স্থির অন্তঃরেখ bool apir_encode_simple_pointer(apir_encoder * enc, ধ্রুবক শূন্য * val) {
    apir_encode_array_আকার(enc, val ? 1 : 0);
    ফেরত val;
}

স্থির অন্তঃরেখ bool apir_decode_simple_pointer(apir_decoder * dec) {
    ফেরত apir_decode_array_আকার_unchecked(dec);
}

/* uint32_t */

স্থির অন্তঃরেখ শূন্য apir_encode_uint32_t(apir_encoder * enc, ধ্রুবক uint32_t * val) {
    apir_encode(enc, 4, val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_encode_uint32_t_array(apir_encoder * enc, ধ্রুবক uint32_t * val, uint32_t গুনতি) {
    ধ্রুবক size_t আকার_2 = আকার(*val) * গুনতি;
    প্রতিজ্ঞা(আকার_2 >= গুনতি);
    apir_encode(enc, আকার_2, val, আকার_2);
}

স্থির অন্তঃরেখ শূন্য apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t গুনতি) {
    ধ্রুবক size_t আকার_2 = আকার(*val) * গুনতি;
    প্রতিজ্ঞা(আকার_2 >= গুনতি);
    apir_decode(dec, আকার_2, val, আকার_2);
}

/* size_t */

স্থির অন্তঃরেখ শূন্য apir_encode_আকার_t(apir_encoder * enc, ধ্রুবক size_t * val) {
    ধ্রুবক uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

স্থির অন্তঃরেখ শূন্য apir_decode_আকার_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

স্থির অন্তঃরেখ শূন্য apir_encode_আকার_t_array(apir_encoder * enc, ধ্রুবক size_t * val, uint32_t গুনতি) {
    যদি (আকার(size_t) == আকার(uint64_t)) {
        apir_encode_uint64_t_array(enc, (ধ্রুবক uint64_t *) val, গুনতি);
    } অন্যথা {
        জন্য (uint32_t i = 0; i < গুনতি; i++) {
            apir_encode_আকার_t(enc, &val[i]);
        }
    }
}

স্থির অন্তঃরেখ শূন্য apir_decode_আকার_t_array(apir_decoder * dec, size_t * val, uint32_t গুনতি) {
    যদি (আকার(size_t) == আকার(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, গুনতি);
    } অন্যথা {
        জন্য (uint32_t i = 0; i < গুনতি; i++) {
            apir_decode_আকার_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

স্থির অন্তঃরেখ শূন্য apir_encode_blob_array(apir_encoder * enc, ধ্রুবক শূন্য * val, size_t আকার_2) {
    apir_encode(enc, (আকার_2 + 3) & ~3, val, আকার_2);
}

স্থির অন্তঃরেখ শূন্য apir_decode_blob_array(apir_decoder * dec, শূন্য * val, size_t আকার_2) {
    apir_decode(dec, (আকার_2 + 3) & ~3, val, আকার_2);
}

/* string */

স্থির অন্তঃরেখ শূন্য apir_encode_char_array(apir_encoder * enc, ধ্রুবক অক্ষর * val, size_t আকার_2) {
    প্রতিজ্ঞা(আকার_2 && strlen(val) < আকার_2);
    apir_encode_blob_array(enc, val, আকার_2);
}

স্থির অন্তঃরেখ শূন্য apir_decode_char_array(apir_decoder * dec, অক্ষর * val, size_t আকার_2) {
    apir_decode_blob_array(dec, val, আকার_2);
    যদি (আকার_2) {
        val[আকার_2 - 1] = '\0';
    } অন্যথা {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_set_fatal(dec);
    }
}

/* (temp) buffer allocation */

স্থির অন্তঃরেখ শূন্য * apir_decoder_alloc_array(size_t আকার_2, size_t গুনতি) {
    size_t alloc_আকার;
    যদি (unlikely(__builtin_mul_overflow(আকার_2, গুনতি, &alloc_আকার))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, আকার_2, গুনতি);
        ফেরত ফাঁকা;
    }

    ফেরত malloc(alloc_আকার);
}

/* bool */

স্থির অন্তঃরেখ শূন্য apir_encode_bool_t(apir_encoder * enc, ধ্রুবক bool * val) {
    apir_encode(enc, আকার(পূর্ণসংখ্যা), val, আকার(bool));
}

স্থির অন্তঃরেখ শূন্য apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, আকার(পূর্ণসংখ্যা), val, আকার(bool));
}

/* apir_buffer_type_host_handle_t */

স্থির অন্তঃরেখ শূন্য apir_encode_apir_বাফার_type_host_handle_t(apir_encoder *                         enc,
                                                              ধ্রুবক apir_বাফার_type_host_handle_t * val) {
    apir_encode(enc, আকার(apir_বাফার_type_host_handle_t), val, আকার(apir_বাফার_type_host_handle_t));
}

স্থির অন্তঃরেখ শূন্য apir_decode_apir_বাফার_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_বাফার_type_host_handle_t * val) {
    apir_decode(dec, আকার(apir_বাফার_type_host_handle_t), val, আকার(apir_বাফার_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

স্থির অন্তঃরেখ শূন্য apir_encode_apir_বাফার_host_handle_t(apir_encoder * enc, ধ্রুবক apir_বাফার_host_handle_t * val) {
    apir_encode(enc, আকার(apir_বাফার_host_handle_t), val, আকার(apir_বাফার_host_handle_t));
}

স্থির অন্তঃরেখ শূন্য apir_decode_apir_বাফার_host_handle_t(apir_decoder * dec, apir_বাফার_host_handle_t * val) {
    apir_decode(dec, আকার(apir_বাফার_host_handle_t), val, আকার(apir_বাফার_host_handle_t));
}

/* uintptr_t */

স্থির অন্তঃরেখ শূন্য apir_encode_uintptr_t(apir_encoder * enc, ধ্রুবক uintptr_t * val) {
    apir_encode(enc, আকার(*val), val, আকার(*val));
}

স্থির অন্তঃরেখ শূন্য apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, আকার(*val), val, আকার(*val));
}
