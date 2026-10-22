#సూచన once

#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cassert>
#అన్తర్భావ <cstring>

#నిర్వచించు likely(x)   __builtin_expect(!!(x), 1)
#నిర్వచించు unlikely(x) __builtin_expect(!!(x), 0)

నిర్మాణం apir_encoder {
    అక్షరం *       cur;
    స్థిరరాశి అక్షరం * ప్రారంభం;
    స్థిరరాశి అక్షరం * చివర;
    bool         fatal;
};

నిర్మాణం apir_decoder {
    స్థిరరాశి అక్షరం * cur;
    స్థిరరాశి అక్షరం * చివర;
    bool         fatal;
};

/*
 * new encoder and decoder
 */

స్థిర apir_decoder apir_new_decoder(స్థిరరాశి అక్షరం * ptr, size_t కొలత) {
    apir_decoder dec = {
        .cur   = ptr,
        .చివర   = ptr + కొలత,
        .fatal = false,
    };

    తిరిగి dec;
}

స్థిర apir_encoder apir_new_encoder(అక్షరం * ptr, size_t కొలత) {
    apir_encoder enc = {
        .cur   = ptr,
        .ప్రారంభం = ptr,
        .చివర   = ptr + కొలత,
        .fatal = false,
    };

    తిరిగి enc;
}

/*
 * fatal flag handling
 */

స్థిర అంతఃవరుస శూన్య apir_encoder_reset_fatal(apir_encoder * enc) {
    enc->fatal = false;
}

స్థిర అంతఃవరుస శూన్య apir_encoder_సమితి_fatal(apir_encoder * enc) {
    enc->fatal = true;
}

స్థిర అంతఃవరుస bool apir_encoder_get_fatal(స్థిరరాశి apir_encoder * enc) {
    తిరిగి enc->fatal;
}

స్థిర అంతఃవరుస శూన్య apir_decoder_reset_fatal(apir_decoder * dec) {
    dec->fatal = false;
}

స్థిర అంతఃవరుస శూన్య apir_decoder_సమితి_fatal(apir_decoder * dec) {
    dec->fatal = true;
}

స్థిర అంతఃవరుస bool apir_decoder_get_fatal(స్థిరరాశి apir_decoder * dec) {
    తిరిగి dec->fatal;
}

/*
 * encode peek
 */

స్థిర అంతఃవరుస bool apir_decoder_peek_అంతర్గత(apir_decoder * dec, size_t కొలత, శూన్య * val, size_t val_కొలత) {
    ప్రతిజ్ఞా(val_కొలత <= కొలత);

    ఒకవేళ (unlikely(కొలత > (size_t) (dec->చివర - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_సమితి_fatal(dec);
        memset(val, 0, val_కొలత);
        తిరిగి false;
    }

    /* we should not rely on the compiler to optimize away memcpy... */
    జ్ఞాపకం_నకలు(val, dec->cur, val_కొలత);
    తిరిగి true;
}

స్థిర అంతఃవరుస శూన్య apir_decoder_peek(apir_decoder * dec, size_t కొలత, శూన్య * val, size_t val_కొలత) {
    apir_decoder_peek_అంతర్గత(dec, కొలత, val, val_కొలత);
}

స్థిర అంతఃవరుస స్థిరరాశి శూన్య * apir_decoder_use_inplace(apir_decoder * dec, size_t కొలత) {
    ఒకవేళ (unlikely(కొలత > (size_t) (dec->చివర - dec->cur))) {
        GGML_LOG_ERROR("%s: reading too much from the decoder ...\n", __func__);
        apir_decoder_సమితి_fatal(dec);
        తిరిగి ఖాళీ;
    }
    స్థిరరాశి శూన్య * addr = dec->cur;
    dec->cur += కొలత;

    తిరిగి addr;
}

/*
 * read/write
 */

స్థిర అంతఃవరుస శూన్య apir_decoder_read(apir_decoder * dec, size_t కొలత, శూన్య * val, size_t val_కొలత) {
    ఒకవేళ (apir_decoder_peek_అంతర్గత(dec, కొలత, val, val_కొలత)) {
        dec->cur += కొలత;
    }
}

స్థిర అంతఃవరుస అక్షరం * apir_encoder_write(apir_encoder * enc, size_t కొలత, స్థిరరాశి శూన్య * val, size_t val_కొలత) {
    ప్రతిజ్ఞా(val_కొలత <= కొలత);
    ప్రతిజ్ఞా(కొలత <= ((size_t) (enc->చివర - enc->cur)));

    అక్షరం * write_addr = enc->cur;
    /* we should not rely on the compiler to optimize away memcpy... */
    జ్ఞాపకం_నకలు(write_addr, val, val_కొలత);
    enc->cur += కొలత;

    తిరిగి write_addr;
}

/*
 * encode/decode
 */

స్థిర అంతఃవరుస శూన్య apir_decode(apir_decoder * dec, size_t కొలత, శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    ప్రతిజ్ఞా(కొలత % 4 == 0);
    apir_decoder_read(dec, కొలత, దత్తాంశం, దత్తాంశం_కొలత);
}

స్థిర అంతఃవరుస శూన్య apir_encode(apir_encoder * enc, size_t కొలత, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    ప్రతిజ్ఞా(కొలత % 4 == 0);
    apir_encoder_write(enc, కొలత, దత్తాంశం, దత్తాంశం_కొలత);
}

/*
 * typed encode/decode
 */

/* uint8_t */

స్థిర అంతఃవరుస శూన్య apir_encode_uint8_t(apir_encoder * enc, స్థిరరాశి uint8_t * val) {
    apir_encode(enc, పరిమాణం(పూర్ణాంకం), val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_decode_uint8_t(apir_decoder * dec, uint8_t * val) {
    apir_decode(dec, పరిమాణం(పూర్ణాంకం), val, పరిమాణం(*val));
}

/* uint64_t */

స్థిర అంతఃవరుస శూన్య apir_encode_uint64_t(apir_encoder * enc, స్థిరరాశి uint64_t * val) {
    apir_encode(enc, 8, val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_decode_uint64_t(apir_decoder * dec, uint64_t * val) {
    apir_decode(dec, 8, val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_encode_uint64_t_array(apir_encoder * enc, స్థిరరాశి uint64_t * val, uint32_t లెక్క) {
    స్థిరరాశి size_t కొలత = పరిమాణం(*val) * లెక్క;
    ప్రతిజ్ఞా(కొలత >= లెక్క);
    apir_encode(enc, కొలత, val, కొలత);
}

స్థిర అంతఃవరుస శూన్య apir_decode_uint64_t_array(apir_decoder * dec, uint64_t * val, uint32_t లెక్క) {
    స్థిరరాశి size_t కొలత = పరిమాణం(*val) * లెక్క;
    ప్రతిజ్ఞా(కొలత >= లెక్క);
    apir_decode(dec, కొలత, val, కొలత);
}

స్థిర అంతఃవరుస స్థిరరాశి uint64_t * apir_decode_uint64_t_array_inplace(apir_decoder * dec, uint32_t లెక్క) {
    తిరిగి (uint64_t *) (uintptr_t) apir_decoder_use_inplace(dec, లెక్క * పరిమాణం(uint64_t));
}

/* int32_t */

స్థిర అంతఃవరుస శూన్య apir_encode_int32_t(apir_encoder * enc, స్థిరరాశి int32_t * val) {
    apir_encode(enc, 4, val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_decode_int32_t(apir_decoder * dec, int32_t * val) {
    apir_decode(dec, 4, val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_encode_int32_t_array(apir_encoder * enc, స్థిరరాశి int32_t * val, uint32_t లెక్క) {
    స్థిరరాశి size_t కొలత = పరిమాణం(*val) * లెక్క;
    ప్రతిజ్ఞా(కొలత >= లెక్క);
    apir_encode(enc, కొలత, val, కొలత);
}

స్థిర అంతఃవరుస శూన్య apir_decode_int32_t_array(apir_decoder * dec, int32_t * val, uint32_t లెక్క) {
    స్థిరరాశి size_t కొలత = పరిమాణం(*val) * లెక్క;
    ప్రతిజ్ఞా(కొలత >= లెక్క);
    apir_decode(dec, కొలత, val, కొలత);
}

/* array size (uint64_t) */

స్థిర అంతఃవరుస శూన్య apir_encode_array_కొలత(apir_encoder * enc, uint64_t కొలత) {
    apir_encode_uint64_t(enc, &కొలత);
}

స్థిర అంతఃవరుస uint64_t apir_decode_array_కొలత(apir_decoder * dec, uint64_t expected_కొలత) {
    uint64_t కొలత;
    apir_decode_uint64_t(dec, &కొలత);
    ఒకవేళ (కొలత != expected_కొలత) {
        GGML_LOG_ERROR("%s: Couldn't decode array from the decoder\n", __func__);
        apir_decoder_సమితి_fatal(dec);
        కొలత = 0;
    }
    తిరిగి కొలత;
}

స్థిర అంతఃవరుస uint64_t apir_decode_array_కొలత_unchecked(apir_decoder * dec) {
    uint64_t కొలత;
    apir_decode_uint64_t(dec, &కొలత);
    తిరిగి కొలత;
}

/* non-array pointer */

స్థిర అంతఃవరుస bool apir_encode_simple_సూచిక(apir_encoder * enc, స్థిరరాశి శూన్య * val) {
    apir_encode_array_కొలత(enc, val ? 1 : 0);
    తిరిగి val;
}

స్థిర అంతఃవరుస bool apir_decode_simple_సూచిక(apir_decoder * dec) {
    తిరిగి apir_decode_array_కొలత_unchecked(dec);
}

/* uint32_t */

స్థిర అంతఃవరుస శూన్య apir_encode_uint32_t(apir_encoder * enc, స్థిరరాశి uint32_t * val) {
    apir_encode(enc, 4, val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_decode_uint32_t(apir_decoder * dec, uint32_t * val) {
    apir_decode(dec, 4, val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_encode_uint32_t_array(apir_encoder * enc, స్థిరరాశి uint32_t * val, uint32_t లెక్క) {
    స్థిరరాశి size_t కొలత = పరిమాణం(*val) * లెక్క;
    ప్రతిజ్ఞా(కొలత >= లెక్క);
    apir_encode(enc, కొలత, val, కొలత);
}

స్థిర అంతఃవరుస శూన్య apir_decode_uint32_t_array(apir_decoder * dec, uint32_t * val, uint32_t లెక్క) {
    స్థిరరాశి size_t కొలత = పరిమాణం(*val) * లెక్క;
    ప్రతిజ్ఞా(కొలత >= లెక్క);
    apir_decode(dec, కొలత, val, కొలత);
}

/* size_t */

స్థిర అంతఃవరుస శూన్య apir_encode_కొలత_t(apir_encoder * enc, స్థిరరాశి size_t * val) {
    స్థిరరాశి uint64_t tmp = *val;
    apir_encode_uint64_t(enc, &tmp);
}

స్థిర అంతఃవరుస శూన్య apir_decode_కొలత_t(apir_decoder * dec, size_t * val) {
    uint64_t tmp;
    apir_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

స్థిర అంతఃవరుస శూన్య apir_encode_కొలత_t_array(apir_encoder * enc, స్థిరరాశి size_t * val, uint32_t లెక్క) {
    ఒకవేళ (పరిమాణం(size_t) == పరిమాణం(uint64_t)) {
        apir_encode_uint64_t_array(enc, (స్థిరరాశి uint64_t *) val, లెక్క);
    } లేకపోతే {
        కొరకు (uint32_t i = 0; i < లెక్క; i++) {
            apir_encode_కొలత_t(enc, &val[i]);
        }
    }
}

స్థిర అంతఃవరుస శూన్య apir_decode_కొలత_t_array(apir_decoder * dec, size_t * val, uint32_t లెక్క) {
    ఒకవేళ (పరిమాణం(size_t) == పరిమాణం(uint64_t)) {
        apir_decode_uint64_t_array(dec, (uint64_t *) val, లెక్క);
    } లేకపోతే {
        కొరకు (uint32_t i = 0; i < లెక్క; i++) {
            apir_decode_కొలత_t(dec, &val[i]);
        }
    }
}

/* opaque blob */

స్థిర అంతఃవరుస శూన్య apir_encode_blob_array(apir_encoder * enc, స్థిరరాశి శూన్య * val, size_t కొలత) {
    apir_encode(enc, (కొలత + 3) & ~3, val, కొలత);
}

స్థిర అంతఃవరుస శూన్య apir_decode_blob_array(apir_decoder * dec, శూన్య * val, size_t కొలత) {
    apir_decode(dec, (కొలత + 3) & ~3, val, కొలత);
}

/* string */

స్థిర అంతఃవరుస శూన్య apir_encode_char_array(apir_encoder * enc, స్థిరరాశి అక్షరం * val, size_t కొలత) {
    ప్రతిజ్ఞా(కొలత && అక్షరశ్రేణి_పొడవు(val) < కొలత);
    apir_encode_blob_array(enc, val, కొలత);
}

స్థిర అంతఃవరుస శూన్య apir_decode_char_array(apir_decoder * dec, అక్షరం * val, size_t కొలత) {
    apir_decode_blob_array(dec, val, కొలత);
    ఒకవేళ (కొలత) {
        val[కొలత - 1] = '\0';
    } లేకపోతే {
        GGML_LOG_ERROR("%s: Couldn't decode the blog array\n", __func__);
        apir_decoder_సమితి_fatal(dec);
    }
}

/* (temp) buffer allocation */

స్థిర అంతఃవరుస శూన్య * apir_decoder_alloc_array(size_t కొలత, size_t లెక్క) {
    size_t alloc_కొలత;
    ఒకవేళ (unlikely(__builtin_mul_overflow(కొలత, లెక్క, &alloc_కొలత))) {
        GGML_LOG_ERROR("%s: overflow in array allocation of %zu * %zu bytes\n", __func__, కొలత, లెక్క);
        తిరిగి ఖాళీ;
    }

    తిరిగి జ్ఞాపకం_ఇవ్వు(alloc_కొలత);
}

/* bool */

స్థిర అంతఃవరుస శూన్య apir_encode_bool_t(apir_encoder * enc, స్థిరరాశి bool * val) {
    apir_encode(enc, పరిమాణం(పూర్ణాంకం), val, పరిమాణం(bool));
}

స్థిర అంతఃవరుస శూన్య apir_decode_bool_t(apir_decoder * dec, bool * val) {
    apir_decode(dec, పరిమాణం(పూర్ణాంకం), val, పరిమాణం(bool));
}

/* apir_buffer_type_host_handle_t */

స్థిర అంతఃవరుస శూన్య apir_encode_apir_బఫరు_type_host_handle_t(apir_encoder *                         enc,
                                                              స్థిరరాశి apir_బఫరు_type_host_handle_t * val) {
    apir_encode(enc, పరిమాణం(apir_బఫరు_type_host_handle_t), val, పరిమాణం(apir_బఫరు_type_host_handle_t));
}

స్థిర అంతఃవరుస శూన్య apir_decode_apir_బఫరు_type_host_handle_t(apir_decoder *                   dec,
                                                              apir_బఫరు_type_host_handle_t * val) {
    apir_decode(dec, పరిమాణం(apir_బఫరు_type_host_handle_t), val, పరిమాణం(apir_బఫరు_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

స్థిర అంతఃవరుస శూన్య apir_encode_apir_బఫరు_host_handle_t(apir_encoder * enc, స్థిరరాశి apir_బఫరు_host_handle_t * val) {
    apir_encode(enc, పరిమాణం(apir_బఫరు_host_handle_t), val, పరిమాణం(apir_బఫరు_host_handle_t));
}

స్థిర అంతఃవరుస శూన్య apir_decode_apir_బఫరు_host_handle_t(apir_decoder * dec, apir_బఫరు_host_handle_t * val) {
    apir_decode(dec, పరిమాణం(apir_బఫరు_host_handle_t), val, పరిమాణం(apir_బఫరు_host_handle_t));
}

/* uintptr_t */

స్థిర అంతఃవరుస శూన్య apir_encode_uintptr_t(apir_encoder * enc, స్థిరరాశి uintptr_t * val) {
    apir_encode(enc, పరిమాణం(*val), val, పరిమాణం(*val));
}

స్థిర అంతఃవరుస శూన్య apir_decode_uintptr_t(apir_decoder * dec, uintptr_t * val) {
    apir_decode(dec, పరిమాణం(*val), val, పరిమాణం(*val));
}
