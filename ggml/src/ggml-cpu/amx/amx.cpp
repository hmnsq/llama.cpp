#अन्तर्भाव "amx.h"
#अन्तर्भाव "common.h"
#अन्तर्भाव "mmq.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "traits.h"

#जर defined(__linux__)
#अन्तर्भाव <sys/syscall.h>
#अन्तर्भाव <unistd.h>
#endif

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <memory>

#जर defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
नामाकाश ggml::cpu::amx {
वर्ग tensor_traits : सार्वजनिक ggml::cpu::tensor_traits {
    तार्किक work_size(पूर्णांक /* n_threads */, स्थिरांक संरचना ggml_tensor * op, माप_प्रकार & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        परत सत्य;
    }

    तार्किक compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * op) override {
        जर (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            परत सत्य;
        }
        परत असत्य;
    }
};

स्थिर ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, संरचना ggml_tensor *) {
    स्थिर tensor_traits traits;
    परत &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
स्थिर शून्य ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    मुक्त(buffer->context);
}

स्थिर शून्य * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    परत (शून्य *) (buffer->context);
}

स्थिर गणना ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor) {
    tensor->extra = (शून्य *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor,
                                                  uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    स_रखो((अक्षर *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

स्थिर शून्य ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor,
                                               स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    जर (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } अन्यथा {
        स_नकल((अक्षर *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}
*/

स्थिर शून्य ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    स_रखो(buffer->context, value, buffer->size);
}

स्थिर ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

स्थिर स्थिरांक अक्षर * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    परत "AMX";

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    शून्य * data = ggml_aligned_malloc(size);
    जर (data == रिक्त) {
        ख_लिखो(मानक_त्रुटि, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        परत रिक्त;
    }

    परत ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

स्थिर माप_प्रकार ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    परत TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

नामाकाश ggml::cpu::amx {
वर्ग extra_buffer_type : ggml::cpu::extra_buffer_type {
    तार्किक supports_op(ggml_backend_dev_t, स्थिरांक संरचना ggml_tensor * op) override {
        जर (op->op != GGML_OP_MUL_MAT) {
            परत असत्य;
        }
        स्वतः * src0 = op->src[0];
        स्वतः * src1 = op->src[1];

        जर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            परत असत्य;
        }
        जर (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            परत असत्य;
        }
        जर (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            परत असत्य;
        }
        जर (op->ne[0] % (TILE_N * 2)) {
            परत असत्य;
        }
        पूर्णांक alignment;
        निवड (src0->type) {
            स्थिती GGML_TYPE_Q4_0:
            स्थिती GGML_TYPE_Q4_1:
            स्थिती GGML_TYPE_Q8_0:
                alignment = TILE_K;
                थांबा;
            स्थिती GGML_TYPE_Q4_K:
            स्थिती GGML_TYPE_Q5_K:
            स्थिती GGML_TYPE_Q6_K:
            स्थिती GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                थांबा;
            स्थिती GGML_TYPE_F16:
                alignment = 16;
                थांबा;
            पूर्वनिर्धारित:
                परत असत्य;
        }
        जर (src0->ne[0] % alignment) {
            परत असत्य;
        }
        जर (src1->type != GGML_TYPE_F32) {
            परत असत्य;
        }
        परत सत्य;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिरांक संरचना ggml_tensor * op) override {
        जर (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            परत (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        परत nullptr;
    }
};
}  // namespace ggml::cpu::amx

स्थिर माप_प्रकार ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, स्थिरांक ggml_tensor * tensor) {
    परत ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

स्थिर तार्किक ggml_amx_init() {
#जर defined(__linux__)
    जर (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        ख_लिखो(मानक_त्रुटि, "AMX is not ready to be used!\n");
        परत असत्य;
    }
    परत सत्य;
#किंवाजर defined(_WIN32)
    परत सत्य;
#अन्यथा
    परत असत्य;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ रिक्त,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ नवीन ggml::cpu::amx::extra_buffer_type(),
    };

    जर (!ggml_amx_init()) {
        परत nullptr;
    }

    परत &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
