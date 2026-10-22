#समावेश "amx.h"
#समावेश "common.h"
#समावेश "mmq.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-cpu.h"
#समावेश "traits.h"

#अगर defined(__linux__)
#समावेश <sys/syscall.h>
#समावेश <unistd.h>
#पूर्ण_अगर

#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <memory>

#अगर defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
श्रेणी tensor_traits : खुला ggml::cpu::tensor_traits {
    bool work_आकार(पूर्णांक /* n_threads */, स्थिर काष्ठा ggml_tensor * op, माप_प्रकार & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        वापस true;
    }

    bool compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * op) override {
        अगर (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            वापस true;
        }
        वापस false;
    }
};

अटल ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_बफर_t, काष्ठा ggml_tensor *) {
    अटल tensor_traits traits;
    वापस &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
अटल व्योम ggml_backend_amx_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    मुक्त(बफर->context);
}

अटल व्योम * ggml_backend_amx_बफर_get_base(ggml_backend_बफर_t बफर) {
    वापस (व्योम *) (बफर->context);
}

अटल क्रमागत ggml_status ggml_backend_amx_बफर_init_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor) {
    tensor->extra = (व्योम *) ggml::cpu::amx::get_tensor_traits(बफर, tensor);

    GGML_UNUSED(बफर);
    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_amx_बफर_memset_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor,
                                                  uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    स_रखो((अक्षर *) tensor->तथ्य + offset, value, size);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_amx_बफर_set_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor,
                                               स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    अगर (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, तथ्य, offset, size);
    } अन्यथा {
        स_नकल((अक्षर *) tensor->तथ्य + offset, तथ्य, size);
    }

    GGML_UNUSED(बफर);
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

अटल व्योम ggml_backend_amx_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    स_रखो(बफर->context, value, बफर->size);
}

अटल ggml_backend_बफर_i ggml_backend_amx_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_amx_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_amx_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_amx_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_बफर_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_बफर_clear,
    /* .reset           = */ nullptr,
};

अटल स्थिर अक्षर * ggml_backend_amx_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस "AMX";

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_t ggml_backend_amx_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    व्योम * तथ्य = ggml_aligned_malloc(size);
    अगर (तथ्य == शून्य) {
        ख_लिखो(मानक_त्रुटि, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        वापस शून्य;
    }

    वापस ggml_backend_बफर_init(buft, ggml_backend_amx_बफर_interface, तथ्य, size);
}

अटल माप_प्रकार ggml_backend_amx_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {
श्रेणी extra_बफर_type : ggml::cpu::extra_बफर_type {
    bool supports_op(ggml_backend_dev_t, स्थिर काष्ठा ggml_tensor * op) override {
        अगर (op->op != GGML_OP_MUL_MAT) {
            वापस false;
        }
        स्वतः * src0 = op->src[0];
        स्वतः * src1 = op->src[1];

        अगर (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            वापस false;
        }
        अगर (!src0->बफर || src0->बफर->buft != ggml_backend_amx_बफर_type()) {
            वापस false;
        }
        अगर (src1->बफर && !ggml_backend_buft_is_host(src1->बफर->buft)) {
            वापस false;
        }
        अगर (op->ne[0] % (TILE_N * 2)) {
            वापस false;
        }
        पूर्णांक alignment;
        चयन (src0->type) {
            हाल GGML_TYPE_Q4_0:
            हाल GGML_TYPE_Q4_1:
            हाल GGML_TYPE_Q8_0:
                alignment = TILE_K;
                अवरोध;
            हाल GGML_TYPE_Q4_K:
            हाल GGML_TYPE_Q5_K:
            हाल GGML_TYPE_Q6_K:
            हाल GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                अवरोध;
            हाल GGML_TYPE_F16:
                alignment = 16;
                अवरोध;
            शेष:
                वापस false;
        }
        अगर (src0->ne[0] % alignment) {
            वापस false;
        }
        अगर (src1->type != GGML_TYPE_F32) {
            वापस false;
        }
        वापस true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(स्थिर काष्ठा ggml_tensor * op) override {
        अगर (op->op == GGML_OP_MUL_MAT && op->src[0]->बफर &&
            op->src[0]->बफर->buft == ggml_backend_amx_बफर_type()) {
            वापस (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        वापस nullptr;
    }
};
}  // namespace ggml::cpu::amx

अटल माप_प्रकार ggml_backend_amx_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    वापस ggml_backend_amx_get_alloc_आकार(tensor);

    GGML_UNUSED(buft);
}

#घोषणा ARCH_GET_XCOMP_PERM     0x1022
#घोषणा ARCH_REQ_XCOMP_PERM     0x1023
#घोषणा XFEATURE_XTILECFG       17
#घोषणा XFEATURE_XTILEDATA      18

अटल bool ggml_amx_init() {
#अगर defined(__linux__)
    अगर (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        ख_लिखो(मानक_त्रुटि, "AMX is not ready to be used!\n");
        वापस false;
    }
    वापस true;
#या_अगर defined(_WIN32)
    वापस true;
#अन्यथा
    वापस false;
#पूर्ण_अगर
}

ggml_backend_बफर_type_t ggml_backend_amx_बफर_type() {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_बफर_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_बफर_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_बफर_type_alloc_बफर,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_बफर_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_बफर_type_get_alloc_आकार,
                        /* .get_alloc_size_n    = */ शून्य,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ नया ggml::cpu::amx::extra_बफर_type(),
    };

    अगर (!ggml_amx_init()) {
        वापस nullptr;
    }

    वापस &ggml_backend_बफर_type_amx;
}

#पूर्ण_अगर  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
