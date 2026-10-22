// Note: porting this file to C++ is a work in progress

#अगर_घोषित _WIN32
#घोषणा WIN32_LEAN_AND_MEAN
#अगर_अघोषित NOMINMAX
#   define NOMINMAX
#पूर्ण_अगर
#समावेश <windows.h>
#पूर्ण_अगर

#समावेश "ggml-backend.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-alloc.h"
#समावेश "ggml-impl.h"

#समावेश <assert.h>
#समावेश <limits.h>
#समावेश <stdarg.h>
#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <string.h>
#समावेश <algorithm>
#समावेश <unordered_map>
#समावेश <vector>

#अगर_घोषित __APPLE__
#समावेश <sys/types.h>
#समावेश <sys/sysctl.h>
#पूर्ण_अगर


// backend buffer type

स्थिर अक्षर * ggml_backend_buft_name(ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(buft);
    वापस buft->iface.get_name(buft);
}

ggml_backend_बफर_t ggml_backend_buft_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    GGML_ASSERT(buft);
    अगर (size == 0) {
        // return a dummy buffer for zero-sized allocations
        वापस ggml_backend_बफर_init(buft, {}, शून्य, 0);
    }
    वापस buft->iface.alloc_बफर(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
काष्ठा ggml_backend_buft_alloc_बफर_n_plan_item {
    माप_प्रकार size;  // total bytes for this buffer
    पूर्णांक    first; // first tensor index (inclusive)
    पूर्णांक    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_बफर_n_plan_t = std::vector<ggml_backend_buft_alloc_बफर_n_plan_item>;

अटल ggml_backend_buft_alloc_बफर_n_plan_t ggml_backend_buft_alloc_बफर_n_plan(
        ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    ggml_backend_buft_alloc_बफर_n_plan_t plan;

    स्थिर माप_प्रकार alignment = ggml_backend_buft_get_alignment(buft);
    स्थिर माप_प्रकार max_size  = ggml_backend_buft_get_max_आकार(buft);

    माप_प्रकार cur_buf_आकार = 0;
    पूर्णांक    first        = 0;

    क्रम (पूर्णांक i = 0; i < n_tensors; i++) {
        माप_प्रकार this_आकार = 0;
        काष्ठा ggml_tensor * t = tensors[i];
        अगर (t->तथ्य == शून्य && t->view_src == शून्य) {
            this_आकार = GGML_PAD(ggml_backend_buft_get_alloc_आकार(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        अगर (cur_buf_आकार > 0 && (cur_buf_आकार + this_आकार) > max_size) {
            plan.push_back({ cur_buf_आकार, first, i });
            cur_buf_आकार = this_आकार;
            first        = i;
        } अन्यथा {
            cur_buf_आकार += this_आकार;
        }
    }

    अगर (cur_buf_आकार > 0) {
        plan.push_back({ cur_buf_आकार, first, n_tensors });
    }

    वापस plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
अटल ggml_backend_बफर_t ggml_backend_buft_alloc_बफर_n_default(ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिर ggml_backend_buft_alloc_बफर_n_plan_t plan = ggml_backend_buft_alloc_बफर_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_बफर_t> buffers;
    buffers.reserve(plan.size());

    क्रम (स्थिर ggml_backend_buft_alloc_बफर_n_plan_item & item : plan) {
        ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(buft, item.size);
        अगर (बफर == शून्य) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            क्रम (ggml_backend_बफर_t b : buffers) {
                ggml_backend_बफर_free(b);
            }
            वापस शून्य;
        }

        काष्ठा ggml_tallocr tallocr = ggml_tallocr_new(बफर);

        // allocate tensors in the current buffer
        काष्ठा ggml_tensor * t_failed = शून्य;
        क्रम (पूर्णांक j = item.first; j < item.last; j++) {
            काष्ठा ggml_tensor * t = tensors[j];
            अगर (t->तथ्य == शून्य) {
                अगर (t->view_src == शून्य) {
                    अगर (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        अवरोध;
                    }
                } अन्यथा अगर (t->बफर == शून्य) {
                    अगर (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        अवरोध;
                    }
                }
            } अन्यथा {
                अगर (t->view_src != शून्य && t->बफर == शून्य) {
                    // view of a pre-allocated tensor
                    अगर (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        अवरोध;
                    }
                }
            }
        }
        अगर (t_failed != शून्य) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            क्रम (ggml_backend_बफर_t b : buffers) {
                ggml_backend_बफर_free(b);
            }
            ggml_backend_बफर_free(बफर);
            वापस शून्य;
        }

        buffers.push_back(बफर);
    }

    अगर (buffers.empty()) {
        वापस शून्य;
    }

    अगर (buffers.size() == 1) {
        वापस buffers[0];
    }

    वापस ggml_backend_multi_बफर_alloc_बफर(buffers.तथ्य(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
अटल माप_प्रकार ggml_backend_buft_get_alloc_आकार_n_default(ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    स्थिर ggml_backend_buft_alloc_बफर_n_plan_t plan = ggml_backend_buft_alloc_बफर_n_plan(buft, tensors, n_tensors);

    माप_प्रकार योग = 0;
    क्रम (स्थिर ggml_backend_buft_alloc_बफर_n_plan_item & item : plan) {
        योग += item.size;
    }
    वापस योग;
}

ggml_backend_बफर_t ggml_backend_buft_alloc_बफर_n(ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    GGML_ASSERT(buft);
    अगर (buft->iface.alloc_बफर_n) {
        वापस buft->iface.alloc_बफर_n(buft, tensors, n_tensors);
    }
    वापस ggml_backend_buft_alloc_बफर_n_default(buft, tensors, n_tensors);
}

माप_प्रकार ggml_backend_buft_get_alignment(ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(buft);
    वापस buft->iface.get_alignment(buft);
}

माप_प्रकार ggml_backend_buft_get_max_आकार(ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    अगर (buft->iface.get_max_आकार) {
        वापस buft->iface.get_max_आकार(buft);
    }
    वापस SIZE_MAX;
}

माप_प्रकार ggml_backend_buft_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    अगर (buft->iface.get_alloc_आकार) {
        माप_प्रकार size = buft->iface.get_alloc_आकार(buft, tensor);
        निश्चित(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_आकार_may_expand(tensor->op));

        वापस size;
    }
    वापस ggml_nbytes(tensor);
}

माप_प्रकार ggml_backend_buft_get_alloc_आकार_n(ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors) {
    GGML_ASSERT(buft);
    अगर (buft->iface.get_alloc_आकार_n) {
        वापस buft->iface.get_alloc_आकार_n(buft, tensors, n_tensors);
    }
    वापस ggml_backend_buft_get_alloc_आकार_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(buft);
    अगर (buft->iface.is_host) {
        वापस buft->iface.is_host(buft);
    }
    वापस false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(buft);
    वापस buft->device;
}

// backend buffer

ggml_backend_बफर_t ggml_backend_बफर_init(
               ggml_backend_बफर_type_t buft,
        काष्ठा ggml_backend_बफर_i      iface,
               व्योम *                     context,
               माप_प्रकार                     size) {
    ggml_backend_बफर_t बफर = नया ggml_backend_बफर {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_बफर_USAGE_ANY
    };

    वापस बफर;
}

स्थिर अक्षर * ggml_backend_बफर_name(ggml_backend_बफर_t बफर) {
    वापस ggml_backend_buft_name(ggml_backend_बफर_get_type(बफर));
}

व्योम ggml_backend_बफर_free(ggml_backend_बफर_t बफर) {
    अगर (बफर == शून्य) {
        वापस;
    }

    अगर (बफर->iface.free_बफर != शून्य) {
        बफर->iface.free_बफर(बफर);
    }
    मिटाओ बफर;
}

माप_प्रकार ggml_backend_बफर_get_आकार(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    वापस बफर->size;
}

व्योम * ggml_backend_बफर_get_base(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    // get_base is optional if the buffer is zero-sized
    अगर (!ggml_backend_बफर_is_meta(बफर) && बफर->size == 0) {
        वापस शून्य;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    अगर (!बफर->iface.get_base) {
        वापस शून्य;
    }

    व्योम * base = बफर->iface.get_base(बफर);

    GGML_ASSERT(base != शून्य && "backend buffer base cannot be NULL");

    वापस base;
}

क्रमागत ggml_status ggml_backend_बफर_init_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(बफर);
    // init_tensor is optional
    अगर (बफर->iface.init_tensor) {
        वापस बफर->iface.init_tensor(बफर, tensor);
    }
    वापस GGML_STATUS_SUCCESS;
}

व्योम ggml_backend_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    GGML_ASSERT(बफर);
    // clear is optional if the buffer is zero-sized
    अगर (बफर->size == 0) {
        वापस;
    }

    बफर->iface.सफा(बफर, value);
}

माप_प्रकार ggml_backend_बफर_get_alignment(ggml_backend_बफर_t बफर) {
    वापस ggml_backend_buft_get_alignment(ggml_backend_बफर_get_type(बफर));
}

माप_प्रकार ggml_backend_बफर_get_max_आकार(ggml_backend_बफर_t बफर) {
    वापस ggml_backend_buft_get_max_आकार(ggml_backend_बफर_get_type(बफर));
}

माप_प्रकार ggml_backend_बफर_get_alloc_आकार(ggml_backend_बफर_t बफर, स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_backend_buft_get_alloc_आकार(ggml_backend_बफर_get_type(बफर), tensor);
}

bool ggml_backend_बफर_is_host(ggml_backend_बफर_t बफर) {
    वापस ggml_backend_buft_is_host(ggml_backend_बफर_get_type(बफर));
}

व्योम ggml_backend_बफर_set_usage(ggml_backend_बफर_t बफर, क्रमागत ggml_backend_बफर_usage usage) {
    GGML_ASSERT(बफर);
    बफर->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    अगर (ggml_backend_बफर_is_multi_बफर(बफर)) {
        ggml_backend_multi_बफर_set_usage(बफर, usage);
    } अन्यथा अगर (ggml_backend_बफर_is_meta(बफर)) {
        ggml_backend_meta_बफर_set_usage(बफर, usage);
    }
}

क्रमागत ggml_backend_बफर_usage ggml_backend_बफर_get_usage(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    वापस बफर->usage;
}

ggml_backend_बफर_type_t ggml_backend_बफर_get_type(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    वापस बफर->buft;
}

व्योम ggml_backend_बफर_reset(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    अगर (बफर->iface.reset) {
        बफर->iface.reset(बफर);
    }
}

bool ggml_backend_बफर_copy_tensor(स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst) {
    ggml_backend_बफर_t dst_buf = dst->view_src ? dst->view_src->बफर : dst->बफर;
    अगर (dst_buf->iface.cpy_tensor) {
        वापस dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    वापस false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    अगर (backend == शून्य) {
        वापस शून्य;
    }
    वापस backend->guid;
}

स्थिर अक्षर * ggml_backend_name(ggml_backend_t backend) {
    अगर (backend == शून्य) {
        वापस "NULL";
    }
    वापस backend->iface.get_name(backend);
}

व्योम ggml_backend_free(ggml_backend_t backend) {
    अगर (backend == शून्य) {
        वापस;
    }

    backend->iface.मुक्त(backend);
}

ggml_backend_बफर_type_t ggml_backend_get_default_बफर_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    वापस ggml_backend_dev_बफर_type(backend->device);
}

ggml_backend_बफर_t ggml_backend_alloc_बफर(ggml_backend_t backend, माप_प्रकार size) {
    वापस ggml_backend_buft_alloc_बफर(ggml_backend_get_default_बफर_type(backend), size);
}

माप_प्रकार ggml_backend_get_alignment(ggml_backend_t backend) {
    वापस ggml_backend_buft_get_alignment(ggml_backend_get_default_बफर_type(backend));
}

माप_प्रकार ggml_backend_get_max_आकार(ggml_backend_t backend) {
    वापस ggml_backend_buft_get_max_आकार(ggml_backend_get_default_बफर_type(backend));
}

व्योम ggml_backend_tensor_set_async(ggml_backend_t backend, काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    अगर (backend->iface.set_tensor_async == शून्य) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, तथ्य, offset, size);
    } अन्यथा {
        backend->iface.set_tensor_async(backend, tensor, तथ्य, offset, size);
    }
}

व्योम ggml_backend_tensor_get_async(ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    अगर (backend->iface.get_tensor_async == शून्य) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, तथ्य, offset, size);
    } अन्यथा {
        backend->iface.get_tensor_async(backend, tensor, तथ्य, offset, size);
    }
}

व्योम ggml_backend_tensor_set_2d_async(ggml_backend_t backend, काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size,
            माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");

    अगर (n_copies <= 1 || backend->iface.set_tensor_2d_async == शून्य) {
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (स्थिर अक्षर *) तथ्य + i*stride_तथ्य, offset + i*stride_tensor, size);
        }
        वापस;
    }
    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
}

व्योम ggml_backend_tensor_get_2d_async(ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size,
            माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");

    अगर (n_copies <= 1 || backend->iface.get_tensor_2d_async == शून्य) {
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (अक्षर *) तथ्य + i*stride_तथ्य, offset + i*stride_tensor, size);
        }
        वापस;
    }
    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
}

व्योम ggml_backend_tensor_set(काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(tensor);
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;
    GGML_ASSERT(buf != शून्य && "tensor buffer not set");

    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, तथ्य, offset, size);
}

व्योम ggml_backend_tensor_get(स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(tensor);
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;
    GGML_ASSERT(buf != शून्य && "tensor buffer not set");

    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, तथ्य, offset, size);
}

व्योम ggml_backend_tensor_set_2d(काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size,
            माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    GGML_ASSERT(tensor);
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;
    GGML_ASSERT(buf != शून्य && "tensor buffer not set");

    अगर (n_copies <= 1 || buf->iface.set_tensor_2d == शून्य) {
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (स्थिर अक्षर *) तथ्य + i*stride_तथ्य, offset + i*stride_tensor, size);
        }
        वापस;
    }
    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
}

व्योम ggml_backend_tensor_get_2d(स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size,
            माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    GGML_ASSERT(tensor);
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;
    GGML_ASSERT(buf != शून्य && "tensor buffer not set");

    अगर (n_copies <= 1 || buf->iface.get_tensor_2d == शून्य) {
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (अक्षर *) तथ्य + i*stride_तथ्य, offset + i*stride_tensor, size);
        }
        वापस;
    }
    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
}

व्योम ggml_backend_tensor_memset(काष्ठा ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(tensor);
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    अगर (size == 0) {
        वापस;
    }

    GGML_ASSERT(buf != शून्य && "tensor buffer not set");
    GGML_ASSERT(tensor->तथ्य != शून्य && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != शून्य && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

व्योम ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    अगर (backend->iface.synchronize == शून्य) {
        वापस;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != शून्य);

    वापस backend->iface.graph_plan_create(backend, cgraph);
}

व्योम ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != शून्य);

    backend->iface.graph_plan_free(backend, plan);
}

क्रमागत ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != शून्य);

    वापस backend->iface.graph_plan_compute(backend, plan);
}

क्रमागत ggml_status ggml_backend_graph_compute(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    क्रमागत ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    वापस err;
}

क्रमागत ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    वापस backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * op) {
    GGML_ASSERT(backend);
    वापस ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(backend);
    वापस ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * op) {
    GGML_ASSERT(backend);
    वापस ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    वापस backend->device;
}

// backend copy

व्योम ggml_backend_tensor_copy(स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    अगर (src == dst) {
        वापस;
    }

    अगर (ggml_backend_बफर_is_host(src->बफर)) {
        ggml_backend_tensor_set(dst, src->तथ्य, 0, ggml_nbytes(src));
    } अन्यथा अगर (ggml_backend_बफर_is_host(dst->बफर)) {
        ggml_backend_tensor_get(src, dst->तथ्य, 0, ggml_nbytes(src));
    } अन्यथा अगर (!ggml_backend_बफर_copy_tensor(src, dst)) {
#अगर_अघोषित न_संशोधन
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_बफर_name(src->बफर), ggml_backend_बफर_name(dst->बफर));
#पूर्ण_अगर // NDEBUG
        माप_प्रकार nbytes = ggml_nbytes(src);
        व्योम * तथ्य = दो_स्मृति(nbytes);
        ggml_backend_tensor_get(src, तथ्य, 0, nbytes);
        ggml_backend_tensor_set(dst, तथ्य, 0, nbytes);
        मुक्त(तथ्य);
    }
}

व्योम ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    अगर (src == dst) {
        वापस;
    }

    GGML_ASSERT(backend_dst);
    अगर (backend_dst->iface.cpy_tensor_async != शून्य) {
        अगर (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            वापस;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    अगर (device == शून्य || device->iface.event_new == शून्य) {
        वापस शून्य;
    }
    वापस device->iface.event_new(device);
}

व्योम ggml_backend_event_free(ggml_backend_event_t event) {
    अगर (event == शून्य) {
        वापस;
    }
    event->device->iface.event_free(event->device, event);
}

व्योम ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != शून्य);

    backend->iface.event_record(backend, event);
}

व्योम ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

व्योम ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != शून्य);

    backend->iface.event_wait(backend, event);
}

अटल व्योम ggml_backend_graph_optimize(ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    अगर (backend->iface.graph_optimize != शून्य) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

स्थिर अक्षर * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    वापस device->iface.get_name(device);
}

स्थिर अक्षर * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    वापस device->iface.get_description(device);
}

व्योम ggml_backend_dev_memory(ggml_backend_dev_t device, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, मुक्त, योग);
}

क्रमागत ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    वापस device->iface.get_type(device);
}

व्योम ggml_backend_dev_get_props(ggml_backend_dev_t device, काष्ठा ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    स_रखो(props, 0, माप(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    वापस device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, स्थिर अक्षर * params) {
    GGML_ASSERT(device);
    वापस device->iface.init_backend(device, params);
}

ggml_backend_बफर_type_t ggml_backend_dev_बफर_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    वापस device->iface.get_बफर_type(device);
}

ggml_backend_बफर_type_t ggml_backend_dev_host_बफर_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    अगर (device->iface.get_host_बफर_type == शून्य) {
        वापस शून्य;
    }

    वापस device->iface.get_host_बफर_type(device);
}

ggml_backend_बफर_t ggml_backend_dev_बफर_from_host_ptr(ggml_backend_dev_t device, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    GGML_ASSERT(device);
    वापस device->iface.बफर_from_host_ptr(device, ptr, size, max_tensor_आकार);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, स्थिर काष्ठा ggml_tensor * op) {
    GGML_ASSERT(device);
    वापस device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_बफर_type_t buft) {
    GGML_ASSERT(device);
    वापस device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, स्थिर काष्ठा ggml_tensor * op) {
    GGML_ASSERT(device);
    अगर (device->iface.offload_op != शून्य) {
        वापस device->iface.offload_op(device, op);
    }

    वापस false;
}

// Backend (reg)

स्थिर अक्षर * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    वापस reg->iface.get_name(reg);
}

माप_प्रकार ggml_backend_reg_dev_गिनती(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    वापस reg->iface.get_device_गिनती(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    GGML_ASSERT(reg);
    वापस reg->iface.get_device(reg, सूचकांक);
}

व्योम * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    GGML_ASSERT(reg);
    अगर (!reg->iface.get_proc_address) {
        वापस शून्य;
    }
    वापस reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

काष्ठा ggml_backend_multi_बफर_context {
    ggml_backend_बफर_t * buffers;
    माप_प्रकार n_buffers;
};

अटल व्योम ggml_backend_multi_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    ggml_backend_multi_बफर_context * ctx = (ggml_backend_multi_बफर_context *) बफर->context;
    क्रम (माप_प्रकार i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_बफर_free(ctx->buffers[i]);
    }

    मुक्त(ctx->buffers);
    मुक्त(ctx);
}

अटल व्योम ggml_backend_multi_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    GGML_ASSERT(बफर);
    ggml_backend_multi_बफर_context * ctx = (ggml_backend_multi_बफर_context *) बफर->context;
    क्रम (माप_प्रकार i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_बफर_clear(ctx->buffers[i], value);
    }
}

अटल स्थिर काष्ठा ggml_backend_बफर_i ggml_backend_multi_बफर_i = {
    /* .free_buffer     = */ ggml_backend_multi_बफर_free_बफर,
    /* .get_base        = */ शून्य,
    /* .init_tensor     = */ शून्य,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ शून्य,
    /* .get_tensor      = */ शून्य,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ शून्य,
    /* .clear           = */ ggml_backend_multi_बफर_clear,
    /* .reset           = */ शून्य,
};

ggml_backend_बफर_t ggml_backend_multi_बफर_alloc_बफर(ggml_backend_बफर_t * buffers, माप_प्रकार n_buffers) {
    ggml_backend_multi_बफर_context * ctx = (ggml_backend_multi_बफर_context *) दो_स्मृति(माप(काष्ठा ggml_backend_multi_बफर_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_बफर_t *) दो_स्मृति(n_buffers * माप(ggml_backend_बफर_t));

    GGML_ASSERT(ctx->buffers != शून्य);

    माप_प्रकार योग_आकार = 0;
    क्रम (माप_प्रकार i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        योग_आकार += ggml_backend_बफर_get_आकार(buffers[i]);
    }

    वापस ggml_backend_बफर_init(buffers[0]->buft, ggml_backend_multi_बफर_i, ctx, योग_आकार);
}

bool ggml_backend_बफर_is_multi_बफर(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    वापस बफर->iface.free_बफर == ggml_backend_multi_बफर_free_बफर;
}

व्योम ggml_backend_multi_बफर_set_usage(ggml_backend_बफर_t बफर, क्रमागत ggml_backend_बफर_usage usage) {
    GGML_ASSERT(बफर);
    GGML_ASSERT(ggml_backend_बफर_is_multi_बफर(बफर));
    ggml_backend_multi_बफर_context * ctx = (ggml_backend_multi_बफर_context *) बफर->context;
    क्रम (माप_प्रकार i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_बफर_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
अटल काष्ठा ggml_tensor * ggml_dup_tensor_layout(काष्ठा ggml_context * ctx, स्थिर काष्ठा ggml_tensor * tensor) {
    काष्ठा ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    वापस dup;
}

अटल bool ggml_is_view_op(क्रमागत ggml_op op) {
    वापस op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#अगर_अघोषित GGML_SCHED_MAX_BACKENDS
#घोषणा GGML_SCHED_MAX_BACKENDS 16
#पूर्ण_अगर

#अगर_अघोषित GGML_SCHED_MAX_SPLIT_INPUTS
#घोषणा GGML_SCHED_MAX_SPLIT_INPUTS 30
#पूर्ण_अगर

#अगर_अघोषित GGML_SCHED_MAX_COPIES
#घोषणा GGML_SCHED_MAX_COPIES 4
#पूर्ण_अगर

काष्ठा ggml_backend_sched_split {
    पूर्णांक backend_id;
    पूर्णांक i_start;
    पूर्णांक i_end;
    काष्ठा ggml_tensor ** inputs;
    पूर्णांक n_inputs;
    पूर्णांक inputs_capacity;
    // graph view of this split
    काष्ठा ggml_cgraph graph;
};

काष्ठा ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    पूर्णांक n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_बफर_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    काष्ठा ggml_hash_set  hash_set;
    पूर्णांक                 * hv_tensor_backend_ids; // [hash_set.size]
    काष्ठा ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    पूर्णांक * गाँठ_backend_ids; // [graph_size]
    पूर्णांक * leaf_backend_ids; // [graph_size]

    पूर्णांक * पूर्व_गाँठ_backend_ids; // [graph_size]
    पूर्णांक * पूर्व_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    काष्ठा ggml_cgraph graph;

    // graph splits
    काष्ठा ggml_backend_sched_split * splits;
    पूर्णांक n_splits;
    पूर्णांक splits_capacity;

    // pipeline parallelism support
    पूर्णांक n_copies;
    पूर्णांक cur_copy;
    पूर्णांक अग्र_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    काष्ठा ggml_tensor ** graph_inputs;
    पूर्णांक n_graph_inputs;
    पूर्णांक graph_inputs_capacity;

    काष्ठा ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    व्योम * callback_eval_user_तथ्य;

    ggml_backend_sched_copy_callback callback_copy;
    व्योम * callback_copy_user_तथ्य;

    ggml_backend_sched_realloc_callback callback_realloc;
    व्योम * callback_realloc_user_तथ्य;

    अक्षर * context_बफर;
    माप_प्रकार context_बफर_आकार;

    bool op_offload;

    पूर्णांक debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    पूर्णांक graph_आकार;
    पूर्णांक पूर्व_graph_आकार;

    पूर्णांक debug_realloc;
};

#घोषणा hash_id(tensor) ggml_hash_find_or_जोड़ो(&sched->hash_set, tensor)
#घोषणा tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#घोषणा tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#घोषणा tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

अटल व्योम ggml_backend_sched_split_inputs_grow(काष्ठा ggml_backend_sched_split * split) {
    पूर्णांक new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    अगर (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    स्वतः * pnew = (काष्ठा ggml_tensor **) पुनः_स्मृति((व्योम *) split->inputs, new_cap * माप(काष्ठा ggml_tensor *));
    अगर (pnew == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * माप(काष्ठा ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

अटल व्योम ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    पूर्णांक new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    अगर (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    स्वतः * pnew = (काष्ठा ggml_tensor **) पुनः_स्मृति((व्योम *) sched->graph_inputs, new_cap * माप(काष्ठा ggml_tensor *));
    अगर (pnew == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * माप(काष्ठा ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
अटल पूर्णांक ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    क्रम (पूर्णांक i = 0; i < sched->n_backends; i++) {
        अगर (sched->backends[i] == backend) {
            वापस i;
        }
    }
    वापस -1;
}

अटल पूर्णांक ggml_backend_sched_backend_from_बफर(ggml_backend_sched_t sched, स्थिर काष्ठा ggml_tensor * tensor, स्थिर काष्ठा ggml_tensor * op) {
    ggml_backend_बफर_t बफर = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;
    अगर (बफर == शून्य) {
        वापस -1;
    }

    // find highest prio backend that supports the buffer type and the op
    क्रम (पूर्णांक i = 0; i < sched->n_backends; i++) {
        अगर (ggml_backend_supports_buft(sched->backends[i], बफर->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            वापस i;
        }
    }

#अगर_अघोषित न_संशोधन
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_बफर_name(बफर), tensor->name);
#पूर्ण_अगर

    वापस -1;
}

#अगर 0
#घोषणा GGML_SCHED_MAX_SPLITS_DEBUG 4096
अटल अक्षर causes[GGML_DEFAULT_GRAPH_आकार*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#घोषणा SET_CAUSE(गाँठ, ...) प्र_लिखो(causes[hash_id(गाँठ)], __VA_ARGS__)
#घोषणा GET_CAUSE(गाँठ) causes[hash_id(गाँठ)]
#अन्यथा
#घोषणा SET_CAUSE(गाँठ, ...)
#घोषणा GET_CAUSE(गाँठ) ""
#पूर्ण_अगर

// returns the backend that should be used for the node based on the current locations
अटल पूर्णांक ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, काष्ठा ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    पूर्णांक cur_backend_id = ggml_backend_sched_backend_from_बफर(sched, tensor, tensor);
    अगर (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        वापस cur_backend_id;
    }

    // view_src
    अगर (tensor->view_src != शून्य) {
        cur_backend_id = ggml_backend_sched_backend_from_बफर(sched, tensor->view_src, tensor);
        अगर (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            वापस cur_backend_id;
        }
    }

    अगर (tensor->बफर || (tensor->view_src && tensor->view_src->बफर)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_बफर_t बफर = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_बफर_name(बफर), ggml_op_name(tensor->op));
    }

    // graph input
    अगर (tensor->flags & GGML_TENSOR_FLAG_निवेश) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        वापस cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    अगर (allow) {
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            स्थिर काष्ठा ggml_tensor * src = tensor->src[i];
            अगर (src == शून्य) {
                जारी;
            }
            अगर (src->बफर != शून्य && src->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS) {
                पूर्णांक src_backend_id = ggml_backend_sched_backend_from_बफर(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                अगर (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_बफर_is_host(src->बफर)) {
                    क्रम (पूर्णांक b = 0; b < src_backend_id; b++) {
                        अगर (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            वापस b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                वापस src_backend_id;
            }
        }
    }

    वापस -1;
}

अटल अक्षर * fmt_आकार(माप_प्रकार size) {
    अटल अक्षर बफर[128];
    अगर (size >= 1024*1024) {
        snprintf(बफर, माप(बफर), "%zuM", size/1024/1024);
    } अन्यथा {
        snprintf(बफर, माप(बफर), "%zuK", size/1024);
    }
    वापस बफर;
}

अटल व्योम ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * graph) {
    पूर्णांक cur_split = 0;
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        अगर (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            क्रम (पूर्णांक j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                अगर (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_आकार(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        अगर (ggml_is_view_op(गाँठ->op)) {
            जारी;
        }
        अगर (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, गाँठ);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(गाँठ), गाँठ->name,
                fmt_आकार(ggml_nbytes(गाँठ)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(गाँठ),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, गाँठ)], गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                काष्ठा ggml_tensor * src = गाँठ->src[j];
                अगर (src == शून्य) {
                    जारी;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_आकार(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

अटल bool ggml_backend_sched_बफर_supported(ggml_backend_sched_t sched, काष्ठा ggml_tensor * t, पूर्णांक backend_id) {
    ggml_backend_बफर_t buf = t->view_src ? t->view_src->बफर : t->बफर;
    ggml_backend_बफर_type_t buft = शून्य;

    अगर (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } अन्यथा {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        पूर्णांक tensor_backend_id = tensor_backend_id(t);
        अगर (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        अगर (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    वापस buft != शून्य && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

अटल व्योम ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, काष्ठा ggml_tensor * गाँठ, पूर्णांक cur_backend_id, पूर्णांक * गाँठ_backend_id) {
    अगर (ggml_backend_supports_op(sched->backends[cur_backend_id], गाँठ)) {
        *गाँठ_backend_id = cur_backend_id;
        SET_CAUSE(गाँठ, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
व्योम ggml_backend_sched_split_graph(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    काष्ठा ggml_init_params params = {
        /* .mem_size =   */ sched->context_बफर_आकार,
        /* .mem_buffer = */ sched->context_बफर,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    अगर (sched->ctx == शून्य) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_अग्र_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    क्रम (पूर्णांक i = 0; i < graph->n_leafs; i++) {
        काष्ठा ggml_tensor * leaf = graph->leafs[i];
        पूर्णांक * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        अगर (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        पूर्णांक * गाँठ_backend_id = &tensor_backend_id(गाँठ);
        // do not overwrite user assignments
        अगर (*गाँठ_backend_id == -1) {
            *गाँठ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, गाँठ);

#अगर 0
            // src
            अगर (गाँठ->op == GGML_OP_NONE) {
                जारी;
            }

            क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                काष्ठा ggml_tensor * src = गाँठ->src[j];
                अगर (src == शून्य) {
                    जारी;
                }
                पूर्णांक * src_backend_id = &tensor_backend_id(src);
                अगर (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#पूर्ण_अगर
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        पूर्णांक cur_backend_id = -1;
        क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
            काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
            अगर (ggml_is_view_op(गाँठ->op)) {
                जारी;
            }
            पूर्णांक * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            अगर (*गाँठ_backend_id != -1) {
                अगर (*गाँठ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } अन्यथा {
                    cur_backend_id = *गाँठ_backend_id;
                }
            } अन्यथा अगर (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }
    // expand gpu up
    {
        पूर्णांक cur_backend_id = -1;
        क्रम (पूर्णांक i = graph->n_nodes - 1; i >= 0; i--) {
            काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
            अगर (ggml_is_view_op(गाँठ->op)) {
                जारी;
            }
            पूर्णांक * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            अगर (*गाँठ_backend_id != -1) {
                अगर (*गाँठ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } अन्यथा {
                    cur_backend_id = *गाँठ_backend_id;
                }
            } अन्यथा अगर (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }
    // expand rest down
    {
        पूर्णांक cur_backend_id = -1;
        क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
            काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
            अगर (ggml_is_view_op(गाँठ->op)) {
                जारी;
            }
            पूर्णांक * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            अगर (*गाँठ_backend_id != -1) {
                cur_backend_id = *गाँठ_backend_id;
            } अन्यथा अगर (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }
    // expand rest up
    {
        पूर्णांक cur_backend_id = -1;
        क्रम (पूर्णांक i = graph->n_nodes - 1; i >= 0; i--) {
            काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
            अगर (ggml_is_view_op(गाँठ->op)) {
                जारी;
            }
            पूर्णांक * गाँठ_backend_id = &tensor_backend_id(गाँठ);
            अगर (*गाँठ_backend_id != -1) {
                cur_backend_id = *गाँठ_backend_id;
            } अन्यथा अगर (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, गाँठ, cur_backend_id, गाँठ_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        अगर (ggml_is_view_op(गाँठ->op)) {
            जारी;
        }
        पूर्णांक * गाँठ_backend_id = &tensor_backend_id(गाँठ);
        अगर (*गाँठ_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            पूर्णांक n_supported_best = -1;
            क्रम (पूर्णांक b = 0; b < sched->n_backends; b++) {
                अगर (ggml_backend_supports_op(sched->backends[b], गाँठ)) {
                    पूर्णांक n_supported = 0;
                    क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                        काष्ठा ggml_tensor * src = गाँठ->src[j];
                        अगर (src == शून्य) {
                            जारी;
                        }
                        अगर ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_बफर_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    अगर (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *गाँठ_backend_id = b;
                        SET_CAUSE(गाँठ, "3.best");
                    }
                }
            }
        } अन्यथा {
            // assigned node: upgrade to higher prio backend if possible
            क्रम (पूर्णांक b = 0; b < *गाँठ_backend_id; b++) {
                अगर (sched->bufts[b] == sched->bufts[*गाँठ_backend_id] && ggml_backend_supports_op(sched->backends[b], गाँठ)) {
                    bool supported = true;
                    क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                        काष्ठा ggml_tensor * src = गाँठ->src[j];
                        अगर (src == शून्य) {
                            जारी;
                        }
                        अगर (!ggml_backend_sched_बफर_supported(sched, src, b)) {
                            supported = false;
                            अवरोध;
                        }
                    }
                    अगर (supported) {
                        *गाँठ_backend_id = b;
                        SET_CAUSE(गाँठ, "3.upg");
                        अवरोध;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        पूर्णांक * cur_backend_id = &tensor_backend_id(गाँठ);
        अगर (गाँठ->view_src != शून्य && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(गाँठ->view_src);
            SET_CAUSE(गाँठ, "4.vsrc");
        }
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            काष्ठा ggml_tensor * src = गाँठ->src[j];
            अगर (src == शून्य) {
                जारी;
            }
            पूर्णांक * src_backend_id = &tensor_backend_id(src);
            अगर (*src_backend_id == -1) {
                अगर (src->view_src != शून्य) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } अन्यथा {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        क्रम (पूर्णांक b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, गाँठ, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        पूर्णांक i_split = 0;
        काष्ठा ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        पूर्णांक i = 0;
        क्रम (; i < graph->n_nodes; i++) {
            काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
            अगर (!ggml_is_view_op(गाँठ->op)) {
                split->backend_id = tensor_backend_id(गाँठ);
                अवरोध;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        पूर्णांक cur_backend_id = split->backend_id;
        क्रम (; i < graph->n_nodes; i++) {
            काष्ठा ggml_tensor * गाँठ = graph->nodes[i];

            अगर (ggml_is_view_op(गाँठ->op)) {
                जारी;
            }

            स्थिर पूर्णांक गाँठ_backend_id = tensor_backend_id(गाँठ);

            GGML_ASSERT(गाँठ_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            अगर (गाँठ_backend_id == cur_backend_id && split->n_inputs > 0) {
                क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                    काष्ठा ggml_tensor * src = गाँठ->src[j];
                    अगर (src == शून्य) {
                        जारी;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    अगर (src->बफर != शून्य && src->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS) {
                        पूर्णांक src_backend_id = tensor_backend_id(src);
                        अगर (src_backend_id != cur_backend_id && !ggml_backend_sched_बफर_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            अवरोध;
                        }
                    }
                }
            }

            अगर (गाँठ_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                अगर (i_split >= sched->splits_capacity) {
                    पूर्णांक old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        पुनः_स्मृति(sched->splits, sched->splits_capacity * माप(काष्ठा ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != शून्य);
                    क्रम (पूर्णांक k = old_cap; k < sched->splits_capacity; k++) {
                        स_रखो(&sched->splits[k], 0, माप(काष्ठा ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = गाँठ_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = गाँठ_backend_id;
            }

            // find inputs that are not on the same backend
            क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
                काष्ठा ggml_tensor * src = गाँठ->src[j];
                अगर (src == शून्य) {
                    जारी;
                }

                माप_प्रकार src_id = hash_id(src);
                स्थिर पूर्णांक src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                अगर (src_backend_id != cur_backend_id && !ggml_backend_sched_बफर_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    अगर (tensor_id_copy(src_id, cur_backend_id, 0) == शून्य) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        क्रम (पूर्णांक c = 0; c < sched->n_copies; c++) {
                            काष्ठा ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            अगर (sched->n_copies > 1) {
                                ggml_set_निवेश(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        पूर्णांक n_inputs = split->n_inputs++;
                        अगर (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    गाँठ->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    अगर (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    अगर (sched->n_copies > 1) {
        क्रम (पूर्णांक i = 0; i < graph->n_leafs; i++) {
            काष्ठा ggml_tensor * leaf = graph->leafs[i];
            अगर ((leaf->flags & GGML_TENSOR_FLAG_निवेश) == 0) {
                जारी;
            }

            स्थिर माप_प्रकार leaf_id = hash_id(leaf);
            स्थिर पूर्णांक leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            अगर (tensor_id_copy(leaf_id, leaf_backend_id, 0) == शून्य) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                क्रम (पूर्णांक c = 0; c < sched->n_copies; c++) {
                    काष्ठा ggml_tensor * tensor_copy;
                    अगर (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } अन्यथा {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_निवेश(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            पूर्णांक n_graph_inputs = sched->n_graph_inputs++;
            अगर (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        पूर्णांक * tmp = sched->गाँठ_backend_ids;
        sched->गाँठ_backend_ids = sched->पूर्व_गाँठ_backend_ids;
        sched->पूर्व_गाँठ_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->पूर्व_leaf_backend_ids;
        sched->पूर्व_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    काष्ठा ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](व्योम * user_तथ्य, ggml_tensor * tensor, ggml_tensor * until) {
            स्वतः & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_तथ्य;
            std::vector<ggml_tensor *> & keep = deps[until];
            अगर (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    क्रम (पूर्णांक i = 0; i < sched->n_splits; i++) {
        काष्ठा ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    पूर्णांक n_dep_nodes = 0;
    क्रम (स्थिर स्वतः & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    पूर्णांक योग_inputs = sched->n_graph_inputs;
    क्रम (पूर्णांक i = 0; i < sched->n_splits; i++) {
        योग_inputs += sched->splits[i].n_inputs;
    }
    पूर्णांक graph_आकार = std::max(graph->n_nodes, graph->n_leafs) + योग_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->पूर्व_graph_आकार = sched->graph_आकार;
    sched->graph_आकार = graph_आकार;

    अगर (sched->graph.size < graph_आकार) {
        sched->graph.size = graph_आकार;
        sched->graph.nodes = (ggml_tensor **) पुनः_स्मृति(sched->graph.nodes, graph_आकार * माप(काष्ठा ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) पुनः_स्मृति(sched->graph.leafs, graph_आकार * माप(काष्ठा ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != शून्य);
        GGML_ASSERT(sched->graph.leafs != शून्य);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    काष्ठा ggml_cgraph * graph_copy = &sched->graph;

    पूर्णांक n_dep_nodes_added = 0;

    क्रम (पूर्णांक i = 0; i < sched->n_splits; i++) {
        काष्ठा ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        क्रम (पूर्णांक j = 0; j < split->n_inputs; j++) {
            निश्चित(graph_copy->size > (graph_copy->n_nodes + 1));

            काष्ठा ggml_tensor * निवेश = split->inputs[j];
            स्थिर माप_प्रकार निवेश_id = hash_id(निवेश);
            काष्ठा ggml_tensor * निवेश_cpy = tensor_id_copy(निवेश_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            काष्ठा ggml_tensor * निवेश_dep = ggml_view_tensor(sched->ctx, निवेश);
            निवेश_dep->src[0] = निवेश;
            sched->गाँठ_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[निवेश_id];
            graph_copy->nodes[graph_copy->n_nodes++] = निवेश_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->गाँठ_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = निवेश_cpy;
        }

        क्रम (पूर्णांक j = split->i_start; j < split->i_end; j++) {
            निश्चित(graph_copy->size > graph_copy->n_nodes);
            sched->गाँठ_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            अगर (alloc_deps.empty()) {
                जारी;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            स्वतः it = alloc_deps.find(graph->nodes[j]);
            अगर (it != alloc_deps.end()) {
                स्थिर std::vector<ggml_tensor *> & keep = it->second;
                क्रम (माप_प्रकार k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    काष्ठा ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    क्रम (माप_प्रकार s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    निश्चित(graph_copy->size > graph_copy->n_nodes);
                    sched->गाँठ_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    अगर (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        क्रम (पूर्णांक i = 0; i < sched->n_graph_inputs; i++) {
            काष्ठा ggml_tensor * निवेश = sched->graph_inputs[i];
            माप_प्रकार id = hash_id(निवेश);
            पूर्णांक backend_id = tensor_backend_id(निवेश);
            क्रम (पूर्णांक c = 0; c < sched->n_copies; c++) {
                काष्ठा ggml_tensor * निवेश_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                निश्चित(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = निवेश_cpy;
            }
        }

        क्रम (पूर्णांक i = 0; i < sched->n_splits; i++) {
            काष्ठा ggml_backend_sched_split * split = &sched->splits[i];
            पूर्णांक backend_id = split->backend_id;
            क्रम (पूर्णांक j = 0; j < split->n_inputs; j++) {
                काष्ठा ggml_tensor * निवेश = split->inputs[j];
                माप_प्रकार id = hash_id(निवेश);
                क्रम (पूर्णांक c = 0; c < sched->n_copies; c++) {
                    काष्ठा ggml_tensor * निवेश_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    निश्चित(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = निवेश_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    क्रम (पूर्णांक i = 0; i < graph->n_leafs; i++) {
        काष्ठा ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        निश्चित(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    क्रम (पूर्णांक i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_अग्र_uid();
    }
}

अटल bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    क्रम (पूर्णांक i = 0; i < sched->graph.n_nodes; i++) {
        अगर (sched->गाँठ_backend_ids[i] != sched->पूर्व_गाँठ_backend_ids[i] &&
            sched->bufts[sched->गाँठ_backend_ids[i]] != sched->bufts[sched->पूर्व_गाँठ_backend_ids[i]]) {
            backend_ids_changed = true;
            अवरोध;
        }
    }
    अगर (!backend_ids_changed) {
        क्रम (पूर्णांक i = 0; i < sched->graph.n_leafs; i++) {
            अगर (sched->leaf_backend_ids[i] != sched->पूर्व_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->पूर्व_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                अवरोध;
            }
        }
    }

    // allocate graph
    अगर (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#अगर_अघोषित न_संशोधन
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#पूर्ण_अगर

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        स्थिर bool graph_आकार_changed = sched->पूर्व_graph_आकार != sched->graph_आकार;

        अगर (sched->callback_realloc != शून्य) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_आकार_changed, sched->callback_realloc_user_तथ्य);
        }

        अगर (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            स्थिर bool unexpected = !backend_ids_changed && !graph_आकार_changed;

            अगर (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_आकार, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        क्रम (पूर्णांक i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        अगर (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->गाँठ_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            वापस false;
        }
        अगर (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            वापस false;
        }
    }

    वापस true;
}

अटल bool ggml_backend_sched_is_host_weight(स्थिर काष्ठा ggml_tensor * t) {
    वापस t->बफर != शून्य &&
        ggml_backend_बफर_get_usage(t->बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS &&
        ggml_backend_बफर_is_host(t->बफर);
}

अटल व्योम ggml_backend_sched_copy_निवेश(ggml_backend_sched_t sched, काष्ठा ggml_backend_sched_split * split, काष्ठा ggml_tensor * निवेश) {
    स्थिर पूर्णांक split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t निवेश_backend = ggml_backend_sched_get_tensor_backend(sched, निवेश);
    काष्ठा ggml_tensor * निवेश_cpy = tensor_copy(निवेश, split_backend_id, sched->cur_copy);

    अगर (निवेश->flags & GGML_TENSOR_FLAG_निवेश) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        अगर (sched->events[split_backend_id][sched->cur_copy] != शून्य) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } अन्यथा {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(निवेश, निवेश_cpy);
        वापस;
    }

    // wait for the split backend to finish using the input before overwriting it
    अगर (sched->events[split_backend_id][sched->cur_copy] != शून्य) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } अन्यथा {
        ggml_backend_synchronize(split_backend);
    }

    अगर (sched->callback_copy != शून्य && ggml_backend_sched_is_host_weight(निवेश) &&
        sched->callback_copy(split_backend, निवेश, निवेश_cpy, &split->graph, sched->callback_copy_user_तथ्य)) {
        वापस;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    अगर (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(निवेश_backend, split_backend, निवेश, निवेश_cpy)) {
        ggml_backend_synchronize(निवेश_backend);
        अगर (sched->events[split_backend_id][sched->cur_copy] != शून्य) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } अन्यथा {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(निवेश, निवेश_cpy);
    }
}

अटल क्रमागत ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    काष्ठा ggml_backend_sched_split * splits = sched->splits;

    पूर्णांक पूर्व_backend_id = -1;

    क्रम (पूर्णांक split_id = 0; split_id < sched->n_splits; split_id++) {
        काष्ठा ggml_backend_sched_split * split = &splits[split_id];
        पूर्णांक split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        अगर (split->n_inputs == 0 && पूर्व_backend_id >= 0 && पूर्व_backend_id != split_backend_id) {
            अगर (sched->events[पूर्व_backend_id][sched->cur_copy] != शून्य) {
                ggml_backend_event_synchronize(sched->events[पूर्व_backend_id][sched->cur_copy]);
            } अन्यथा {
                ggml_backend_synchronize(sched->backends[पूर्व_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        क्रम (पूर्णांक निवेश_id = 0; निवेश_id < split->n_inputs; निवेश_id++) {
            अगर (!ggml_backend_sched_is_host_weight(split->inputs[निवेश_id])) {
                ggml_backend_sched_copy_निवेश(sched, split, split->inputs[निवेश_id]);
            }
        }
        क्रम (पूर्णांक निवेश_id = 0; निवेश_id < split->n_inputs; निवेश_id++) {
            अगर (ggml_backend_sched_is_host_weight(split->inputs[निवेश_id])) {
                ggml_backend_sched_copy_निवेश(sched, split, split->inputs[निवेश_id]);
            }
        }

        अगर (!sched->callback_eval) {
            क्रमागत ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            अगर (ec != GGML_STATUS_SUCCESS) {
                वापस ec;
            }
        } अन्यथा {
            // similar to ggml_backend_compare_graph_backend
            क्रम (पूर्णांक j0 = 0; j0 < split->graph.n_nodes; j0++) {
                काष्ठा ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_तथ्य);

                पूर्णांक j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                जबतक (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_तथ्य);
                }

                काष्ठा ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                क्रमागत ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                अगर (ec != GGML_STATUS_SUCCESS) {
                    वापस ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                अगर (need && !sched->callback_eval(t, false, sched->callback_eval_user_तथ्य)) {
                    अवरोध;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        अगर (sched->events[split_backend_id][sched->cur_copy] != शून्य) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        पूर्व_backend_id = split_backend_id;
    }

    वापस GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_बफर_type_t * bufts,
        पूर्णांक n_backends,
        माप_प्रकार graph_आकार,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    काष्ठा ggml_backend_sched * sched = (ggml_backend_sched *) सुस्मृति(1, माप(काष्ठा ggml_backend_sched));

    स्थिर अक्षर * GGML_SCHED_DEBUG = दो_पर्या("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? म_से_प(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#अगर_घोषित GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#पूर्ण_अगर
    स्थिर अक्षर * GGML_SCHED_DEBUG_REALLOC = दो_पर्या("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? म_से_प(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_आकार);
    sched->hv_tensor_backend_ids = (पूर्णांक *) दो_स्मृति(sched->hash_set.size * माप(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) दो_स्मृति(sched->hash_set.size * sched->n_backends * sched->n_copies * माप(काष्ठा ggml_tensor *));

    स्थिर माप_प्रकार ggml_sched_max_splits = graph_आकार; // at most there is one split for each node in the graph
    स्थिर माप_प्रकार nodes_आकार = graph_आकार + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->गाँठ_backend_ids = (पूर्णांक *) सुस्मृति(nodes_आकार, माप(sched->गाँठ_backend_ids[0]));
    sched->leaf_backend_ids = (पूर्णांक *) सुस्मृति(nodes_आकार, माप(sched->leaf_backend_ids[0]));
    sched->पूर्व_गाँठ_backend_ids = (पूर्णांक *) सुस्मृति(nodes_आकार, माप(sched->पूर्व_गाँठ_backend_ids[0]));
    sched->पूर्व_leaf_backend_ids = (पूर्णांक *) सुस्मृति(nodes_आकार, माप(sched->पूर्व_leaf_backend_ids[0]));

    sched->graph_आकार      = 0;
    sched->पूर्व_graph_आकार = 0;

    sched->context_बफर_आकार = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*माप(काष्ठा ggml_tensor) + ggml_graph_overhead_custom(graph_आकार, false);
    sched->context_बफर = (अक्षर *) दो_स्मृति(sched->context_बफर_आकार);

    स्थिर पूर्णांक initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) सुस्मृति(initial_splits_capacity, माप(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (काष्ठा ggml_tensor **) सुस्मृति(sched->graph_inputs_capacity, माप(काष्ठा ggml_tensor *));

    क्रम (पूर्णांक b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_बफर_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        अगर (sched->n_copies > 1) {
            क्रम (पूर्णांक c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    वापस sched;
}

व्योम ggml_backend_sched_free(ggml_backend_sched_t sched) {
    अगर (sched == शून्य) {
        वापस;
    }
    क्रम (पूर्णांक b = 0; b < sched->n_backends; b++) {
        क्रम (पूर्णांक c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    क्रम (पूर्णांक i = 0; i < sched->splits_capacity; i++) {
        मुक्त(sched->splits[i].inputs);
    }
    मुक्त(sched->splits);
    मुक्त(sched->graph_inputs);
    मुक्त(sched->hv_tensor_backend_ids);
    मुक्त(sched->hv_tensor_copies);
    मुक्त(sched->गाँठ_backend_ids);
    मुक्त(sched->leaf_backend_ids);
    मुक्त(sched->पूर्व_गाँठ_backend_ids);
    मुक्त(sched->पूर्व_leaf_backend_ids);
    मुक्त(sched->context_बफर);
    मुक्त(sched->graph.nodes);
    मुक्त(sched->graph.leafs);
    मुक्त(sched);
}

व्योम ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    अगर (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        स_रखो(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * माप(sched->hv_tensor_backend_ids[0]));
        स_रखो(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * माप(काष्ठा ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

व्योम ggml_backend_sched_reserve_आकार(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * measure_graph, माप_प्रकार * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((पूर्णांक)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_आकार(sched->galloc, &sched->graph, sched->गाँठ_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((पूर्णांक)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    अगर (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->गाँठ_backend_ids, sched->leaf_backend_ids)) {
        वापस false;
    }

    ggml_backend_sched_reset(sched);

    वापस true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((पूर्णांक)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->अग्र_copy;
    sched->अग्र_copy = (sched->अग्र_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    अगर (!ggml_backend_sched_alloc_splits(sched)) {
        वापस false;
    }

    sched->is_alloc = true;

    वापस true;
}

क्रमागत ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * graph) {
    क्रमागत ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    वापस err;
}

क्रमागत ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, काष्ठा ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    अगर (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    अगर (!sched->is_alloc) {
        अगर (!ggml_backend_sched_alloc_graph(sched, graph)) {
            वापस GGML_STATUS_ALLOC_FAILED;
        }
    }

    वापस ggml_backend_sched_compute_splits(sched);
}

व्योम ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    क्रम (पूर्णांक i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    अगर (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->अग्र_copy = 0;
    }
}

व्योम ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, व्योम * user_तथ्य) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_तथ्य = user_तथ्य;
}

व्योम ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, व्योम * user_तथ्य) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_तथ्य = user_तथ्य;
}

व्योम ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, व्योम * user_तथ्य) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_तथ्य = user_तथ्य;
}

पूर्णांक ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    वापस sched->n_splits;
}

पूर्णांक ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    वापस sched->n_copies;
}

पूर्णांक ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    वापस sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, पूर्णांक i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    वापस sched->backends[i];
}

ggml_backend_बफर_type_t ggml_backend_sched_get_बफर_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    पूर्णांक backend_सूचकांक = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_सूचकांक >= 0 && backend_सूचकांक < sched->n_backends);

    वापस sched->bufts[backend_सूचकांक];
}

माप_प्रकार ggml_backend_sched_get_बफर_आकार(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    पूर्णांक backend_सूचकांक = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_सूचकांक >= 0 && backend_सूचकांक < sched->n_backends);

    वापस ggml_gallocr_get_बफर_आकार(sched->galloc, backend_सूचकांक);
}

व्योम ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, काष्ठा ggml_tensor * गाँठ, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    पूर्णांक backend_सूचकांक = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_सूचकांक >= 0 && backend_सूचकांक < sched->n_backends);
    tensor_backend_id(गाँठ) = backend_सूचकांक;
    SET_CAUSE(गाँठ, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, काष्ठा ggml_tensor * गाँठ) {
    GGML_ASSERT(sched);
    पूर्णांक backend_सूचकांक = tensor_backend_id(गाँठ);
    अगर (backend_सूचकांक == -1) {
        वापस शून्य;
    }
    वापस sched->backends[backend_सूचकांक];
}

// utils

bool ggml_op_alloc_आकार_may_expand(क्रमागत ggml_op op) {
    चयन (op) {
        हाल GGML_OP_FLASH_ATTN_EXT:
        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
        हाल GGML_OP_CUMSUM:
        हाल GGML_OP_ARGSORT:
        हाल GGML_OP_शीर्ष_K:
        हाल GGML_OP_ARGMAX:
            वापस true;
        शेष:
            वापस false;
    }
}

क्रमागत ggml_status ggml_backend_view_init(काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->बफर == शून्य);
    GGML_ASSERT(tensor->view_src != शून्य);
    GGML_ASSERT(tensor->view_src->बफर != शून्य);
    GGML_ASSERT(tensor->view_src->तथ्य != शून्य);

    tensor->बफर = tensor->view_src->बफर;
    tensor->तथ्य = (अक्षर *)tensor->view_src->तथ्य + tensor->view_offs;
    वापस ggml_backend_बफर_init_tensor(tensor->बफर, tensor);
}

क्रमागत ggml_status ggml_backend_tensor_alloc(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor, व्योम * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->बफर == शून्य);
    GGML_ASSERT(tensor->तथ्य == शून्य);
    GGML_ASSERT(tensor->view_src == शून्य);
    GGML_ASSERT(addr >= ggml_backend_बफर_get_base(बफर));
    GGML_ASSERT(ggml_backend_बफर_is_meta(बफर) ||
        (अक्षर *) addr + ggml_backend_बफर_get_alloc_आकार(बफर, tensor) <=
        (अक्षर *) ggml_backend_बफर_get_base(बफर) + ggml_backend_बफर_get_आकार(बफर));

    tensor->बफर = बफर;
    tensor->तथ्य = addr;
    वापस ggml_backend_बफर_init_tensor(बफर, tensor);
}

अटल काष्ठा ggml_tensor * graph_copy_dup_tensor(काष्ठा ggml_hash_set hash_set, काष्ठा ggml_tensor ** गाँठ_copies,
    काष्ठा ggml_context * ctx_allocated, काष्ठा ggml_context * ctx_unallocated, काष्ठा ggml_tensor * src) {

    GGML_ASSERT(src != शून्य);
    GGML_ASSERT(src->तथ्य && "graph must be allocated");

    माप_प्रकार id = ggml_hash_जोड़ो(&hash_set, src);
    अगर (id == GGML_HASHSET_ALREADY_EXISTS) {
        वापस गाँठ_copies[ggml_hash_find(&hash_set, src)];
    }

    काष्ठा ggml_tensor * dst = ggml_dup_tensor_layout(src->तथ्य && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    अगर (src->view_src != शून्य) {
        dst->view_src = graph_copy_dup_tensor(hash_set, गाँठ_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    स_नकल(dst->op_params, src->op_params, माप(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        काष्ठा ggml_tensor * s = src->src[i];
        अगर (s == शून्य) {
            जारी;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, गाँठ_copies, ctx_allocated, ctx_unallocated, s);
    }

    गाँठ_copies[id] = dst;
    वापस dst;
}

अटल व्योम graph_copy_init_tensor(काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor ** गाँठ_copies, bool * गाँठ_init, काष्ठा ggml_tensor * src) {
    माप_प्रकार id = ggml_hash_find(hash_set, src);
    अगर (गाँठ_init[id]) {
        वापस;
    }
    गाँठ_init[id] = true;

    काष्ठा ggml_tensor * dst = गाँठ_copies[id];
    अगर (dst->view_src != शून्य) {
        graph_copy_init_tensor(hash_set, गाँठ_copies, गाँठ_init, src->view_src);
        क्रमागत ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    अन्यथा {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        काष्ठा ggml_tensor * s = src->src[i];
        अगर (s == शून्य) {
            जारी;
        }
        graph_copy_init_tensor(hash_set, गाँठ_copies, गाँठ_init, s);
    }
}

काष्ठा ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, काष्ठा ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    काष्ठा ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    काष्ठा ggml_tensor ** गाँठ_copies = (ggml_tensor **) सुस्मृति(hash_set.size, माप(गाँठ_copies[0])); // NOLINT
    bool * गाँठ_init = (bool *) सुस्मृति(hash_set.size, माप(गाँठ_init[0]));

    काष्ठा ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ शून्य,
        /* .no_alloc   = */ true
    };

    काष्ठा ggml_context * ctx_allocated = ggml_init(params);
    काष्ठा ggml_context * ctx_unallocated = ggml_init(params);

    अगर (ctx_allocated == शून्य || ctx_unallocated == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        मुक्त(गाँठ_copies);
        मुक्त(गाँठ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        वापस {
            /* .buffer           = */ शून्य,
            /* .ctx_allocated    = */ शून्य,
            /* .ctx_unallocated  = */ शून्य,
            /* .graph            = */ शून्य,
        };
    }

    // dup nodes
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, गाँठ_copies, ctx_allocated, ctx_unallocated, गाँठ);
    }

    // allocate nodes
    ggml_backend_बफर_t बफर = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    अगर (बफर == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        मुक्त(गाँठ_copies);
        मुक्त(गाँठ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        वापस {
            /* .buffer           = */ शून्य,
            /* .ctx_allocated    = */ शून्य,
            /* .ctx_unallocated  = */ शून्य,
            /* .graph            = */ शून्य,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, गाँठ_copies, गाँठ_init, गाँठ);
    }

    // build graph copy
    काष्ठा ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    क्रम (पूर्णांक i = 0; i < graph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = graph->nodes[i];
        काष्ठा ggml_tensor * गाँठ_copy = गाँठ_copies[ggml_hash_find(&hash_set, गाँठ)];
        graph_copy->nodes[i] = गाँठ_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    मुक्त(गाँठ_copies);
    मुक्त(गाँठ_init);

    वापस {
        /* .buffer           = */ बफर,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

व्योम ggml_backend_graph_copy_free(काष्ठा ggml_backend_graph_copy copy) {
    ggml_backend_बफर_free(copy.बफर);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, काष्ठा ggml_cgraph * graph, ggml_backend_eval_callback callback, व्योम * user_तथ्य, काष्ठा ggml_tensor स्थिर * स्थिर * test_nodes, माप_प्रकार अंक_test_nodes) {
    काष्ठा ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    अगर (copy.बफर == शून्य) {
        वापस false;
    }

    काष्ठा ggml_cgraph * g1 = graph;
    काष्ठा ggml_cgraph * g2 = copy.graph;

    निश्चित(g1->n_nodes == g2->n_nodes);

    अगर (अंक_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        क्रम (पूर्णांक i = 0; i < g1->n_nodes; i++) {
            क्रम (माप_प्रकार j = 0; j < अंक_test_nodes; ++j) {
                अगर (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_तथ्य);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } अन्यथा {
        क्रम (पूर्णांक i = 0; i < g1->n_nodes; i++) {
            काष्ठा ggml_tensor * t1 = g1->nodes[i];
            काष्ठा ggml_tensor * t2 = g2->nodes[i];

            निश्चित(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            काष्ठा ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            काष्ठा ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            अगर (ggml_is_view_op(t1->op)) {
                जारी;
            }

            // compare results, calculate rms etc
            अगर (!callback(i, t1, t2, user_तथ्य)) {
                अवरोध;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    वापस true;
}

// CPU backend - buffer

अटल व्योम * ggml_backend_cpu_बफर_get_base(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    uintptr_t तथ्य = (uintptr_t)बफर->context;

    // align the buffer
    अगर (तथ्य % TENSOR_ALIGNMENT != 0) {
        तथ्य = GGML_PAD(तथ्य, TENSOR_ALIGNMENT);
    }

    वापस (व्योम *)तथ्य;
}

अटल व्योम ggml_backend_cpu_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    GGML_ASSERT(बफर);
    ggml_aligned_free(बफर->context, बफर->size);
}

अटल व्योम ggml_backend_cpu_बफर_memset_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(tensor);
    स_रखो((अक्षर *)tensor->तथ्य + offset, value, size);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_cpu_बफर_set_tensor(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(tensor);
    स_नकल((अक्षर *)tensor->तथ्य + offset, तथ्य, size);

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_cpu_बफर_get_tensor(ggml_backend_बफर_t बफर, स्थिर काष्ठा ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    GGML_ASSERT(tensor);
    स_नकल(तथ्य, (स्थिर अक्षर *)tensor->तथ्य + offset, size);

    GGML_UNUSED(बफर);
}

अटल bool ggml_backend_cpu_बफर_cpy_tensor(ggml_backend_बफर_t बफर, स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst) {
    GGML_ASSERT(src);
    अगर (ggml_backend_बफर_is_host(src->बफर)) {
        स_नकल(dst->तथ्य, src->तथ्य, ggml_nbytes(src));
        वापस true;
    }
    वापस false;

    GGML_UNUSED(बफर);
}

अटल व्योम ggml_backend_cpu_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    GGML_ASSERT(बफर);
    स_रखो(बफर->context, value, बफर->size);
}

अटल स्थिर काष्ठा ggml_backend_बफर_i ggml_backend_cpu_बफर_i = {
    /* .free_buffer     = */ ggml_backend_cpu_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_cpu_बफर_get_base,
    /* .init_tensor     = */ शून्य, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_cpu_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_बफर_clear,
    /* .reset           = */ शून्य,
};

अटल स्थिर काष्ठा ggml_backend_बफर_i ggml_backend_cpu_बफर_from_ptr_i = {
    /* .free_buffer     = */ शून्य, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_बफर_get_base,
    /* .init_tensor     = */ शून्य, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_cpu_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_बफर_clear,
    /* .reset           = */ शून्य,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

अटल स्थिर अक्षर * ggml_backend_cpu_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस "CPU";

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_t ggml_backend_cpu_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    व्योम * तथ्य = ggml_aligned_malloc(size);

    अगर (तथ्य == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        वापस शून्य;
    }

    वापस ggml_backend_बफर_init(buft, ggml_backend_cpu_बफर_i, तथ्य, size);
}

अटल माप_प्रकार ggml_backend_cpu_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

अटल bool ggml_backend_cpu_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    वापस true;

    GGML_UNUSED(buft);
}

ggml_backend_बफर_type_t ggml_backend_cpu_बफर_type(व्योम) {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_cpu_बफर_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_बफर_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_बफर_type_alloc_बफर,
            /* .alloc_buffer_n      = */ शून्य,
            /* .get_alignment       = */ ggml_backend_cpu_बफर_type_get_alignment,
            /* .get_max_size        = */ शून्य, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ शून्य, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ शून्य,
            /* .is_host             = */ ggml_backend_cpu_बफर_type_is_host,
        },
        /* .device  = */ शून्य, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ शून्य,
    };

    वापस &ggml_backend_cpu_बफर_type;
}

अटल स्थिर अक्षर * ggml_backend_cpu_बफर_from_ptr_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस "CPU_Mapped";

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_type_t ggml_backend_cpu_बफर_from_ptr_type(व्योम) {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_cpu_बफर_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_बफर_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_बफर_type_alloc_बफर,
            /* .alloc_buffer_n      = */ शून्य,
            /* .get_alignment       = */ ggml_backend_cpu_बफर_type_get_alignment,
            /* .get_max_size        = */ शून्य, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ शून्य, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ शून्य,
            /* .is_host             = */ ggml_backend_cpu_बफर_type_is_host,
        },
        /* .device  = */ शून्य, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ शून्य,
    };

    वापस &ggml_backend_cpu_बफर_type;
}

ggml_backend_बफर_t ggml_backend_cpu_बफर_from_ptr(व्योम * ptr, माप_प्रकार size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    वापस ggml_backend_बफर_init(ggml_backend_cpu_बफर_from_ptr_type(), ggml_backend_cpu_बफर_from_ptr_i, ptr, size);
}
