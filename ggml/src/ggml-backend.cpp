// Note: porting this file to C++ is a work in progress

#جے_تعریف_شدہ _WIN32
#تعریف_کرو WIN32_LEAN_AND_اوسط
#جے_بنا_تعریف NOMINMAX
#   define NOMINMAX
#جے_انت
#ضمن <windows.h>
#جے_انت

#ضمن "ggml-backend.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-alloc.h"
#ضمن "ggml-impl.h"

#ضمن <assert.h>
#ضمن <limits.h>
#ضمن <stdarg.h>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <string.h>
#ضمن <algorithm>
#ضمن <unordered_map>
#ضمن <vector>

#جے_تعریف_شدہ __APPLE__
#ضمن <sys/types.h>
#ضمن <sys/sysctl.h>
#جے_انت


// backend buffer type

ثابت حرف * ggml_backend_buft_name(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_name(buft);
}

ggml_backend_بفر_t ggml_backend_buft_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    إذا (size == 0) {
        // return a dummy buffer for zero-sized allocations
        أرجع ggml_backend_بفر_init(buft, {}, عدم, 0);
    }
    أرجع buft->iface.alloc_بفر(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
بنية ggml_backend_buft_alloc_بفر_n_plan_شے {
    size_t size;  // total bytes for this buffer
    صحيح    پہلا; // first tensor index (inclusive)
    صحيح    آخری;  // last tensor index (exclusive)
};

ورتوں ggml_backend_buft_alloc_بفر_n_plan_t = std::vector<ggml_backend_buft_alloc_بفر_n_plan_شے>;

ساكن ggml_backend_buft_alloc_بفر_n_plan_t ggml_backend_buft_alloc_بفر_n_plan(
        ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ggml_backend_buft_alloc_بفر_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = ggml_backend_buft_get_max_ناپ(buft);

    size_t cur_buf_ناپ = 0;
    صحيح    پہلا        = 0;

    لكل (صحيح i = 0; i < n_tensors; i++) {
        size_t this_ناپ = 0;
        بنية ggml_tensor * t = tensors[i];
        إذا (t->انکڑے == عدم && t->view_src == عدم) {
            this_ناپ = GGML_PAD(ggml_backend_buft_get_alloc_ناپ(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        إذا (cur_buf_ناپ > 0 && (cur_buf_ناپ + this_ناپ) > max_size) {
            plan.push_back({ cur_buf_ناپ, پہلا, i });
            cur_buf_ناپ = this_ناپ;
            پہلا        = i;
        } وإلا {
            cur_buf_ناپ += this_ناپ;
        }
    }

    إذا (cur_buf_ناپ > 0) {
        plan.push_back({ cur_buf_ناپ, پہلا, n_tensors });
    }

    أرجع plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ساكن ggml_backend_بفر_t ggml_backend_buft_alloc_بفر_n_default(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_بفر_n_plan_t plan = ggml_backend_buft_alloc_بفر_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_بفر_t> buffers;
    buffers.reserve(plan.size());

    لكل (ثابت ggml_backend_buft_alloc_بفر_n_plan_شے & شے : plan) {
        ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, شے.size);
        إذا (بفر == عدم) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), شے.size);
            لكل (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            أرجع عدم;
        }

        بنية ggml_tallocr tallocr = ggml_tallocr_new(بفر);

        // allocate tensors in the current buffer
        بنية ggml_tensor * t_failed = عدم;
        لكل (صحيح j = شے.پہلا; j < شے.آخری; j++) {
            بنية ggml_tensor * t = tensors[j];
            إذا (t->انکڑے == عدم) {
                إذا (t->view_src == عدم) {
                    إذا (ggml_tallocr_alloc(&tallocr, t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                } وإلا إذا (t->بفر == عدم) {
                    إذا (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            } وإلا {
                إذا (t->view_src != عدم && t->بفر == عدم) {
                    // view of a pre-allocated tensor
                    إذا (ggml_backend_view_init(t) != GGML_حالت_SUCCESS) {
                        t_failed = t;
                        اكسر;
                    }
                }
            }
        }
        إذا (t_failed != عدم) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            لكل (ggml_backend_بفر_t b : buffers) {
                ggml_backend_بفر_free(b);
            }
            ggml_backend_بفر_free(بفر);
            أرجع عدم;
        }

        buffers.push_back(بفر);
    }

    إذا (buffers.خالی_تھاواں()) {
        أرجع عدم;
    }

    إذا (buffers.size() == 1) {
        أرجع buffers[0];
    }

    أرجع ggml_backend_multi_بفر_alloc_بفر(buffers.انکڑے(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ساكن size_t ggml_backend_buft_get_alloc_ناپ_n_default(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    ثابت ggml_backend_buft_alloc_بفر_n_plan_t plan = ggml_backend_buft_alloc_بفر_n_plan(buft, tensors, n_tensors);

    size_t کل = 0;
    لكل (ثابت ggml_backend_buft_alloc_بفر_n_plan_شے & شے : plan) {
        کل += شے.size;
    }
    أرجع کل;
}

ggml_backend_بفر_t ggml_backend_buft_alloc_بفر_n(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    إذا (buft->iface.alloc_بفر_n) {
        أرجع buft->iface.alloc_بفر_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_alloc_بفر_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_ناپ(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    إذا (buft->iface.get_max_ناپ) {
        أرجع buft->iface.get_max_ناپ(buft);
    }
    أرجع SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_ناپ(ggml_backend_بفر_type_t buft, ثابت بنية ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    إذا (buft->iface.get_alloc_ناپ) {
        size_t size = buft->iface.get_alloc_ناپ(buft, tensor);
        أكد(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_خالی_تھاواں(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_ناپ_may_expand(tensor->op));

        أرجع size;
    }
    أرجع ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_ناپ_n(ggml_backend_بفر_type_t buft, بنية ggml_tensor ** tensors, صحيح n_tensors) {
    GGML_ASSERT(buft);
    إذا (buft->iface.get_alloc_ناپ_n) {
        أرجع buft->iface.get_alloc_ناپ_n(buft, tensors, n_tensors);
    }
    أرجع ggml_backend_buft_get_alloc_ناپ_n_default(buft, tensors, n_tensors);
}

منطقي ggml_backend_buft_is_host(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    إذا (buft->iface.is_host) {
        أرجع buft->iface.is_host(buft);
    }
    أرجع خطأ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(buft);
    أرجع buft->device;
}

// backend buffer

ggml_backend_بفر_t ggml_backend_بفر_init(
               ggml_backend_بفر_type_t buft,
        بنية ggml_backend_بفر_i      iface,
               فراغ *                     context,
               size_t                     size) {
    ggml_backend_بفر_t بفر = جديد ggml_backend_بفر {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_بفر_USAGE_ANY
    };

    أرجع بفر;
}

ثابت حرف * ggml_backend_بفر_name(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_name(ggml_backend_بفر_get_type(بفر));
}

فراغ ggml_backend_بفر_free(ggml_backend_بفر_t بفر) {
    إذا (بفر == عدم) {
        أرجع;
    }

    إذا (بفر->iface.free_بفر != عدم) {
        بفر->iface.free_بفر(بفر);
    }
    مٹاؤ بفر;
}

size_t ggml_backend_بفر_get_ناپ(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->size;
}

فراغ * ggml_backend_بفر_get_ادھار(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    // get_base is optional if the buffer is zero-sized
    إذا (!ggml_backend_بفر_is_meta(بفر) && بفر->size == 0) {
        أرجع عدم;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    إذا (!بفر->iface.get_ادھار) {
        أرجع عدم;
    }

    فراغ * ادھار = بفر->iface.get_ادھار(بفر);

    GGML_ASSERT(ادھار != عدم && "backend buffer base cannot be NULL");

    أرجع ادھار;
}

تعداد ggml_حالت ggml_backend_بفر_init_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor) {
    GGML_ASSERT(بفر);
    // init_tensor is optional
    إذا (بفر->iface.init_tensor) {
        أرجع بفر->iface.init_tensor(بفر, tensor);
    }
    أرجع GGML_حالت_SUCCESS;
}

فراغ ggml_backend_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    // clear is optional if the buffer is zero-sized
    إذا (بفر->size == 0) {
        أرجع;
    }

    بفر->iface.clear(بفر, value);
}

size_t ggml_backend_بفر_get_alignment(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_بفر_get_type(بفر));
}

size_t ggml_backend_بفر_get_max_ناپ(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_get_max_ناپ(ggml_backend_بفر_get_type(بفر));
}

size_t ggml_backend_بفر_get_alloc_ناپ(ggml_backend_بفر_t بفر, ثابت بنية ggml_tensor * tensor) {
    أرجع ggml_backend_buft_get_alloc_ناپ(ggml_backend_بفر_get_type(بفر), tensor);
}

منطقي ggml_backend_بفر_is_host(ggml_backend_بفر_t بفر) {
    أرجع ggml_backend_buft_is_host(ggml_backend_بفر_get_type(بفر));
}

فراغ ggml_backend_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, تعداد ggml_backend_بفر_usage usage) {
    GGML_ASSERT(بفر);
    بفر->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    إذا (ggml_backend_بفر_is_multi_بفر(بفر)) {
        ggml_backend_multi_بفر_سیٹ_usage(بفر, usage);
    } وإلا إذا (ggml_backend_بفر_is_meta(بفر)) {
        ggml_backend_meta_بفر_سیٹ_usage(بفر, usage);
    }
}

تعداد ggml_backend_بفر_usage ggml_backend_بفر_get_usage(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->usage;
}

ggml_backend_بفر_type_t ggml_backend_بفر_get_type(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->buft;
}

فراغ ggml_backend_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    إذا (بفر->iface.reset) {
        بفر->iface.reset(بفر);
    }
}

منطقي ggml_backend_بفر_copy_tensor(ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    ggml_backend_بفر_t dst_buf = dst->view_src ? dst->view_src->بفر : dst->بفر;
    إذا (dst_buf->iface.cpy_tensor) {
        أرجع dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    أرجع خطأ;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    إذا (backend == عدم) {
        أرجع عدم;
    }
    أرجع backend->guid;
}

ثابت حرف * ggml_backend_name(ggml_backend_t backend) {
    إذا (backend == عدم) {
        أرجع "NULL";
    }
    أرجع backend->iface.get_name(backend);
}

فراغ ggml_backend_free(ggml_backend_t backend) {
    إذا (backend == عدم) {
        أرجع;
    }

    backend->iface.حرر(backend);
}

ggml_backend_بفر_type_t ggml_backend_get_default_بفر_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_بفر_type(backend->device);
}

ggml_backend_بفر_t ggml_backend_alloc_بفر(ggml_backend_t backend, size_t size) {
    أرجع ggml_backend_buft_alloc_بفر(ggml_backend_get_default_بفر_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_alignment(ggml_backend_get_default_بفر_type(backend));
}

size_t ggml_backend_get_max_ناپ(ggml_backend_t backend) {
    أرجع ggml_backend_buft_get_max_ناپ(ggml_backend_get_default_بفر_type(backend));
}

فراغ ggml_backend_tensor_سیٹ_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    إذا (backend->iface.سیٹ_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_سیٹ(tensor, انکڑے, وتھ, size);
    } وإلا {
        backend->iface.سیٹ_tensor_async(backend, tensor, انکڑے, وتھ, size);
    }
}

فراغ ggml_backend_tensor_get_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    إذا (backend->iface.get_tensor_async == عدم) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, انکڑے, وتھ, size);
    } وإلا {
        backend->iface.get_tensor_async(backend, tensor, انکڑے, وتھ, size);
    }
}

فراغ ggml_backend_tensor_سیٹ_2d_async(ggml_backend_t backend, بنية ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");

    إذا (n_copies <= 1 || backend->iface.سیٹ_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_سیٹ_async(backend, tensor, (ثابت حرف *) انکڑے + i*stride_انکڑے, وتھ + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.سیٹ_tensor_2d_async(backend, tensor, انکڑے, وتھ, size, n_copies, stride_tensor, stride_انکڑے);
}

فراغ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ثابت بنية ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");

    إذا (n_copies <= 1 || backend->iface.get_tensor_2d_async == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (حرف *) انکڑے + i*stride_انکڑے, وتھ + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, انکڑے, وتھ, size, n_copies, stride_tensor, stride_انکڑے);
}

فراغ ggml_backend_tensor_سیٹ(بنية ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.سیٹ_tensor(buf, tensor, انکڑے, وتھ, size);
}

فراغ ggml_backend_tensor_get(ثابت بنية ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, انکڑے, وتھ, size);
}

فراغ ggml_backend_tensor_سیٹ_2d(بنية ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (n_copies <= 1 || buf->iface.سیٹ_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_سیٹ(tensor, (ثابت حرف *) انکڑے + i*stride_انکڑے, وتھ + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.سیٹ_tensor_2d(buf, tensor, انکڑے, وتھ, size, n_copies, stride_tensor, stride_انکڑے);
}

فراغ ggml_backend_tensor_get_2d(ثابت بنية ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_انکڑے) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    GGML_ASSERT(buf != عدم && "tensor buffer not set");

    إذا (n_copies <= 1 || buf->iface.get_tensor_2d == عدم) {
        لكل (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (حرف *) انکڑے + i*stride_انکڑے, وتھ + i*stride_tensor, size);
        }
        أرجع;
    }
    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, انکڑے, وتھ, size, n_copies, stride_tensor, stride_انکڑے);
}

فراغ ggml_backend_tensor_memset(بنية ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    إذا (size == 0) {
        أرجع;
    }

    GGML_ASSERT(buf != عدم && "tensor buffer not set");
    GGML_ASSERT(tensor->انکڑے != عدم && "tensor not allocated");
    GGML_ASSERT(وتھ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != عدم && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, وتھ, size);
}

فراغ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    إذا (backend->iface.synchronize == عدم) {
        أرجع;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_گراف_plan_t ggml_backend_گراف_plan_create(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_create != عدم);

    أرجع backend->iface.گراف_plan_create(backend, cgraph);
}

فراغ ggml_backend_گراف_plan_free(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_free != عدم);

    backend->iface.گراف_plan_free(backend, plan);
}

تعداد ggml_حالت ggml_backend_گراف_plan_compute(ggml_backend_t backend, ggml_backend_گراف_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.گراف_plan_compute != عدم);

    أرجع backend->iface.گراف_plan_compute(backend, plan);
}

تعداد ggml_حالت ggml_backend_گراف_compute(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    تعداد ggml_حالت err = ggml_backend_گراف_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    أرجع err;
}

تعداد ggml_حالت ggml_backend_گراف_compute_async(ggml_backend_t backend, بنية ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    أرجع backend->iface.گراف_compute(backend, cgraph);
}

منطقي ggml_backend_supports_op(ggml_backend_t backend, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_supports_op(backend->device, op);
}

منطقي ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_supports_buft(backend->device, buft);
}

منطقي ggml_backend_offload_op(ggml_backend_t backend, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(backend);
    أرجع ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    أرجع backend->device;
}

// backend copy

فراغ ggml_backend_tensor_copy(ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    إذا (src == dst) {
        أرجع;
    }

    إذا (ggml_backend_بفر_is_host(src->بفر)) {
        ggml_backend_tensor_سیٹ(dst, src->انکڑے, 0, ggml_nbytes(src));
    } وإلا إذا (ggml_backend_بفر_is_host(dst->بفر)) {
        ggml_backend_tensor_get(src, dst->انکڑے, 0, ggml_nbytes(src));
    } وإلا إذا (!ggml_backend_بفر_copy_tensor(src, dst)) {
#جے_بنا_تعریف NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_بفر_name(src->بفر), ggml_backend_بفر_name(dst->بفر));
#جے_انت // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        فراغ * انکڑے = حافظہ_دیو(nbytes);
        ggml_backend_tensor_get(src, انکڑے, 0, nbytes);
        ggml_backend_tensor_سیٹ(dst, انکڑے, 0, nbytes);
        حرر(انکڑے);
    }
}

فراغ ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    إذا (src == dst) {
        أرجع;
    }

    GGML_ASSERT(backend_dst);
    إذا (backend_dst->iface.cpy_tensor_async != عدم) {
        إذا (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            أرجع;
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
    إذا (device == عدم || device->iface.event_new == عدم) {
        أرجع عدم;
    }
    أرجع device->iface.event_new(device);
}

فراغ ggml_backend_event_free(ggml_backend_event_t event) {
    إذا (event == عدم) {
        أرجع;
    }
    event->device->iface.event_free(event->device, event);
}

فراغ ggml_backend_event_ریکارڈ(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_ریکارڈ != عدم);

    backend->iface.event_ریکارڈ(backend, event);
}

فراغ ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

فراغ ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != عدم);

    backend->iface.event_wait(backend, event);
}

ساكن فراغ ggml_backend_گراف_optimize(ggml_backend_t backend, بنية ggml_cgraph * cgraph, بنية ggml_backend_گراف_optimize_params * params) {
    GGML_ASSERT(backend);
    إذا (backend->iface.گراف_optimize != عدم) {
        backend->iface.گراف_optimize(backend, cgraph, params);
    }
}

// Backend device

ثابت حرف * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_name(device);
}

ثابت حرف * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_description(device);
}

فراغ ggml_backend_dev_حافظہ(ggml_backend_dev_t device, size_t * حرر, size_t * کل) {
    GGML_ASSERT(device);
    device->iface.get_حافظہ(device, حرر, کل);
}

تعداد ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_type(device);
}

فراغ ggml_backend_dev_get_props(ggml_backend_dev_t device, بنية ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, حجم(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ثابت حرف * params) {
    GGML_ASSERT(device);
    أرجع device->iface.init_backend(device, params);
}

ggml_backend_بفر_type_t ggml_backend_dev_بفر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    أرجع device->iface.get_بفر_type(device);
}

ggml_backend_بفر_type_t ggml_backend_dev_host_بفر_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    إذا (device->iface.get_host_بفر_type == عدم) {
        أرجع عدم;
    }

    أرجع device->iface.get_host_بفر_type(device);
}

ggml_backend_بفر_t ggml_backend_dev_بفر_from_host_ptr(ggml_backend_dev_t device, فراغ * ptr, size_t size, size_t max_tensor_ناپ) {
    GGML_ASSERT(device);
    أرجع device->iface.بفر_from_host_ptr(device, ptr, size, max_tensor_ناپ);
}

منطقي ggml_backend_dev_supports_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_op(device, op);
}

منطقي ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_بفر_type_t buft) {
    GGML_ASSERT(device);
    أرجع device->iface.supports_buft(device, buft);
}

منطقي ggml_backend_dev_offload_op(ggml_backend_dev_t device, ثابت بنية ggml_tensor * op) {
    GGML_ASSERT(device);
    إذا (device->iface.offload_op != عدم) {
        أرجع device->iface.offload_op(device, op);
    }

    أرجع خطأ;
}

// Backend (reg)

ثابت حرف * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_گنتی(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device_گنتی(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t اشاریہ) {
    GGML_ASSERT(reg);
    أرجع reg->iface.get_device(reg, اشاریہ);
}

فراغ * ggml_backend_reg_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ASSERT(reg);
    إذا (!reg->iface.get_proc_پتہ) {
        أرجع عدم;
    }
    أرجع reg->iface.get_proc_پتہ(reg, name);
}

// multi-buffer buffer

بنية ggml_backend_multi_بفر_context {
    ggml_backend_بفر_t * buffers;
    size_t n_buffers;
};

ساكن فراغ ggml_backend_multi_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_free(ctx->buffers[i]);
    }

    حرر(ctx->buffers);
    حرر(ctx);
}

ساكن فراغ ggml_backend_multi_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_clear(ctx->buffers[i], value);
    }
}

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_multi_بفر_i = {
    /* .free_buffer     = */ ggml_backend_multi_بفر_free_بفر,
    /* .get_base        = */ عدم,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ عدم,
    /* .get_tensor      = */ عدم,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,
    /* .clear           = */ ggml_backend_multi_بفر_clear,
    /* .reset           = */ عدم,
};

ggml_backend_بفر_t ggml_backend_multi_بفر_alloc_بفر(ggml_backend_بفر_t * buffers, size_t n_buffers) {
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) حافظہ_دیو(حجم(بنية ggml_backend_multi_بفر_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_بفر_t *) حافظہ_دیو(n_buffers * حجم(ggml_backend_بفر_t));

    GGML_ASSERT(ctx->buffers != عدم);

    size_t کل_ناپ = 0;
    لكل (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        کل_ناپ += ggml_backend_بفر_get_ناپ(buffers[i]);
    }

    أرجع ggml_backend_بفر_init(buffers[0]->buft, ggml_backend_multi_بفر_i, ctx, کل_ناپ);
}

منطقي ggml_backend_بفر_is_multi_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    أرجع بفر->iface.free_بفر == ggml_backend_multi_بفر_free_بفر;
}

فراغ ggml_backend_multi_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, تعداد ggml_backend_بفر_usage usage) {
    GGML_ASSERT(بفر);
    GGML_ASSERT(ggml_backend_بفر_is_multi_بفر(بفر));
    ggml_backend_multi_بفر_context * ctx = (ggml_backend_multi_بفر_context *) بفر->context;
    لكل (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_بفر_سیٹ_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ساكن بنية ggml_tensor * ggml_dup_tensor_layout(بنية ggml_context * ctx, ثابت بنية ggml_tensor * tensor) {
    بنية ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    أرجع dup;
}

ساكن منطقي ggml_is_view_op(تعداد ggml_op op) {
    أرجع op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_ترتیب_بدلو || op == GGML_OP_TRANSPOSE;
}

// scheduler

#جے_بنا_تعریف GGML_SCHED_MAX_BACKENDS
#تعریف_کرو GGML_SCHED_MAX_BACKENDS 16
#جے_انت

#جے_بنا_تعریف GGML_SCHED_MAX_ونڈو_INPUTS
#تعریف_کرو GGML_SCHED_MAX_ونڈو_INPUTS 30
#جے_انت

#جے_بنا_تعریف GGML_SCHED_MAX_COPIES
#تعریف_کرو GGML_SCHED_MAX_COPIES 4
#جے_انت

بنية ggml_backend_sched_ونڈو {
    صحيح backend_id;
    صحيح i_شروع;
    صحيح i_سرا;
    بنية ggml_tensor ** inputs;
    صحيح n_inputs;
    صحيح inputs_سمرتھا;
    // graph view of this split
    بنية ggml_cgraph graph;
};

بنية ggml_backend_sched {
    منطقي is_reset; // true if the scheduler has been reset since the last graph split
    منطقي is_alloc;

    صحيح n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_بفر_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    بنية ggml_hash_سیٹ  hash_سیٹ;
    صحيح                 * hv_tensor_backend_ids; // [hash_set.size]
    بنية ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    صحيح * گرہ_backend_ids; // [graph_size]
    صحيح * پتی_backend_ids; // [graph_size]

    صحيح * prev_گرہ_backend_ids; // [graph_size]
    صحيح * prev_پتی_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    بنية ggml_cgraph graph;

    // graph splits
    بنية ggml_backend_sched_ونڈو * splits;
    صحيح n_splits;
    صحيح splits_سمرتھا;

    // pipeline parallelism support
    صحيح n_copies;
    صحيح cur_copy;
    صحيح اگلا_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    بنية ggml_tensor ** گراف_inputs;
    صحيح n_گراف_inputs;
    صحيح گراف_inputs_سمرتھا;

    بنية ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    فراغ * callback_eval_user_انکڑے;

    ggml_backend_sched_copy_callback callback_copy;
    فراغ * callback_copy_user_انکڑے;

    ggml_backend_sched_realloc_callback callback_realloc;
    فراغ * callback_realloc_user_انکڑے;

    حرف * context_بفر;
    size_t context_بفر_ناپ;

    منطقي op_offload;

    صحيح debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    صحيح گراف_ناپ;
    صحيح prev_گراف_ناپ;

    صحيح debug_realloc;
};

#تعریف_کرو hash_id(tensor) ggml_hash_لبھو_or_داخل_کرو(&sched->hash_سیٹ, tensor)
#تعریف_کرو tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#تعریف_کرو tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#تعریف_کرو tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ساكن فراغ ggml_backend_sched_ونڈو_inputs_grow(بنية ggml_backend_sched_ونڈو * ونڈو) {
    صحيح new_cap = GGML_SCHED_MAX_ونڈو_INPUTS;
    إذا (ونڈو->inputs_سمرتھا > 0) {
        new_cap = 2*ونڈو->inputs_سمرتھا;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, ونڈو->inputs_سمرتھا, new_cap);
    }
    خودکار * pnew = (بنية ggml_tensor **) realloc((فراغ *) ونڈو->inputs, new_cap * حجم(بنية ggml_tensor *));
    إذا (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    ونڈو->inputs = pnew;
    ونڈو->inputs_سمرتھا = new_cap;
}

ساكن فراغ ggml_backend_sched_گراف_inputs_grow(ggml_backend_sched_t sched) {
    صحيح new_cap = GGML_SCHED_MAX_ونڈو_INPUTS;
    إذا (sched->گراف_inputs_سمرتھا > 0) {
        new_cap = 2*sched->گراف_inputs_سمرتھا;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->گراف_inputs_سمرتھا, new_cap);
    }
    خودکار * pnew = (بنية ggml_tensor **) realloc((فراغ *) sched->گراف_inputs, new_cap * حجم(بنية ggml_tensor *));
    إذا (pnew == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * حجم(بنية ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->گراف_inputs = pnew;
    sched->گراف_inputs_سمرتھا = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ساكن صحيح ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        إذا (sched->backends[i] == backend) {
            أرجع i;
        }
    }
    أرجع -1;
}

ساكن صحيح ggml_backend_sched_backend_from_بفر(ggml_backend_sched_t sched, ثابت بنية ggml_tensor * tensor, ثابت بنية ggml_tensor * op) {
    ggml_backend_بفر_t بفر = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
    إذا (بفر == عدم) {
        أرجع -1;
    }

    // find highest prio backend that supports the buffer type and the op
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        إذا (ggml_backend_supports_buft(sched->backends[i], بفر->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            أرجع i;
        }
    }

#جے_بنا_تعریف NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_بفر_name(بفر), tensor->name);
#جے_انت

    أرجع -1;
}

#إذا 0
#تعریف_کرو GGML_SCHED_MAX_SPLITS_DEBUG 4096
ساكن حرف causes[GGML_DEFAULT_گراف_ناپ*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_ونڈو_INPUTS][128]; // debug only
#تعریف_کرو سیٹ_CAUSE(گرہ, ...) sprintf(causes[hash_id(گرہ)], __VA_ARGS__)
#تعریف_کرو GET_CAUSE(گرہ) causes[hash_id(گرہ)]
#وإلا
#تعریف_کرو سیٹ_CAUSE(گرہ, ...)
#تعریف_کرو GET_CAUSE(گرہ) ""
#جے_انت

// returns the backend that should be used for the node based on the current locations
ساكن صحيح ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, بنية ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    صحيح cur_backend_id = ggml_backend_sched_backend_from_بفر(sched, tensor, tensor);
    إذا (cur_backend_id != -1) {
        سیٹ_CAUSE(tensor, "1.dst");
        أرجع cur_backend_id;
    }

    // view_src
    إذا (tensor->view_src != عدم) {
        cur_backend_id = ggml_backend_sched_backend_from_بفر(sched, tensor->view_src, tensor);
        إذا (cur_backend_id != -1) {
            سیٹ_CAUSE(tensor, "1.vsrc");
            أرجع cur_backend_id;
        }
    }

    إذا (tensor->بفر || (tensor->view_src && tensor->view_src->بفر)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_بفر_t بفر = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_بفر_name(بفر), ggml_op_name(tensor->op));
    }

    // graph input
    إذا (tensor->flags & GGML_TENSOR_FLAG_داخلہ) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        سیٹ_CAUSE(tensor, "1.inp");
        أرجع cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    منطقي allow = سچ;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    إذا (allow) {
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            ثابت بنية ggml_tensor * src = tensor->src[i];
            إذا (src == عدم) {
                استمر;
            }
            إذا (src->بفر != عدم && src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                صحيح src_backend_id = ggml_backend_sched_backend_from_بفر(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                إذا (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_بفر_is_host(src->بفر)) {
                    لكل (صحيح b = 0; b < src_backend_id; b++) {
                        إذا (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            سیٹ_CAUSE(tensor, "1.off");
                            أرجع b;
                        }
                    }
                }
                سیٹ_CAUSE(tensor, "1.wgt%d", i);
                أرجع src_backend_id;
            }
        }
    }

    أرجع -1;
}

ساكن حرف * fmt_ناپ(size_t size) {
    ساكن حرف بفر[128];
    إذا (size >= 1024*1024) {
        snprintf(بفر, حجم(بفر), "%zuM", size/1024/1024);
    } وإلا {
        snprintf(بفر, حجم(بفر), "%zuK", size/1024);
    }
    أرجع بفر;
}

ساكن فراغ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    صحيح cur_ونڈو = 0;
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        إذا (cur_ونڈو < sched->n_splits && i == sched->splits[cur_ونڈو].i_شروع) {
            ggml_backend_t ونڈو_backend = sched->backends[sched->splits[cur_ونڈو].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_ونڈو, ggml_backend_name(ونڈو_backend),
                sched->splits[cur_ونڈو].n_inputs);
            لكل (صحيح j = 0; j < sched->splits[cur_ونڈو].n_inputs; j++) {
                إذا (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_ونڈو].inputs[j]->name,
                    fmt_ناپ(ggml_nbytes(sched->splits[cur_ونڈو].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_ونڈو++;
        }
        بنية ggml_tensor * گرہ = graph->nodes[i];
        إذا (ggml_is_view_op(گرہ->op)) {
            استمر;
        }
        إذا (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, گرہ);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(گرہ), گرہ->name,
                fmt_ناپ(ggml_nbytes(گرہ)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(گرہ),
                graph->use_counts[ggml_hash_لبھو(&graph->ویکھیا_hash_سیٹ, گرہ)], گرہ->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = گرہ->src[j];
                إذا (src == عدم) {
                    استمر;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_ناپ(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ساكن منطقي ggml_backend_sched_بفر_supported(ggml_backend_sched_t sched, بنية ggml_tensor * t, صحيح backend_id) {
    ggml_backend_بفر_t buf = t->view_src ? t->view_src->بفر : t->بفر;
    ggml_backend_بفر_type_t buft = عدم;

    إذا (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } وإلا {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        صحيح tensor_backend_id = tensor_backend_id(t);
        إذا (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        إذا (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    أرجع buft != عدم && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ساكن فراغ ggml_backend_sched_سیٹ_if_supported(ggml_backend_sched_t sched, بنية ggml_tensor * گرہ, صحيح cur_backend_id, صحيح * گرہ_backend_id) {
    إذا (ggml_backend_supports_op(sched->backends[cur_backend_id], گرہ)) {
        *گرہ_backend_id = cur_backend_id;
        سیٹ_CAUSE(گرہ, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
فراغ ggml_backend_sched_ونڈو_گراف(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_گراف_inputs = 0;
    sched->is_reset = خطأ;

    بنية ggml_init_params params = {
        /* .mem_size =   */ sched->context_بفر_ناپ,
        /* .mem_buffer = */ sched->context_بفر,
        /* .no_alloc =   */ سچ
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    إذا (sched->ctx == عدم) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_گراف_اگلا_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * پتی = graph->leafs[i];
        صحيح * پتی_backend_id = &tensor_backend_id(پتی);
        // do not overwrite user assignments
        إذا (*پتی_backend_id == -1) {
            *پتی_backend_id = ggml_backend_sched_backend_id_from_cur(sched, پتی);
        }
    }

    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = graph->nodes[i];
        صحيح * گرہ_backend_id = &tensor_backend_id(گرہ);
        // do not overwrite user assignments
        إذا (*گرہ_backend_id == -1) {
            *گرہ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, گرہ);

#إذا 0
            // src
            إذا (گرہ->op == GGML_OP_NONE) {
                استمر;
            }

            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = گرہ->src[j];
                إذا (src == عدم) {
                    استمر;
                }
                صحيح * src_backend_id = &tensor_backend_id(src);
                إذا (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#جے_انت
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گرہ = graph->nodes[i];
            إذا (ggml_is_view_op(گرہ->op)) {
                استمر;
            }
            صحيح * گرہ_backend_id = &tensor_backend_id(گرہ);
            إذا (*گرہ_backend_id != -1) {
                إذا (*گرہ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *گرہ_backend_id;
                }
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
            }
        }
    }
    // expand gpu up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * گرہ = graph->nodes[i];
            إذا (ggml_is_view_op(گرہ->op)) {
                استمر;
            }
            صحيح * گرہ_backend_id = &tensor_backend_id(گرہ);
            إذا (*گرہ_backend_id != -1) {
                إذا (*گرہ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } وإلا {
                    cur_backend_id = *گرہ_backend_id;
                }
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
            }
        }
    }
    // expand rest down
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = 0; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گرہ = graph->nodes[i];
            إذا (ggml_is_view_op(گرہ->op)) {
                استمر;
            }
            صحيح * گرہ_backend_id = &tensor_backend_id(گرہ);
            إذا (*گرہ_backend_id != -1) {
                cur_backend_id = *گرہ_backend_id;
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
            }
        }
    }
    // expand rest up
    {
        صحيح cur_backend_id = -1;
        لكل (صحيح i = graph->n_nodes - 1; i >= 0; i--) {
            بنية ggml_tensor * گرہ = graph->nodes[i];
            إذا (ggml_is_view_op(گرہ->op)) {
                استمر;
            }
            صحيح * گرہ_backend_id = &tensor_backend_id(گرہ);
            إذا (*گرہ_backend_id != -1) {
                cur_backend_id = *گرہ_backend_id;
            } وإلا إذا (cur_backend_id != -1) {
                ggml_backend_sched_سیٹ_if_supported(sched, گرہ, cur_backend_id, گرہ_backend_id);
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
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = graph->nodes[i];
        إذا (ggml_is_view_op(گرہ->op)) {
            استمر;
        }
        صحيح * گرہ_backend_id = &tensor_backend_id(گرہ);
        إذا (*گرہ_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            صحيح n_supported_بہترین = -1;
            لكل (صحيح b = 0; b < sched->n_backends; b++) {
                إذا (ggml_backend_supports_op(sched->backends[b], گرہ)) {
                    صحيح n_supported = 0;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = گرہ->src[j];
                        إذا (src == عدم) {
                            استمر;
                        }
                        إذا ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_بفر_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    إذا (n_supported > n_supported_بہترین) {
                        n_supported_بہترین = n_supported;
                        *گرہ_backend_id = b;
                        سیٹ_CAUSE(گرہ, "3.best");
                    }
                }
            }
        } وإلا {
            // assigned node: upgrade to higher prio backend if possible
            لكل (صحيح b = 0; b < *گرہ_backend_id; b++) {
                إذا (sched->bufts[b] == sched->bufts[*گرہ_backend_id] && ggml_backend_supports_op(sched->backends[b], گرہ)) {
                    منطقي supported = سچ;
                    لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                        بنية ggml_tensor * src = گرہ->src[j];
                        إذا (src == عدم) {
                            استمر;
                        }
                        إذا (!ggml_backend_sched_بفر_supported(sched, src, b)) {
                            supported = خطأ;
                            اكسر;
                        }
                    }
                    إذا (supported) {
                        *گرہ_backend_id = b;
                        سیٹ_CAUSE(گرہ, "3.upg");
                        اكسر;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = graph->nodes[i];
        صحيح * cur_backend_id = &tensor_backend_id(گرہ);
        إذا (گرہ->view_src != عدم && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(گرہ->view_src);
            سیٹ_CAUSE(گرہ, "4.vsrc");
        }
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            بنية ggml_tensor * src = گرہ->src[j];
            إذا (src == عدم) {
                استمر;
            }
            صحيح * src_backend_id = &tensor_backend_id(src);
            إذا (*src_backend_id == -1) {
                إذا (src->view_src != عدم) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    سیٹ_CAUSE(src, "4.vsrc");
                } وإلا {
                    *src_backend_id = *cur_backend_id;
                    سیٹ_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        لكل (صحيح b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_سیٹ_if_supported(sched, گرہ, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        صحيح i_ونڈو = 0;
        بنية ggml_backend_sched_ونڈو * ونڈو = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        صحيح i = 0;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گرہ = graph->nodes[i];
            إذا (!ggml_is_view_op(گرہ->op)) {
                ونڈو->backend_id = tensor_backend_id(گرہ);
                اكسر;
            }
        }
        ونڈو->i_شروع = 0;
        ونڈو->n_inputs = 0;
        صحيح cur_backend_id = ونڈو->backend_id;
        لكل (; i < graph->n_nodes; i++) {
            بنية ggml_tensor * گرہ = graph->nodes[i];

            إذا (ggml_is_view_op(گرہ->op)) {
                استمر;
            }

            ثابت صحيح گرہ_backend_id = tensor_backend_id(گرہ);

            GGML_ASSERT(گرہ_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            منطقي need_new_ونڈو = خطأ;
            إذا (گرہ_backend_id == cur_backend_id && ونڈو->n_inputs > 0) {
                لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                    بنية ggml_tensor * src = گرہ->src[j];
                    إذا (src == عدم) {
                        استمر;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    إذا (src->بفر != عدم && src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                        صحيح src_backend_id = tensor_backend_id(src);
                        إذا (src_backend_id != cur_backend_id && !ggml_backend_sched_بفر_supported(sched, src, cur_backend_id)) {
                            need_new_ونڈو = سچ;
                            اكسر;
                        }
                    }
                }
            }

            إذا (گرہ_backend_id != cur_backend_id || need_new_ونڈو) {
                ونڈو->i_سرا = i;
                i_ونڈو++;
                إذا (i_ونڈو >= sched->splits_سمرتھا) {
                    صحيح old_cap = sched->splits_سمرتھا;
                    sched->splits_سمرتھا *= 2;
                    sched->splits = (ggml_backend_sched_ونڈو *)
                        realloc(sched->splits, sched->splits_سمرتھا * حجم(بنية ggml_backend_sched_ونڈو));
                    GGML_ASSERT(sched->splits != عدم);
                    لكل (صحيح k = old_cap; k < sched->splits_سمرتھا; k++) {
                        memset(&sched->splits[k], 0, حجم(بنية ggml_backend_sched_ونڈو));
                    }
                }
                ونڈو = &sched->splits[i_ونڈو];
                ونڈو->backend_id = گرہ_backend_id;
                ونڈو->i_شروع = i;
                ونڈو->n_inputs = 0;
                cur_backend_id = گرہ_backend_id;
            }

            // find inputs that are not on the same backend
            لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
                بنية ggml_tensor * src = گرہ->src[j];
                إذا (src == عدم) {
                    استمر;
                }

                size_t src_id = hash_id(src);
                ثابت صحيح src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                إذا (src_backend_id != cur_backend_id && !ggml_backend_sched_بفر_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    إذا (tensor_id_copy(src_id, cur_backend_id, 0) == عدم) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        لكل (صحيح c = 0; c < sched->n_copies; c++) {
                            بنية ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            إذا (sched->n_copies > 1) {
                                ggml_سیٹ_داخلہ(tensor_copy);
                                ggml_سیٹ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            سیٹ_CAUSE(tensor_copy, "4.cpy");
                        }
                        صحيح n_inputs = ونڈو->n_inputs++;
                        إذا (n_inputs >= ونڈو->inputs_سمرتھا) {
                            ggml_backend_sched_ونڈو_inputs_grow(ونڈو);
                        }
                        ونڈو->inputs[n_inputs] = src;
                    }
                    گرہ->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        ونڈو->i_سرا = graph->n_nodes;
        sched->n_splits = i_ونڈو + 1;
    }

    إذا (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    إذا (sched->n_copies > 1) {
        لكل (صحيح i = 0; i < graph->n_leafs; i++) {
            بنية ggml_tensor * پتی = graph->leafs[i];
            إذا ((پتی->flags & GGML_TENSOR_FLAG_داخلہ) == 0) {
                استمر;
            }

            ثابت size_t پتی_id = hash_id(پتی);
            ثابت صحيح پتی_backend_id = tensor_backend_id(پتی);
            GGML_ASSERT(پتی_backend_id != -1); // all leafs should be assigned by now

            إذا (tensor_id_copy(پتی_id, پتی_backend_id, 0) == عدم) {
                ggml_backend_t backend = sched->backends[پتی_backend_id];
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * tensor_copy;
                    إذا (c == sched->cur_copy) {
                        tensor_copy = پتی; // use the original tensor as the current copy
                    } وإلا {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, پتی);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), پتی->name, c);
                    }
                    ggml_سیٹ_داخلہ(tensor_copy);
                    ggml_سیٹ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(پتی_id, پتی_backend_id, c) = tensor_copy;
                    سیٹ_CAUSE(tensor_copy, "6.cpy");
                }
            }

            صحيح n_گراف_inputs = sched->n_گراف_inputs++;
            إذا (n_گراف_inputs >= sched->گراف_inputs_سمرتھا) {
                ggml_backend_sched_گراف_inputs_grow(sched);
            }
            sched->گراف_inputs[n_گراف_inputs] = پتی;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        صحيح * tmp = sched->گرہ_backend_ids;
        sched->گرہ_backend_ids = sched->prev_گرہ_backend_ids;
        sched->prev_گرہ_backend_ids = tmp;

        tmp = sched->پتی_backend_ids;
        sched->پتی_backend_ids = sched->prev_پتی_backend_ids;
        sched->prev_پتی_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    بنية ggml_backend_گراف_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](فراغ * user_انکڑے, ggml_tensor * tensor, ggml_tensor * until) {
            خودکار & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_انکڑے;
            std::vector<ggml_tensor *> & keep = deps[until];
            إذا (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_ونڈو * ونڈو = &sched->splits[i];
        ونڈو->graph = ggml_گراف_view(graph, ونڈو->i_شروع, ونڈو->i_سرا);

        ggml_backend_گراف_optimize(sched->backends[ونڈو->backend_id], &ونڈو->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    صحيح n_dep_nodes = 0;
    لكل (ثابت خودکار & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    صحيح کل_inputs = sched->n_گراف_inputs;
    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        کل_inputs += sched->splits[i].n_inputs;
    }
    صحيح گراف_ناپ = std::max(graph->n_nodes, graph->n_leafs) + کل_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_گراف_ناپ = sched->گراف_ناپ;
    sched->گراف_ناپ = گراف_ناپ;

    إذا (sched->graph.size < گراف_ناپ) {
        sched->graph.size = گراف_ناپ;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, گراف_ناپ * حجم(بنية ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, گراف_ناپ * حجم(بنية ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != عدم);
        GGML_ASSERT(sched->graph.leafs != عدم);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    بنية ggml_cgraph * گراف_copy = &sched->graph;

    صحيح n_dep_nodes_added = 0;

    لكل (صحيح i = 0; i < sched->n_splits; i++) {
        بنية ggml_backend_sched_ونڈو * ونڈو = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        لكل (صحيح j = 0; j < ونڈو->n_inputs; j++) {
            أكد(گراف_copy->size > (گراف_copy->n_nodes + 1));

            بنية ggml_tensor * داخلہ = ونڈو->inputs[j];
            ثابت size_t داخلہ_id = hash_id(داخلہ);
            بنية ggml_tensor * داخلہ_cpy = tensor_id_copy(داخلہ_id, ونڈو->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            بنية ggml_tensor * داخلہ_dep = ggml_view_tensor(sched->ctx, داخلہ);
            داخلہ_dep->src[0] = داخلہ;
            sched->گرہ_backend_ids[گراف_copy->n_nodes] = sched->hv_tensor_backend_ids[داخلہ_id];
            گراف_copy->nodes[گراف_copy->n_nodes++] = داخلہ_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->گرہ_backend_ids[گراف_copy->n_nodes] = ونڈو->backend_id;
            گراف_copy->nodes[گراف_copy->n_nodes++] = داخلہ_cpy;
        }

        لكل (صحيح j = ونڈو->i_شروع; j < ونڈو->i_سرا; j++) {
            أكد(گراف_copy->size > گراف_copy->n_nodes);
            sched->گرہ_backend_ids[گراف_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            گراف_copy->nodes[گراف_copy->n_nodes++] = graph->nodes[j];

            إذا (alloc_deps.خالی_تھاواں()) {
                استمر;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            خودکار it = alloc_deps.find(graph->nodes[j]);
            إذا (it != alloc_deps.end()) {
                ثابت std::vector<ggml_tensor *> & keep = it->second;
                لكل (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    بنية ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    لكل (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    أكد(گراف_copy->size > گراف_copy->n_nodes);
                    sched->گرہ_backend_ids[گراف_copy->n_nodes] = ونڈو->backend_id;
                    گراف_copy->nodes[گراف_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    إذا (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        لكل (صحيح i = 0; i < sched->n_گراف_inputs; i++) {
            بنية ggml_tensor * داخلہ = sched->گراف_inputs[i];
            size_t id = hash_id(داخلہ);
            صحيح backend_id = tensor_backend_id(داخلہ);
            لكل (صحيح c = 0; c < sched->n_copies; c++) {
                بنية ggml_tensor * داخلہ_cpy = tensor_id_copy(id, backend_id, c);
                sched->پتی_backend_ids[گراف_copy->n_leafs] = backend_id;
                أكد(گراف_copy->size > گراف_copy->n_leafs);
                گراف_copy->leafs[گراف_copy->n_leafs++] = داخلہ_cpy;
            }
        }

        لكل (صحيح i = 0; i < sched->n_splits; i++) {
            بنية ggml_backend_sched_ونڈو * ونڈو = &sched->splits[i];
            صحيح backend_id = ونڈو->backend_id;
            لكل (صحيح j = 0; j < ونڈو->n_inputs; j++) {
                بنية ggml_tensor * داخلہ = ونڈو->inputs[j];
                size_t id = hash_id(داخلہ);
                لكل (صحيح c = 0; c < sched->n_copies; c++) {
                    بنية ggml_tensor * داخلہ_cpy = tensor_id_copy(id, backend_id, c);
                    sched->پتی_backend_ids[گراف_copy->n_leafs] = backend_id;
                    أكد(گراف_copy->size > گراف_copy->n_leafs);
                    گراف_copy->leafs[گراف_copy->n_leafs++] = داخلہ_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    لكل (صحيح i = 0; i < graph->n_leafs; i++) {
        بنية ggml_tensor * پتی = graph->leafs[i];
        sched->پتی_backend_ids[گراف_copy->n_leafs] = tensor_backend_id(پتی);
        أكد(گراف_copy->size > گراف_copy->n_leafs);
        گراف_copy->leafs[گراف_copy->n_leafs++] = پتی;
    }

    // set ids for all splits
    لكل (صحيح i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_گراف_اگلا_uid();
    }
}

ساكن منطقي ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    منطقي backend_ids_بدلیا = خطأ;
    لكل (صحيح i = 0; i < sched->graph.n_nodes; i++) {
        إذا (sched->گرہ_backend_ids[i] != sched->prev_گرہ_backend_ids[i] &&
            sched->bufts[sched->گرہ_backend_ids[i]] != sched->bufts[sched->prev_گرہ_backend_ids[i]]) {
            backend_ids_بدلیا = سچ;
            اكسر;
        }
    }
    إذا (!backend_ids_بدلیا) {
        لكل (صحيح i = 0; i < sched->graph.n_leafs; i++) {
            إذا (sched->پتی_backend_ids[i] != sched->prev_پتی_backend_ids[i] &&
                sched->bufts[sched->پتی_backend_ids[i]] != sched->bufts[sched->prev_پتی_backend_ids[i]]) {
                backend_ids_بدلیا = سچ;
                اكسر;
            }
        }
    }

    // allocate graph
    إذا (backend_ids_بدلیا || !ggml_gallocr_alloc_گراف(sched->galloc, &sched->graph)) {
#جے_بنا_تعریف NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_بدلیا);
#جے_انت

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ثابت منطقي گراف_ناپ_بدلیا = sched->prev_گراف_ناپ != sched->گراف_ناپ;

        إذا (sched->callback_realloc != عدم) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_بدلیا, گراف_ناپ_بدلیا, sched->callback_realloc_user_انکڑے);
        }

        إذا (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ثابت منطقي unexpected = !backend_ids_بدلیا && !گراف_ناپ_بدلیا;

            إذا (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->گراف_ناپ, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        لكل (صحيح i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        إذا (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گرہ_backend_ids, sched->پتی_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            أرجع خطأ;
        }
        إذا (!ggml_gallocr_alloc_گراف(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

ساكن منطقي ggml_backend_sched_is_host_بھار(ثابت بنية ggml_tensor * t) {
    أرجع t->بفر != عدم &&
        ggml_backend_بفر_get_usage(t->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS &&
        ggml_backend_بفر_is_host(t->بفر);
}

ساكن فراغ ggml_backend_sched_copy_داخلہ(ggml_backend_sched_t sched, بنية ggml_backend_sched_ونڈو * ونڈو, بنية ggml_tensor * داخلہ) {
    ثابت صحيح ونڈو_backend_id = ونڈو->backend_id;
    ggml_backend_t ونڈو_backend = sched->backends[ونڈو_backend_id];
    ggml_backend_t داخلہ_backend = ggml_backend_sched_get_tensor_backend(sched, داخلہ);
    بنية ggml_tensor * داخلہ_cpy = tensor_copy(داخلہ, ونڈو_backend_id, sched->cur_copy);

    إذا (داخلہ->flags & GGML_TENSOR_FLAG_داخلہ) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        إذا (sched->events[ونڈو_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[ونڈو_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(ونڈو_backend);
        }
        ggml_backend_tensor_copy(داخلہ, داخلہ_cpy);
        أرجع;
    }

    // wait for the split backend to finish using the input before overwriting it
    إذا (sched->events[ونڈو_backend_id][sched->cur_copy] != عدم) {
        ggml_backend_event_wait(ونڈو_backend, sched->events[ونڈو_backend_id][sched->cur_copy]);
    } وإلا {
        ggml_backend_synchronize(ونڈو_backend);
    }

    إذا (sched->callback_copy != عدم && ggml_backend_sched_is_host_بھار(داخلہ) &&
        sched->callback_copy(ونڈو_backend, داخلہ, داخلہ_cpy, &ونڈو->graph, sched->callback_copy_user_انکڑے)) {
        أرجع;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    إذا (!ونڈو_backend->iface.cpy_tensor_async || !ونڈو_backend->iface.cpy_tensor_async(داخلہ_backend, ونڈو_backend, داخلہ, داخلہ_cpy)) {
        ggml_backend_synchronize(داخلہ_backend);
        إذا (sched->events[ونڈو_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_synchronize(sched->events[ونڈو_backend_id][sched->cur_copy]);
        } وإلا {
            ggml_backend_synchronize(ونڈو_backend);
        }
        ggml_backend_tensor_copy(داخلہ, داخلہ_cpy);
    }
}

ساكن تعداد ggml_حالت ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    بنية ggml_backend_sched_ونڈو * splits = sched->splits;

    صحيح prev_backend_id = -1;

    لكل (صحيح ونڈو_id = 0; ونڈو_id < sched->n_splits; ونڈو_id++) {
        بنية ggml_backend_sched_ونڈو * ونڈو = &splits[ونڈو_id];
        صحيح ونڈو_backend_id = ونڈو->backend_id;
        ggml_backend_t ونڈو_backend = sched->backends[ونڈو_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        إذا (ونڈو->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != ونڈو_backend_id) {
            إذا (sched->events[prev_backend_id][sched->cur_copy] != عدم) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } وإلا {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        لكل (صحيح داخلہ_id = 0; داخلہ_id < ونڈو->n_inputs; داخلہ_id++) {
            إذا (!ggml_backend_sched_is_host_بھار(ونڈو->inputs[داخلہ_id])) {
                ggml_backend_sched_copy_داخلہ(sched, ونڈو, ونڈو->inputs[داخلہ_id]);
            }
        }
        لكل (صحيح داخلہ_id = 0; داخلہ_id < ونڈو->n_inputs; داخلہ_id++) {
            إذا (ggml_backend_sched_is_host_بھار(ونڈو->inputs[داخلہ_id])) {
                ggml_backend_sched_copy_داخلہ(sched, ونڈو, ونڈو->inputs[داخلہ_id]);
            }
        }

        إذا (!sched->callback_eval) {
            تعداد ggml_حالت ec = ggml_backend_گراف_compute_async(ونڈو_backend, &ونڈو->graph);
            إذا (ec != GGML_حالت_SUCCESS) {
                أرجع ec;
            }
        } وإلا {
            // similar to ggml_backend_compare_graph_backend
            لكل (صحيح j0 = 0; j0 < ونڈو->graph.n_nodes; j0++) {
                بنية ggml_tensor * t = ونڈو->graph.nodes[j0];

                // check if the user needs data from this node
                منطقي need = sched->callback_eval(t, سچ, sched->callback_eval_user_انکڑے);

                صحيح j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                طالما (!need && j1 < ونڈو->graph.n_nodes - 1) {
                    t = ونڈو->graph.nodes[++j1];
                    need = sched->callback_eval(t, سچ, sched->callback_eval_user_انکڑے);
                }

                بنية ggml_cgraph gv = ggml_گراف_view(&ونڈو->graph, j0, j1 + 1);

                تعداد ggml_حالت ec = ggml_backend_گراف_compute_async(ونڈو_backend, &gv);
                إذا (ec != GGML_حالت_SUCCESS) {
                    أرجع ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(ونڈو_backend);

                إذا (need && !sched->callback_eval(t, خطأ, sched->callback_eval_user_انکڑے)) {
                    اكسر;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        إذا (sched->events[ونڈو_backend_id][sched->cur_copy] != عدم) {
            ggml_backend_event_ریکارڈ(sched->events[ونڈو_backend_id][sched->cur_copy], ونڈو_backend);
        }

        prev_backend_id = ونڈو_backend_id;
    }

    أرجع GGML_حالت_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_بفر_type_t * bufts,
        صحيح n_backends,
        size_t گراف_ناپ,
        منطقي parallel,
        منطقي op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    بنية ggml_backend_sched * sched = (ggml_backend_sched *) صفر_حافظہ_دیو(1, حجم(بنية ggml_backend_sched));

    ثابت حرف * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#جے_تعریف_شدہ GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#جے_انت
    ثابت حرف * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_سیٹ    = ggml_hash_سیٹ_new(گراف_ناپ);
    sched->hv_tensor_backend_ids = (صحيح *) حافظہ_دیو(sched->hash_سیٹ.size * حجم(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) حافظہ_دیو(sched->hash_سیٹ.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));

    ثابت size_t ggml_sched_max_splits = گراف_ناپ; // at most there is one split for each node in the graph
    ثابت size_t nodes_ناپ = گراف_ناپ + ggml_sched_max_splits*GGML_SCHED_MAX_ونڈو_INPUTS*2;
    sched->گرہ_backend_ids = (صحيح *) صفر_حافظہ_دیو(nodes_ناپ, حجم(sched->گرہ_backend_ids[0]));
    sched->پتی_backend_ids = (صحيح *) صفر_حافظہ_دیو(nodes_ناپ, حجم(sched->پتی_backend_ids[0]));
    sched->prev_گرہ_backend_ids = (صحيح *) صفر_حافظہ_دیو(nodes_ناپ, حجم(sched->prev_گرہ_backend_ids[0]));
    sched->prev_پتی_backend_ids = (صحيح *) صفر_حافظہ_دیو(nodes_ناپ, حجم(sched->prev_پتی_backend_ids[0]));

    sched->گراف_ناپ      = 0;
    sched->prev_گراف_ناپ = 0;

    sched->context_بفر_ناپ = ggml_sched_max_splits*GGML_SCHED_MAX_ونڈو_INPUTS*2*حجم(بنية ggml_tensor) + ggml_گراف_overhead_custom(گراف_ناپ, خطأ);
    sched->context_بفر = (حرف *) حافظہ_دیو(sched->context_بفر_ناپ);

    ثابت صحيح initial_splits_سمرتھا = 16;
    sched->splits = (ggml_backend_sched_ونڈو *) صفر_حافظہ_دیو(initial_splits_سمرتھا, حجم(sched->splits[0]));
    sched->splits_سمرتھا = initial_splits_سمرتھا;

    sched->گراف_inputs_سمرتھا = GGML_SCHED_MAX_ونڈو_INPUTS;
    sched->گراف_inputs = (بنية ggml_tensor **) صفر_حافظہ_دیو(sched->گراف_inputs_سمرتھا, حجم(بنية ggml_tensor *));

    لكل (صحيح b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_بفر_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        إذا (sched->n_copies > 1) {
            لكل (صحيح c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    أرجع sched;
}

فراغ ggml_backend_sched_free(ggml_backend_sched_t sched) {
    إذا (sched == عدم) {
        أرجع;
    }
    لكل (صحيح b = 0; b < sched->n_backends; b++) {
        لكل (صحيح c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_سیٹ_free(&sched->hash_سیٹ);
    لكل (صحيح i = 0; i < sched->splits_سمرتھا; i++) {
        حرر(sched->splits[i].inputs);
    }
    حرر(sched->splits);
    حرر(sched->گراف_inputs);
    حرر(sched->hv_tensor_backend_ids);
    حرر(sched->hv_tensor_copies);
    حرر(sched->گرہ_backend_ids);
    حرر(sched->پتی_backend_ids);
    حرر(sched->prev_گرہ_backend_ids);
    حرر(sched->prev_پتی_backend_ids);
    حرر(sched->context_بفر);
    حرر(sched->graph.nodes);
    حرر(sched->graph.leafs);
    حرر(sched);
}

فراغ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    إذا (!sched->is_reset) {
        ggml_hash_سیٹ_reset(&sched->hash_سیٹ);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_سیٹ.size * حجم(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_سیٹ.size * sched->n_backends * sched->n_copies * حجم(بنية ggml_tensor *));
        sched->is_reset = سچ;
    }
    sched->is_alloc = خطأ;
}

فراغ ggml_backend_sched_reserve_ناپ(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_گراف, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_سیٹ.size >= measure_گراف->n_nodes + measure_گراف->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_ونڈو_گراف(sched, measure_گراف);

    ggml_gallocr_reserve_n_ناپ(sched->galloc, &sched->graph, sched->گرہ_backend_ids, sched->پتی_backend_ids, sizes);
}

منطقي ggml_backend_sched_reserve(ggml_backend_sched_t sched, بنية ggml_cgraph * measure_گراف) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_سیٹ.size >= measure_گراف->n_nodes + measure_گراف->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_ونڈو_گراف(sched, measure_گراف);

    إذا (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->گرہ_backend_ids, sched->پتی_backend_ids)) {
        أرجع خطأ;
    }

    ggml_backend_sched_reset(sched);

    أرجع سچ;
}

منطقي ggml_backend_sched_alloc_گراف(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((صحيح)sched->hash_سیٹ.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->اگلا_copy;
    sched->اگلا_copy = (sched->اگلا_copy + 1) % sched->n_copies;

    ggml_backend_sched_ونڈو_گراف(sched, graph);

    إذا (!ggml_backend_sched_alloc_splits(sched)) {
        أرجع خطأ;
    }

    sched->is_alloc = سچ;

    أرجع سچ;
}

تعداد ggml_حالت ggml_backend_sched_گراف_compute(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    تعداد ggml_حالت err = ggml_backend_sched_گراف_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    أرجع err;
}

تعداد ggml_حالت ggml_backend_sched_گراف_compute_async(ggml_backend_sched_t sched, بنية ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    إذا (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    إذا (!sched->is_alloc) {
        إذا (!ggml_backend_sched_alloc_گراف(sched, graph)) {
            أرجع GGML_حالت_ALLOC_FAILED;
        }
    }

    أرجع ggml_backend_sched_compute_splits(sched);
}

فراغ ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    لكل (صحيح i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    إذا (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->اگلا_copy = 0;
    }
}

فراغ ggml_backend_sched_سیٹ_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, فراغ * user_انکڑے) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_انکڑے = user_انکڑے;
}

فراغ ggml_backend_sched_سیٹ_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, فراغ * user_انکڑے) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_انکڑے = user_انکڑے;
}

فراغ ggml_backend_sched_سیٹ_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, فراغ * user_انکڑے) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_انکڑے = user_انکڑے;
}

صحيح ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    أرجع sched->n_splits;
}

صحيح ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    أرجع sched->n_copies;
}

صحيح ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    أرجع sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, صحيح i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    أرجع sched->backends[i];
}

ggml_backend_بفر_type_t ggml_backend_sched_get_بفر_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_اشاریہ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_اشاریہ >= 0 && backend_اشاریہ < sched->n_backends);

    أرجع sched->bufts[backend_اشاریہ];
}

size_t ggml_backend_sched_get_بفر_ناپ(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_اشاریہ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_اشاریہ >= 0 && backend_اشاریہ < sched->n_backends);

    أرجع ggml_gallocr_get_بفر_ناپ(sched->galloc, backend_اشاریہ);
}

فراغ ggml_backend_sched_سیٹ_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * گرہ, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    صحيح backend_اشاریہ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_اشاریہ >= 0 && backend_اشاریہ < sched->n_backends);
    tensor_backend_id(گرہ) = backend_اشاریہ;
    سیٹ_CAUSE(گرہ, "usr");
    sched->is_reset = خطأ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, بنية ggml_tensor * گرہ) {
    GGML_ASSERT(sched);
    صحيح backend_اشاریہ = tensor_backend_id(گرہ);
    إذا (backend_اشاریہ == -1) {
        أرجع عدم;
    }
    أرجع sched->backends[backend_اشاریہ];
}

// utils

منطقي ggml_op_alloc_ناپ_may_expand(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_FLASH_ATTN_EXT:
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_چوٹی_K:
        حالة GGML_OP_ARGMAX:
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

تعداد ggml_حالت ggml_backend_view_init(بنية ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->بفر == عدم);
    GGML_ASSERT(tensor->view_src != عدم);
    GGML_ASSERT(tensor->view_src->بفر != عدم);
    GGML_ASSERT(tensor->view_src->انکڑے != عدم);

    tensor->بفر = tensor->view_src->بفر;
    tensor->انکڑے = (حرف *)tensor->view_src->انکڑے + tensor->view_offs;
    أرجع ggml_backend_بفر_init_tensor(tensor->بفر, tensor);
}

تعداد ggml_حالت ggml_backend_tensor_alloc(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor, فراغ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->بفر == عدم);
    GGML_ASSERT(tensor->انکڑے == عدم);
    GGML_ASSERT(tensor->view_src == عدم);
    GGML_ASSERT(addr >= ggml_backend_بفر_get_ادھار(بفر));
    GGML_ASSERT(ggml_backend_بفر_is_meta(بفر) ||
        (حرف *) addr + ggml_backend_بفر_get_alloc_ناپ(بفر, tensor) <=
        (حرف *) ggml_backend_بفر_get_ادھار(بفر) + ggml_backend_بفر_get_ناپ(بفر));

    tensor->بفر = بفر;
    tensor->انکڑے = addr;
    أرجع ggml_backend_بفر_init_tensor(بفر, tensor);
}

ساكن بنية ggml_tensor * گراف_copy_dup_tensor(بنية ggml_hash_سیٹ hash_سیٹ, بنية ggml_tensor ** گرہ_copies,
    بنية ggml_context * ctx_allocated, بنية ggml_context * ctx_unallocated, بنية ggml_tensor * src) {

    GGML_ASSERT(src != عدم);
    GGML_ASSERT(src->انکڑے && "graph must be allocated");

    size_t id = ggml_hash_داخل_کرو(&hash_سیٹ, src);
    إذا (id == GGML_HASHSET_ALREADY_EXISTS) {
        أرجع گرہ_copies[ggml_hash_لبھو(&hash_سیٹ, src)];
    }

    بنية ggml_tensor * dst = ggml_dup_tensor_layout(src->انکڑے && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    إذا (src->view_src != عدم) {
        dst->view_src = گراف_copy_dup_tensor(hash_سیٹ, گرہ_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    حافظہ_نقل(dst->op_params, src->op_params, حجم(dst->op_params));
    ggml_سیٹ_name(dst, src->name);

    // copy src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        إذا (s == عدم) {
            استمر;
        }
        dst->src[i] = گراف_copy_dup_tensor(hash_سیٹ, گرہ_copies, ctx_allocated, ctx_unallocated, s);
    }

    گرہ_copies[id] = dst;
    أرجع dst;
}

ساكن فراغ گراف_copy_init_tensor(بنية ggml_hash_سیٹ * hash_سیٹ, بنية ggml_tensor ** گرہ_copies, منطقي * گرہ_init, بنية ggml_tensor * src) {
    size_t id = ggml_hash_لبھو(hash_سیٹ, src);
    إذا (گرہ_init[id]) {
        أرجع;
    }
    گرہ_init[id] = سچ;

    بنية ggml_tensor * dst = گرہ_copies[id];
    إذا (dst->view_src != عدم) {
        گراف_copy_init_tensor(hash_سیٹ, گرہ_copies, گرہ_init, src->view_src);
        تعداد ggml_حالت حالت = ggml_backend_view_init(dst);
        GGML_ASSERT(حالت == GGML_حالت_SUCCESS);
    }
    وإلا {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        بنية ggml_tensor * s = src->src[i];
        إذا (s == عدم) {
            استمر;
        }
        گراف_copy_init_tensor(hash_سیٹ, گرہ_copies, گرہ_init, s);
    }
}

بنية ggml_backend_گراف_copy ggml_backend_گراف_copy(ggml_backend_t backend, بنية ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    بنية ggml_hash_سیٹ hash_سیٹ = ggml_hash_سیٹ_new(graph->ویکھیا_hash_سیٹ.size);
    بنية ggml_tensor ** گرہ_copies = (ggml_tensor **) صفر_حافظہ_دیو(hash_سیٹ.size, حجم(گرہ_copies[0])); // NOLINT
    منطقي * گرہ_init = (منطقي *) صفر_حافظہ_دیو(hash_سیٹ.size, حجم(گرہ_init[0]));

    بنية ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_سیٹ.size + ggml_گراف_overhead_custom(graph->size, خطأ),
        /* .mem_buffer = */ عدم,
        /* .no_alloc   = */ سچ
    };

    بنية ggml_context * ctx_allocated = ggml_init(params);
    بنية ggml_context * ctx_unallocated = ggml_init(params);

    إذا (ctx_allocated == عدم || ctx_unallocated == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_سیٹ_free(&hash_سیٹ);
        حرر(گرہ_copies);
        حرر(گرہ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        أرجع {
            /* .buffer           = */ عدم,
            /* .ctx_allocated    = */ عدم,
            /* .ctx_unallocated  = */ عدم,
            /* .graph            = */ عدم,
        };
    }

    // dup nodes
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = graph->nodes[i];
        گراف_copy_dup_tensor(hash_سیٹ, گرہ_copies, ctx_allocated, ctx_unallocated, گرہ);
    }

    // allocate nodes
    ggml_backend_بفر_t بفر = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    إذا (بفر == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_سیٹ_free(&hash_سیٹ);
        حرر(گرہ_copies);
        حرر(گرہ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        أرجع {
            /* .buffer           = */ عدم,
            /* .ctx_allocated    = */ عدم,
            /* .ctx_unallocated  = */ عدم,
            /* .graph            = */ عدم,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = graph->nodes[i];
        گراف_copy_init_tensor(&hash_سیٹ, گرہ_copies, گرہ_init, گرہ);
    }

    // build graph copy
    بنية ggml_cgraph * گراف_copy = ggml_new_گراف_custom(ctx_allocated, graph->size, خطأ);
    لكل (صحيح i = 0; i < graph->n_nodes; i++) {
        بنية ggml_tensor * گرہ = graph->nodes[i];
        بنية ggml_tensor * گرہ_copy = گرہ_copies[ggml_hash_لبھو(&hash_سیٹ, گرہ)];
        گراف_copy->nodes[i] = گرہ_copy;
    }
    گراف_copy->n_nodes = graph->n_nodes;

    ggml_hash_سیٹ_free(&hash_سیٹ);
    حرر(گرہ_copies);
    حرر(گرہ_init);

    أرجع {
        /* .buffer           = */ بفر,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ گراف_copy,
    };
}

فراغ ggml_backend_گراف_copy_free(بنية ggml_backend_گراف_copy copy) {
    ggml_backend_بفر_free(copy.بفر);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

منطقي ggml_backend_compare_گراف_backend(ggml_backend_t backend1, ggml_backend_t backend2, بنية ggml_cgraph * graph, ggml_backend_eval_callback callback, فراغ * user_انکڑے, بنية ggml_tensor ثابت * ثابت * test_nodes, size_t عدد_test_nodes) {
    بنية ggml_backend_گراف_copy copy = ggml_backend_گراف_copy(backend2, graph);
    إذا (copy.بفر == عدم) {
        أرجع خطأ;
    }

    بنية ggml_cgraph * g1 = graph;
    بنية ggml_cgraph * g2 = copy.graph;

    أكد(g1->n_nodes == g2->n_nodes);

    إذا (عدد_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_گراف_compute(backend1, g1);
        ggml_backend_گراف_compute(backend2, g2);

        منطقي verified = خطأ;
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            لكل (size_t j = 0; j < عدد_test_nodes; ++j) {
                إذا (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_انکڑے);
                    verified = سچ;
                }
            }
        }
        GGML_ASSERT(verified);
    } وإلا {
        لكل (صحيح i = 0; i < g1->n_nodes; i++) {
            بنية ggml_tensor * t1 = g1->nodes[i];
            بنية ggml_tensor * t2 = g2->nodes[i];

            أكد(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            بنية ggml_cgraph g1v = ggml_گراف_view(g1, i, i + 1);
            بنية ggml_cgraph g2v = ggml_گراف_view(g2, i, i + 1);

            ggml_backend_گراف_compute(backend1, &g1v);
            ggml_backend_گراف_compute(backend2, &g2v);

            إذا (ggml_is_view_op(t1->op)) {
                استمر;
            }

            // compare results, calculate rms etc
            إذا (!callback(i, t1, t2, user_انکڑے)) {
                اكسر;
            }
        }
    }
    ggml_backend_گراف_copy_free(copy);

    أرجع سچ;
}

// CPU backend - buffer

ساكن فراغ * ggml_backend_cpu_بفر_get_ادھار(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    uintptr_t انکڑے = (uintptr_t)بفر->context;

    // align the buffer
    إذا (انکڑے % TENSOR_ALIGNMENT != 0) {
        انکڑے = GGML_PAD(انکڑے, TENSOR_ALIGNMENT);
    }

    أرجع (فراغ *)انکڑے;
}

ساكن فراغ ggml_backend_cpu_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    GGML_ASSERT(بفر);
    ggml_aligned_free(بفر->context, بفر->size);
}

ساكن فراغ ggml_backend_cpu_بفر_memset_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    GGML_ASSERT(tensor);
    memset((حرف *)tensor->انکڑے + وتھ, value, size);

    GGML_بن_ورتیا(بفر);
}

ساكن فراغ ggml_backend_cpu_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, بنية ggml_tensor * tensor, ثابت فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(tensor);
    حافظہ_نقل((حرف *)tensor->انکڑے + وتھ, انکڑے, size);

    GGML_بن_ورتیا(بفر);
}

ساكن فراغ ggml_backend_cpu_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت بنية ggml_tensor * tensor, فراغ * انکڑے, size_t وتھ, size_t size) {
    GGML_ASSERT(tensor);
    حافظہ_نقل(انکڑے, (ثابت حرف *)tensor->انکڑے + وتھ, size);

    GGML_بن_ورتیا(بفر);
}

ساكن منطقي ggml_backend_cpu_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت بنية ggml_tensor * src, بنية ggml_tensor * dst) {
    GGML_ASSERT(src);
    إذا (ggml_backend_بفر_is_host(src->بفر)) {
        حافظہ_نقل(dst->انکڑے, src->انکڑے, ggml_nbytes(src));
        أرجع سچ;
    }
    أرجع خطأ;

    GGML_بن_ورتیا(بفر);
}

ساكن فراغ ggml_backend_cpu_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_ASSERT(بفر);
    memset(بفر->context, value, بفر->size);
}

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_cpu_بفر_i = {
    /* .free_buffer     = */ ggml_backend_cpu_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_cpu_بفر_get_ادھار,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_بفر_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_cpu_بفر_from_ptr_i = {
    /* .free_buffer     = */ عدم, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_بفر_get_ادھار,
    /* .init_tensor     = */ عدم, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cpu_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_بفر_clear,
    /* .reset           = */ عدم,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ساكن ثابت حرف * ggml_backend_cpu_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU";

    GGML_بن_ورتیا(buft);
}

ساكن ggml_backend_بفر_t ggml_backend_cpu_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    فراغ * انکڑے = ggml_aligned_malloc(size);

    إذا (انکڑے == عدم) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        أرجع عدم;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_cpu_بفر_i, انکڑے, size);
}

ساكن size_t ggml_backend_cpu_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع TENSOR_ALIGNMENT;

    GGML_بن_ورتیا(buft);
}

ساكن منطقي ggml_backend_cpu_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    أرجع سچ;

    GGML_بن_ورتیا(buft);
}

ggml_backend_بفر_type_t ggml_backend_cpu_بفر_type(فراغ) {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_بفر_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_بفر_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_بفر_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_بفر_type;
}

ساكن ثابت حرف * ggml_backend_cpu_بفر_from_ptr_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع "CPU_Mapped";

    GGML_بن_ورتیا(buft);
}

ساكن ggml_backend_بفر_type_t ggml_backend_cpu_بفر_from_ptr_type(فراغ) {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cpu_بفر_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_بفر_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_بفر_type_alloc_بفر,
            /* .alloc_buffer_n      = */ عدم,
            /* .get_alignment       = */ ggml_backend_cpu_بفر_type_get_alignment,
            /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ عدم, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ عدم,
            /* .is_host             = */ ggml_backend_cpu_بفر_type_is_host,
        },
        /* .device  = */ عدم, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ عدم,
    };

    أرجع &ggml_backend_cpu_بفر_type;
}

ggml_backend_بفر_t ggml_backend_cpu_بفر_from_ptr(فراغ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    أرجع ggml_backend_بفر_init(ggml_backend_cpu_بفر_from_ptr_type(), ggml_backend_cpu_بفر_from_ptr_i, ptr, size);
}
