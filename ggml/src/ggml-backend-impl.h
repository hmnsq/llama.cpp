#ہدایت once

// ggml-backend internal header

#شامل "ggml-backend.h"

#اگر_تعریف_شدہ  __cplusplus
خارجی "C" {
#اگر_ختم

    #تعریف_کرو GGML_BACKEND_API_نسخہ 3

    //
    // Backend buffer type
    //

    ساخت ggml_backend_بفر_type_i {
        ثابت حرف *          (*get_name)        (ggml_backend_بفر_type_t buft);
        // allocate a buffer of this type
        ggml_backend_بفر_t (*alloc_بفر)    (ggml_backend_بفر_type_t buft, size_t سائز);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_بفر_t (*alloc_بفر_n)  (ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_بفر_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_سائز)    (ggml_backend_بفر_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_سائز)  (ggml_backend_بفر_type_t buft, ثابت ساخت ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_سائز_n)(ggml_backend_بفر_type_t buft, ساخت ggml_tensor ** tensors, صحیح n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_بفر_type_t buft);
    };

    ساخت ggml_backend_بفر_type {
        ساخت ggml_backend_بفر_type_i  iface;
        ggml_backend_dev_t device;
        خالی * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_سائز_may_expand(شمار ggml_op op);

    //
    // Backend buffer
    //

    ساخت ggml_backend_بفر_i {
        // (optional) free the buffer
        خالی         (*free_بفر)  (ggml_backend_بفر_t بفر);
        // base address of the buffer
        خالی *       (*get_بنیاد)     (ggml_backend_بفر_t بفر);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        شمار ggml_حالت (*init_tensor)(ggml_backend_بفر_t بفر, ساخت ggml_tensor * tensor);
        // tensor data access
        خالی         (*memset_tensor)(ggml_backend_بفر_t بفر,       ساخت ggml_tensor * tensor,     uint8_t قدر, size_t انحراف, size_t سائز);
        خالی         (*سیٹ_tensor)   (ggml_backend_بفر_t بفر,       ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز);
        خالی         (*get_tensor)   (ggml_backend_بفر_t بفر, ثابت ساخت ggml_tensor * tensor,       خالی * ڈیٹا, size_t انحراف, size_t سائز);
        // (optional) 2d data copies
        خالی         (*سیٹ_tensor_2d)(ggml_backend_بفر_t بفر,       ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا);
        خالی         (*get_tensor_2d)(ggml_backend_بفر_t بفر, ثابت ساخت ggml_tensor * tensor,       خالی * ڈیٹا, size_t انحراف, size_t سائز, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_بفر_t بفر, ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst);
        // clear the entire buffer
        خالی         (*clear)        (ggml_backend_بفر_t بفر, uint8_t قدر);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        خالی         (*reset)        (ggml_backend_بفر_t بفر);
    };

    ساخت ggml_backend_بفر {
        ساخت ggml_backend_بفر_i  iface;
        ggml_backend_بفر_type_t    buft;
        خالی * context;
        size_t سائز;
        شمار ggml_backend_بفر_usage usage;
    };

    GGML_API ggml_backend_بفر_t ggml_backend_بفر_init(
                   ggml_backend_بفر_type_t buft,
            ساخت ggml_backend_بفر_i      iface,
                   خالی *                     context,
                   size_t                     سائز);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_بفر_copy_tensor(ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_بفر_t ggml_backend_multi_بفر_alloc_بفر(ggml_backend_بفر_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_بفر_is_multi_بفر(ggml_backend_بفر_t بفر);
    GGML_API خالی                  ggml_backend_multi_بفر_سیٹ_usage(ggml_backend_بفر_t بفر, شمار ggml_backend_بفر_usage usage);
    GGML_API خالی                  ggml_backend_meta_بفر_سیٹ_usage (ggml_backend_بفر_t بفر, شمار ggml_backend_بفر_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_بفر_is_meta(ggml_backend_بفر_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_بفر_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t اشاریہ);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    ساخت ggml_backend_گراف_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        خالی (*جوڑو_alloc_dep)(خالی * user_ڈیٹا, ساخت ggml_tensor * tensor, ساخت ggml_tensor * until);
        خالی * user_ڈیٹا;
    };

    ساخت ggml_backend_i {
        ثابت حرف * (*get_name)(ggml_backend_t backend);

        خالی (*آزاد)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        خالی (*سیٹ_tensor_async)   (ggml_backend_t backend,       ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز);
        خالی (*get_tensor_async)   (ggml_backend_t backend, ثابت ساخت ggml_tensor * tensor,       خالی * ڈیٹا, size_t انحراف, size_t سائز);
        خالی (*سیٹ_tensor_2d_async)(ggml_backend_t backend,       ساخت ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t سائز, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا);
        خالی (*get_tensor_2d_async)(ggml_backend_t backend, ثابت ساخت ggml_tensor * tensor,       خالی * ڈیٹا, size_t انحراف, size_t سائز, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ساخت ggml_tensor * src, ساخت ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        خالی (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_گراف_plan_t (*گراف_plan_create) (ggml_backend_t backend, ثابت ساخت ggml_cgraph * cgraph);
        خالی                      (*گراف_plan_free)   (ggml_backend_t backend, ggml_backend_گراف_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        خالی                      (*گراف_plan_update) (ggml_backend_t backend, ggml_backend_گراف_plan_t plan, ثابت ساخت ggml_cgraph * cgraph);
        // compute the graph with the plan
        شمار ggml_حالت          (*گراف_plan_compute)(ggml_backend_t backend, ggml_backend_گراف_plan_t plan);

        // compute graph (always async if supported by the backend)
        شمار ggml_حالت          (*گراف_compute)     (ggml_backend_t backend, ساخت ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        خالی (*event_ریکارڈ)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        خالی (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        خالی                      (*گراف_optimize)    (ggml_backend_t backend, ساخت ggml_cgraph * cgraph, ساخت ggml_backend_گراف_optimize_params * params);
    };

    ساخت ggml_backend {
        ggml_guid_t guid;
        ساخت ggml_backend_i iface;
        ggml_backend_dev_t device;
        خالی * context;
    };

    ساخت ggml_backend_event {
        ساخت ggml_backend_device * device;
        خالی * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    ساخت ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        ثابت حرف * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        ثابت حرف * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        خالی         (*get_حافظہ)(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل);

        // device type
        شمار ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        خالی (*get_props)(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, ثابت حرف * params);

        // preferred buffer type
        ggml_backend_بفر_type_t (*get_بفر_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_بفر_type_t (*get_host_بفر_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_بفر_t (*بفر_from_host_ptr)(ggml_backend_dev_t dev, خالی * ptr, size_t سائز, size_t max_tensor_سائز);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        خالی                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        خالی                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    ساخت ggml_backend_device {
        ساخت ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        خالی * context;
    };

    //
    // Backend (reg)
    //

    ساخت ggml_backend_reg_i {
        ثابت حرف * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_گنتی)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t اشاریہ);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        خالی * (*get_proc_پتہ)(ggml_backend_reg_t reg, ثابت حرف * name);
    };

    ساخت ggml_backend_reg {
        صحیح api_نسخہ; // initialize to GGML_BACKEND_API_VERSION
        ساخت ggml_backend_reg_i iface;
        خالی * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    قسمتعریف ggml_backend_reg_t (*ggml_backend_init_t)(خالی);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    قسمتعریف صحیح                (*ggml_backend_score_t)(خالی);

#اگر_تعریف_شدہ GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            خارجی "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(خالی); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(خالی) {                 \
                واپس reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            خارجی "C" {                                   \
            GGML_BACKEND_API صحیح ggml_backend_score(خالی); \
            }                                              \
            صحیح ggml_backend_score(خالی) {                 \
                واپس score_fn();                         \
            }
#    ورنہ
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(خالی);  \
            ggml_backend_reg_t                  ggml_backend_init(خالی) { \
                واپس reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API صحیح ggml_backend_score(خالی);  \
            صحیح                  ggml_backend_score(خالی) { \
                واپس score_fn();                          \
            }
#    endif
#ورنہ
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#اگر_ختم

#اگر_تعریف_شدہ  __cplusplus
}
#اگر_ختم
