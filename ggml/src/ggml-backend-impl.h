#pragma once

// ggml-backend internal header

#अन्तर्भाव "ggml-backend.h"

#ifdef  __cplusplus
बाह्य "C" {
#endif

    #define GGML_BACKEND_API_VERSION 3

    //
    // Backend buffer type
    //

    संरचना ggml_backend_buffer_type_i {
        अचर वर्ण *          (*get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, माप_प्रकार size);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_buffer_t (*alloc_buffer_n)  (ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors);
        // tensor alignment
        माप_प्रकार                (*get_alignment)   (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        माप_प्रकार                (*get_max_size)    (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        माप_प्रकार                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, अचर संरचना ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        माप_प्रकार                (*get_alloc_size_n)(ggml_backend_buffer_type_t buft, संरचना ggml_tensor ** tensors, पूर्णाङ्क n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
    };

    संरचना ggml_backend_buffer_type {
        संरचना ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        शून्य * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_size_may_expand(गणना ggml_op op);

    //
    // Backend buffer
    //

    संरचना ggml_backend_buffer_i {
        // (optional) free the buffer
        शून्य         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        शून्य *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        गणना ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, संरचना ggml_tensor * tensor);
        // tensor data access
        शून्य         (*memset_tensor)(ggml_backend_buffer_t buffer,       संरचना ggml_tensor * tensor,     uint8_t मान, माप_प्रकार offset, माप_प्रकार size);
        शून्य         (*set_tensor)   (ggml_backend_buffer_t buffer,       संरचना ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size);
        शून्य         (*get_tensor)   (ggml_backend_buffer_t buffer, अचर संरचना ggml_tensor * tensor,       शून्य * data, माप_प्रकार offset, माप_प्रकार size);
        // (optional) 2d data copies
        शून्य         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       संरचना ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data);
        शून्य         (*get_tensor_2d)(ggml_backend_buffer_t buffer, अचर संरचना ggml_tensor * tensor,       शून्य * data, माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, अचर संरचना ggml_tensor * src, संरचना ggml_tensor * dst);
        // clear the entire buffer
        शून्य         (*clear)        (ggml_backend_buffer_t buffer, uint8_t मान);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        शून्य         (*reset)        (ggml_backend_buffer_t buffer);
    };

    संरचना ggml_backend_buffer {
        संरचना ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        शून्य * context;
        माप_प्रकार size;
        गणना ggml_backend_buffer_usage usage;
    };

    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t buft,
            संरचना ggml_backend_buffer_i      iface,
                   शून्य *                     context,
                   माप_प्रकार                     size);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_buffer_copy_tensor(अचर संरचना ggml_tensor * src, संरचना ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, माप_प्रकार n_buffers);
    GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_API शून्य                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, गणना ggml_backend_buffer_usage usage);
    GGML_API शून्य                  ggml_backend_meta_buffer_set_usage (ggml_backend_buffer_t buffer, गणना ggml_backend_buffer_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);

    GGML_API माप_प्रकार         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, माप_प्रकार index);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    संरचना ggml_backend_graph_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        शून्य (*add_alloc_dep)(शून्य * user_data, संरचना ggml_tensor * tensor, संरचना ggml_tensor * until);
        शून्य * user_data;
    };

    संरचना ggml_backend_i {
        अचर वर्ण * (*get_name)(ggml_backend_t backend);

        शून्य (*मुक्त)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        शून्य (*set_tensor_async)   (ggml_backend_t backend,       संरचना ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size);
        शून्य (*get_tensor_async)   (ggml_backend_t backend, अचर संरचना ggml_tensor * tensor,       शून्य * data, माप_प्रकार offset, माप_प्रकार size);
        शून्य (*set_tensor_2d_async)(ggml_backend_t backend,       संरचना ggml_tensor * tensor, अचर शून्य * data, माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data);
        शून्य (*get_tensor_2d_async)(ggml_backend_t backend, अचर संरचना ggml_tensor * tensor,       शून्य * data, माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, अचर संरचना ggml_tensor * src, संरचना ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        शून्य (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, अचर संरचना ggml_cgraph * cgraph);
        शून्य                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        शून्य                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, अचर संरचना ggml_cgraph * cgraph);
        // compute the graph with the plan
        गणना ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        गणना ggml_status          (*graph_compute)     (ggml_backend_t backend, संरचना ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        शून्य (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        शून्य (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        शून्य                      (*graph_optimize)    (ggml_backend_t backend, संरचना ggml_cgraph * cgraph, संरचना ggml_backend_graph_optimize_params * params);
    };

    संरचना ggml_backend {
        ggml_guid_t guid;
        संरचना ggml_backend_i iface;
        ggml_backend_dev_t device;
        शून्य * context;
    };

    संरचना ggml_backend_event {
        संरचना ggml_backend_device * device;
        शून्य * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    संरचना ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        अचर वर्ण * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        अचर वर्ण * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        शून्य         (*get_memory)(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total);

        // device type
        गणना ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        शून्य (*get_props)(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, अचर वर्ण * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, शून्य * ptr, माप_प्रकार size, माप_प्रकार max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, अचर संरचना ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, अचर संरचना ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        शून्य                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        शून्य                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    संरचना ggml_backend_device {
        संरचना ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        शून्य * context;
    };

    //
    // Backend (reg)
    //

    संरचना ggml_backend_reg_i {
        अचर वर्ण * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        माप_प्रकार             (*get_device_गन्ती)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, माप_प्रकार index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        शून्य * (*get_proc_address)(ggml_backend_reg_t reg, अचर वर्ण * name);
    };

    संरचना ggml_backend_reg {
        पूर्णाङ्क api_version; // initialize to GGML_BACKEND_API_VERSION
        संरचना ggml_backend_reg_i iface;
        शून्य * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    प्रकारनिर्देश ggml_backend_reg_t (*ggml_backend_init_t)(शून्य);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    प्रकारनिर्देश पूर्णाङ्क                (*ggml_backend_score_t)(शून्य);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            बाह्य "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(शून्य); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(शून्य) {                 \
                फर्काऊ reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            बाह्य "C" {                                   \
            GGML_BACKEND_API पूर्णाङ्क ggml_backend_score(शून्य); \
            }                                              \
            पूर्णाङ्क ggml_backend_score(शून्य) {                 \
                फर्काऊ score_fn();                         \
            }
#    अन्यथा
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(शून्य);  \
            ggml_backend_reg_t                  ggml_backend_init(शून्य) { \
                फर्काऊ reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API पूर्णाङ्क ggml_backend_score(शून्य);  \
            पूर्णाङ्क                  ggml_backend_score(शून्य) { \
                फर्काऊ score_fn();                          \
            }
#    endif
#अन्यथा
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
