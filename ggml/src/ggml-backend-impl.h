#pragma once

// ggml-backend internal header

#অন্তর্ভাৱ "ggml-backend.h"

#ifdef  __cplusplus
বাহ্য "C" {
#endif

    #define GGML_BACKEND_API_VERSION 3

    //
    // Backend buffer type
    //

    সংরচনা ggml_backend_buffer_type_i {
        ধ্রুৱ ৱর্ণ *          (*get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_buffer_t (*alloc_buffer_n)  (ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)    (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, ধ্রুৱ সংরচনা ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_size_n)(ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
    };

    সংরচনা ggml_backend_buffer_type {
        সংরচনা ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        শূন্য * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_size_may_expand(গণনা ggml_op op);

    //
    // Backend buffer
    //

    সংরচনা ggml_backend_buffer_i {
        // (optional) free the buffer
        শূন্য         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        শূন্য *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        গণনা ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, সংরচনা ggml_tensor * tensor);
        // tensor data access
        শূন্য         (*memset_tensor)(ggml_backend_buffer_t buffer,       সংরচনা ggml_tensor * tensor,     uint8_t মান, size_t offset, size_t size);
        শূন্য         (*set_tensor)   (ggml_backend_buffer_t buffer,       সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size);
        শূন্য         (*get_tensor)   (ggml_backend_buffer_t buffer, ধ্রুৱ সংরচনা ggml_tensor * tensor,       শূন্য * data, size_t offset, size_t size);
        // (optional) 2d data copies
        শূন্য         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        শূন্য         (*get_tensor_2d)(ggml_backend_buffer_t buffer, ধ্রুৱ সংরচনা ggml_tensor * tensor,       শূন্য * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst);
        // clear the entire buffer
        শূন্য         (*clear)        (ggml_backend_buffer_t buffer, uint8_t মান);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        শূন্য         (*reset)        (ggml_backend_buffer_t buffer);
    };

    সংরচনা ggml_backend_buffer {
        সংরচনা ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        শূন্য * context;
        size_t size;
        গণনা ggml_backend_buffer_usage usage;
    };

    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t buft,
            সংরচনা ggml_backend_buffer_i      iface,
                   শূন্য *                     context,
                   size_t                     size);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_buffer_copy_tensor(ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_API শূন্য                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, গণনা ggml_backend_buffer_usage usage);
    GGML_API শূন্য                  ggml_backend_meta_buffer_set_usage (ggml_backend_buffer_t buffer, গণনা ggml_backend_buffer_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    সংরচনা ggml_backend_graph_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        শূন্য (*add_alloc_dep)(শূন্য * user_data, সংরচনা ggml_tensor * tensor, সংরচনা ggml_tensor * until);
        শূন্য * user_data;
    };

    সংরচনা ggml_backend_i {
        ধ্রুৱ ৱর্ণ * (*get_name)(ggml_backend_t backend);

        শূন্য (*মুক্ত)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        শূন্য (*set_tensor_async)   (ggml_backend_t backend,       সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size);
        শূন্য (*get_tensor_async)   (ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_tensor * tensor,       শূন্য * data, size_t offset, size_t size);
        শূন্য (*set_tensor_2d_async)(ggml_backend_t backend,       সংরচনা ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        শূন্য (*get_tensor_2d_async)(ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_tensor * tensor,       শূন্য * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুৱ সংরচনা ggml_tensor * src, সংরচনা ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        শূন্য (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, ধ্রুৱ সংরচনা ggml_cgraph * cgraph);
        শূন্য                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        শূন্য                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, ধ্রুৱ সংরচনা ggml_cgraph * cgraph);
        // compute the graph with the plan
        গণনা ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        গণনা ggml_status          (*graph_compute)     (ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        শূন্য (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        শূন্য (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        শূন্য                      (*graph_optimize)    (ggml_backend_t backend, সংরচনা ggml_cgraph * cgraph, সংরচনা ggml_backend_graph_optimize_params * params);
    };

    সংরচনা ggml_backend {
        ggml_guid_t guid;
        সংরচনা ggml_backend_i iface;
        ggml_backend_dev_t device;
        শূন্য * context;
    };

    সংরচনা ggml_backend_event {
        সংরচনা ggml_backend_device * device;
        শূন্য * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    সংরচনা ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        ধ্রুৱ ৱর্ণ * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        ধ্রুৱ ৱর্ণ * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        শূন্য         (*get_memory)(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total);

        // device type
        গণনা ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        শূন্য (*get_props)(ggml_backend_dev_t dev, সংরচনা ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, ধ্রুৱ ৱর্ণ * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, শূন্য * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, ধ্রুৱ সংরচনা ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, ধ্রুৱ সংরচনা ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        শূন্য                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        শূন্য                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    সংরচনা ggml_backend_device {
        সংরচনা ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        শূন্য * context;
    };

    //
    // Backend (reg)
    //

    সংরচনা ggml_backend_reg_i {
        ধ্রুৱ ৱর্ণ * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_গুনতি)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        শূন্য * (*get_proc_address)(ggml_backend_reg_t reg, ধ্রুৱ ৱর্ণ * name);
    };

    সংরচনা ggml_backend_reg {
        পূর্ণাঙ্ক api_version; // initialize to GGML_BACKEND_API_VERSION
        সংরচনা ggml_backend_reg_i iface;
        শূন্য * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    প্রকারনির্দেশ ggml_backend_reg_t (*ggml_backend_init_t)(শূন্য);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    প্রকারনির্দেশ পূর্ণাঙ্ক                (*ggml_backend_score_t)(শূন্য);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            বাহ্য "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(শূন্য); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(শূন্য) {                 \
                ঘূৰাই reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            বাহ্য "C" {                                   \
            GGML_BACKEND_API পূর্ণাঙ্ক ggml_backend_score(শূন্য); \
            }                                              \
            পূর্ণাঙ্ক ggml_backend_score(শূন্য) {                 \
                ঘূৰাই score_fn();                         \
            }
#    অন্যথা
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(শূন্য);  \
            ggml_backend_reg_t                  ggml_backend_init(শূন্য) { \
                ঘূৰাই reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API পূর্ণাঙ্ক ggml_backend_score(শূন্য);  \
            পূর্ণাঙ্ক                  ggml_backend_score(শূন্য) { \
                ঘূৰাই score_fn();                          \
            }
#    endif
#অন্যথা
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
