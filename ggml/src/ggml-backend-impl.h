#आशय once

// ggml-backend internal header

#समावेश "ggml-backend.h"

#अगर_घोषित  __cplusplus
बाह्य "C" {
#पूर्ण_अगर

    #घोषणा GGML_BACKEND_API_VERSION 3

    //
    // Backend buffer type
    //

    काष्ठा ggml_backend_बफर_type_i {
        स्थिर अक्षर *          (*get_name)        (ggml_backend_बफर_type_t buft);
        // allocate a buffer of this type
        ggml_backend_बफर_t (*alloc_बफर)    (ggml_backend_बफर_type_t buft, माप_प्रकार आकार);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_बफर_t (*alloc_बफर_n)  (ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors);
        // tensor alignment
        माप_प्रकार                (*get_alignment)   (ggml_backend_बफर_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        माप_प्रकार                (*get_max_आकार)    (ggml_backend_बफर_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        माप_प्रकार                (*get_alloc_आकार)  (ggml_backend_बफर_type_t buft, स्थिर काष्ठा ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        माप_प्रकार                (*get_alloc_आकार_n)(ggml_backend_बफर_type_t buft, काष्ठा ggml_tensor ** tensors, पूर्णांक n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_बफर_type_t buft);
    };

    काष्ठा ggml_backend_बफर_type {
        काष्ठा ggml_backend_बफर_type_i  iface;
        ggml_backend_dev_t device;
        व्योम * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_आकार_may_expand(क्रमागत ggml_op op);

    //
    // Backend buffer
    //

    काष्ठा ggml_backend_बफर_i {
        // (optional) free the buffer
        व्योम         (*free_बफर)  (ggml_backend_बफर_t बफर);
        // base address of the buffer
        व्योम *       (*get_base)     (ggml_backend_बफर_t बफर);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        क्रमागत ggml_status (*init_tensor)(ggml_backend_बफर_t बफर, काष्ठा ggml_tensor * tensor);
        // tensor data access
        व्योम         (*memset_tensor)(ggml_backend_बफर_t बफर,       काष्ठा ggml_tensor * tensor,     uint8_t मूल्य, माप_प्रकार offset, माप_प्रकार आकार);
        व्योम         (*set_tensor)   (ggml_backend_बफर_t बफर,       काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
        व्योम         (*get_tensor)   (ggml_backend_बफर_t बफर, स्थिर काष्ठा ggml_tensor * tensor,       व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
        // (optional) 2d data copies
        व्योम         (*set_tensor_2d)(ggml_backend_बफर_t बफर,       काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य);
        व्योम         (*get_tensor_2d)(ggml_backend_बफर_t बफर, स्थिर काष्ठा ggml_tensor * tensor,       व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_बफर_t बफर, स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst);
        // clear the entire buffer
        व्योम         (*clear)        (ggml_backend_बफर_t बफर, uint8_t मूल्य);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        व्योम         (*reset)        (ggml_backend_बफर_t बफर);
    };

    काष्ठा ggml_backend_बफर {
        काष्ठा ggml_backend_बफर_i  iface;
        ggml_backend_बफर_type_t    buft;
        व्योम * context;
        माप_प्रकार आकार;
        क्रमागत ggml_backend_बफर_usage usage;
    };

    GGML_API ggml_backend_बफर_t ggml_backend_बफर_init(
                   ggml_backend_बफर_type_t buft,
            काष्ठा ggml_backend_बफर_i      iface,
                   व्योम *                     context,
                   माप_प्रकार                     आकार);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_बफर_copy_tensor(स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_बफर_t ggml_backend_multi_बफर_alloc_बफर(ggml_backend_बफर_t * buffers, माप_प्रकार n_buffers);
    GGML_API bool                  ggml_backend_बफर_is_multi_बफर(ggml_backend_बफर_t बफर);
    GGML_API व्योम                  ggml_backend_multi_बफर_set_usage(ggml_backend_बफर_t बफर, क्रमागत ggml_backend_बफर_usage usage);
    GGML_API व्योम                  ggml_backend_meta_बफर_set_usage (ggml_backend_बफर_t बफर, क्रमागत ggml_backend_बफर_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_बफर_is_meta(ggml_backend_बफर_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_बफर_type_t buft);

    GGML_API माप_प्रकार         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, माप_प्रकार सूचकांक);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    काष्ठा ggml_backend_graph_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        व्योम (*add_alloc_dep)(व्योम * user_तथ्य, काष्ठा ggml_tensor * tensor, काष्ठा ggml_tensor * until);
        व्योम * user_तथ्य;
    };

    काष्ठा ggml_backend_i {
        स्थिर अक्षर * (*get_name)(ggml_backend_t backend);

        व्योम (*मुक्त)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        व्योम (*set_tensor_async)   (ggml_backend_t backend,       काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
        व्योम (*get_tensor_async)   (ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * tensor,       व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);
        व्योम (*set_tensor_2d_async)(ggml_backend_t backend,       काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य);
        व्योम (*get_tensor_2d_async)(ggml_backend_t backend, स्थिर काष्ठा ggml_tensor * tensor,       व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, स्थिर काष्ठा ggml_tensor * src, काष्ठा ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        व्योम (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, स्थिर काष्ठा ggml_cgraph * cgraph);
        व्योम                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        व्योम                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, स्थिर काष्ठा ggml_cgraph * cgraph);
        // compute the graph with the plan
        क्रमागत ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        क्रमागत ggml_status          (*graph_compute)     (ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        व्योम (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        व्योम (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        व्योम                      (*graph_optimize)    (ggml_backend_t backend, काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_backend_graph_optimize_params * params);
    };

    काष्ठा ggml_backend {
        ggml_guid_t guid;
        काष्ठा ggml_backend_i iface;
        ggml_backend_dev_t device;
        व्योम * context;
    };

    काष्ठा ggml_backend_event {
        काष्ठा ggml_backend_device * device;
        व्योम * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    काष्ठा ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        स्थिर अक्षर * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        स्थिर अक्षर * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        व्योम         (*get_memory)(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग);

        // device type
        क्रमागत ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        व्योम (*get_props)(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, स्थिर अक्षर * params);

        // preferred buffer type
        ggml_backend_बफर_type_t (*get_बफर_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_बफर_type_t (*get_host_बफर_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_बफर_t (*बफर_from_host_ptr)(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार आकार, माप_प्रकार max_tensor_आकार);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        व्योम                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        व्योम                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    काष्ठा ggml_backend_device {
        काष्ठा ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        व्योम * context;
    };

    //
    // Backend (reg)
    //

    काष्ठा ggml_backend_reg_i {
        स्थिर अक्षर * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        माप_प्रकार             (*get_device_गिनती)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, माप_प्रकार सूचकांक);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        व्योम * (*get_proc_address)(ggml_backend_reg_t reg, स्थिर अक्षर * name);
    };

    काष्ठा ggml_backend_reg {
        पूर्णांक api_version; // initialize to GGML_BACKEND_API_VERSION
        काष्ठा ggml_backend_reg_i iface;
        व्योम * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    प्रकार ggml_backend_reg_t (*ggml_backend_init_t)(व्योम);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    प्रकार पूर्णांक                (*ggml_backend_score_t)(व्योम);

#अगर_घोषित GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            बाह्य "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(व्योम); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(व्योम) {                 \
                वापस reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            बाह्य "C" {                                   \
            GGML_BACKEND_API पूर्णांक ggml_backend_score(व्योम); \
            }                                              \
            पूर्णांक ggml_backend_score(व्योम) {                 \
                वापस score_fn();                         \
            }
#    अन्यथा
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(व्योम);  \
            ggml_backend_reg_t                  ggml_backend_init(व्योम) { \
                वापस reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API पूर्णांक ggml_backend_score(व्योम);  \
            पूर्णांक                  ggml_backend_score(व्योम) { \
                वापस score_fn();                          \
            }
#    endif
#अन्यथा
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#पूर्ण_अगर

#अगर_घोषित  __cplusplus
}
#पूर्ण_अगर
