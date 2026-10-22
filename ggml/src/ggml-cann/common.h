/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#که_نه_تعریف_شوی CANN_COMMON_H
#تعریف_کړه CANN_COMMON_H

#ضمن "../ggml-impl.h"
#ضمن "../include/ggml-cann.h"
#ضمن "../include/ggml.h"

#ضمن <acl/acl.h>
#ضمن <unistd.h>

#ضمن <atomic>
#ضمن <condition_variable>
#ضمن <cstdio>
#ضمن <functional>
#ضمن <iostream>
#ضمن <list>
#ضمن <map>
#ضمن <memory>
#ضمن <mutex>
#ضمن <optional>
#ضمن <string>
#ضمن <thread>
#ضمن <vector>

#تعریف_کړه ماتریکس_لیکه_PADDING    512
#تعریف_کړه GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] فراغ ggml_cann_error(ثابت حرف * stmt, ثابت حرف * func, ثابت حرف * file, صحيح کرښه, ثابت حرف * پیغام_2);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#تعریف_کړه ACL_CHECK_GEN(stmt, success, error_fn)                                \
    افعل {                                                                      \
        صحيح err_کوډ = (stmt);                                                \
        إذا (err_کوډ != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } طالما (0);

#تعریف_کړه ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_پیغام)

/**
 * @brief Contains information about CANN devices.
 */
بنية ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_شمېر;

    /**
     * @brief Information about a single CANN device.
     */
    بنية cann_device_info {
        صحيح    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t ټول_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

ثابت ggml_cann_device_info & ggml_cann_info();

فراغ    ggml_cann_سېټ_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(ثابت std::string & name);
bool                       parse_bool(ثابت std::string & ارزښت);
صحيح                        parse_integer(ثابت std::string & ارزښت);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
بنية ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    مجازي ~ggml_cann_pool() = افتراضي;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    مجازي فراغ * alloc(size_t اندازه, size_t * actual_اندازه) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    مجازي فراغ حرر(فراغ * ptr, size_t اندازه) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
بنية ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    فراغ *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_اندازه = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = افتراضي;

    /**
     * @brief Constructor that initializes the memory pool.
     * @param pool Reference to the memory pool.
     */
    explicit ggml_cann_pool_alloc(ggml_cann_pool & pool) : pool(&pool) {}

    /**
     * @brief Constructor that initializes the memory pool and allocates memory.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     */
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t اندازه) : pool(&pool) { alloc(اندازه); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        إذا (ptr != nullptr) {
            pool->حرر(ptr, actual_اندازه);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    فراغ * alloc(size_t اندازه) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(اندازه, &this->actual_اندازه);
        أرجع ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    فراغ * alloc(ggml_cann_pool & pool, size_t اندازه) {
        this->pool = &pool;
        أرجع alloc(اندازه);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    فراغ * get() { أرجع ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & چلوونکی=(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & چلوونکی=(ggml_cann_pool_alloc &&) = delete;
};

#که_تعریف_شوی USE_ACL_ګراف
بنية ggml_ګراف_غوټه_properties {
    // dst tensor
    فراغ *    غوټه_پته;
    ggml_type غوټه_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    فراغ *    src_پته[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op غوټه_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * غوټه) {
        إذا (غوټه->معلومات != this->غوټه_پته && غوټه->op != GGML_OP_VIEW) {
            أرجع false;
        }

        إذا (غوټه->op != this->غوټه_op) {
            أرجع false;
        }

        إذا (غوټه->type != this->غوټه_type) {
            أرجع false;
        }

        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            إذا (غوټه->ne[i] != this->ne[i]) {
                أرجع false;
            }
            إذا (غوټه->nb[i] != this->nb[i]) {
                أرجع false;
            }
        }

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (غوټه->src[i]) {
                إذا (غوټه->src[i]->معلومات != this->src_پته[i] && غوټه->op != GGML_OP_VIEW) {
                    أرجع false;
                }

                إذا (غوټه->src[i]->type != this->src_type[i]) {
                    أرجع false;
                }

                لكل (صحيح d = 0; d < GGML_MAX_DIMS; d++) {
                    إذا (غوټه->src[i]->ne[d] != this->src_ne[i][d]) {
                        أرجع false;
                    }
                    إذا (غوټه->src[i]->nb[d] != this->src_nb[i][d]) {
                        أرجع false;
                    }
                }
            } وإلا {
                إذا (this->src_پته[i] != nullptr) {
                    أرجع false;
                }
            }
        }

        أرجع memcmp(this->op_params, غوټه->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

بنية ggml_cann_ګراف {
    ~ggml_cann_ګراف() {
        إذا (ګراف != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(ګراف));
        }
    }

    aclmdlRI ګراف = nullptr;

    std::ویکټور<ggml_ګراف_غوټه_properties> ggml_ګراف_properties;

    /**
     * @brief Create a new CANN graph from a ggml computation graph.
     *
     * This function creates a new ggml_cann_graph object and fills its node properties
     * (operation type, dimensions, strides, input sources, and operation parameters)
     * based on the current ggml computation graph.
     *
     * Each node in the ggml graph is mapped to a property entry in the new CANN graph:
     * - node address
     * - operation type
     * - shape (ne) and strides (nb)
     * - source tensor addresses
     * - operation parameters
     *
     * @param cgraph The current ggml computation graph.
     * @return Pointer to the newly created ggml_cann_graph object.
     */
    ساكن ggml_cann_ګراف * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_ګراف * new_ګراف = new ggml_cann_ګراف();
        new_ګراف->ggml_ګراف_properties.resize(cgraph->n_nodes);

        لكل (صحيح غوټه_idx = 0; غوټه_idx < cgraph->n_nodes; ++غوټه_idx) {
            ggml_tensor * غوټه = cgraph->nodes[غوټه_idx];
            خپلکاری &        prop = new_ګراف->ggml_ګراف_properties[غوټه_idx];

            prop.غوټه_پته = غوټه->معلومات;
            prop.غوټه_op      = غوټه->op;
            prop.غوټه_type    = غوټه->type;

            std::copy_n(غوټه->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(غوټه->nb, GGML_MAX_DIMS, prop.nb);

            لكل (صحيح src = 0; src < GGML_MAX_SRC; ++src) {
                إذا (غوټه->src[src]) {
                    prop.src_پته[src] = غوټه->src[src]->معلومات;
                    prop.src_type[src]    = غوټه->src[src]->type;
                    std::copy_n(غوټه->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(غوټه->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } وإلا {
                    prop.src_پته[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_شمېر;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            حافظه_نقل(prop.op_params, غوټه->op_params, GGML_MAX_OP_PARAMS);
        }

        أرجع new_ګراف;
    }

    /**
     * @brief Check whether this CANN graph matches the given ggml computation graph.
     *
     * This function compares the number of nodes and each node's properties
     * (operation type, dimensions, strides, inputs, and operation parameters)
     * to determine whether this CANN graph matches the given ggml graph.
     *
     * @param cgraph The current ggml computation graph.
     * @return true if this CANN graph matches the ggml graph; false otherwise.
     */
    bool matches_cgraph(ggml_cgraph * cgraph) {
        إذا (this->ggml_ګراف_properties.اندازه() != static_cast<size_t>(cgraph->n_nodes)) {
            أرجع false;
        }

        لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
            إذا (!this->ggml_ګراف_properties[i].has_matching_properties(cgraph->nodes[i])) {
                أرجع false;
            }
        }

        أرجع true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
بنية ggml_cann_ګراف_lru_cache {
    size_t ظرفیت;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_ګراف *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_ګراف_lru_cache() { ظرفیت = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").ارزښت_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    فراغ ټېل(ggml_cann_ګراف * new_غوټه) {
        إذا (cache_list.اندازه() >= ظرفیت) {
            ggml_cann_ګراف * old = cache_list.شا();
            cache_list.راوباسه_شا();
            delete old;  // free the old graph
        }
        cache_list.ټېل_front(new_غوټه);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    فراغ clear() {
        لكل (خپلکاری ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_ګراف_lru_cache() { clear(); }

    /**
     * @brief Find a cached CANN graph that matches the given ggml graph and move it to front.
     *
     * This function iterates through the cached CANN graphs stored in the LRU cache and
     * compares them against the given ggml computation graph. If a matching graph is found,
     * it is promoted to the front of the LRU cache and returned. Otherwise, the function
     * returns nullptr.
     *
     * @param cgraph The current ggml computation graph.
     * @return true if found; false otherwise.
     */
    bool پیدا_کړه_and_move_منزل_front(ggml_cgraph * cgraph) {
        لكل (خپلکاری & ګراف_ptr : this->cache_list) {
            إذا (ګراف_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(ګراف_ptr);
                cache_list.ټېل_front(ګراف_ptr);
                أرجع true;
            }
        }
        أرجع false;
    }
};
#د_که_پای  // USE_ACL_GRAPH

بنية ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        إذا (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        إذا (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        إذا (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        إذا (position_select_شاخص) {
            ACL_CHECK(aclrtFree(position_select_شاخص));
        }
        إذا (theta_scale_exp_host) {
            حرر(theta_scale_exp_host);
        }
        إذا (position_select_شاخص_host) {
            حرر(position_select_شاخص_host);
        }
        إذا (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_اوږدتیا,
               int64_t position_اوږدتیا,
               عائم   ext_ضریب,
               عائم   theta_scale,
               عائم   freq_scale,
               عائم   attn_ضریب,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_کارول_شوی,
               bool    is_imrope,
               صحيح     sections[4]) {
        أرجع this->theta_scale_اوږدتیا == theta_scale_اوږدتیا && this->position_اوږدتیا == position_اوږدتیا &&
               this->ext_ضریب == ext_ضریب && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_ضریب == attn_ضریب && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_کارول_شوی == mrope_کارول_شوی && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    فراغ سېټ(int64_t theta_scale_اوږدتیا,
             int64_t position_اوږدتیا,
             عائم   ext_ضریب,
             عائم   theta_scale,
             عائم   freq_scale,
             عائم   attn_ضریب,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_کارول_شوی,
             bool    is_imrope,
             صحيح     sections[4]) {
        this->theta_scale_اوږدتیا = theta_scale_اوږدتیا;
        this->position_اوږدتیا    = position_اوږدتیا;
        this->ext_ضریب         = ext_ضریب;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_ضریب        = attn_ضریب;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_کارول_شوی         = mrope_کارول_شوی;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    فراغ *  theta_scale_cache          = nullptr;
    عائم * theta_scale_exp_host       = nullptr;
    صحيح *   position_select_شاخص_host = nullptr;
    فراغ *  position_select_شاخص      = nullptr;
    فراغ *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    فراغ *  sin_cache                  = nullptr;
    فراغ *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_اوږدتیا         = 0;
    int64_t position_اوږدتیا            = 0;
    bool    cached                     = false;
    عائم   ext_ضریب                 = 0.0f;
    عائم   theta_scale                = 0.0f;
    عائم   freq_scale                 = 0.0f;
    عائم   attn_ضریب                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_کارول_شوی                 = false;
    صحيح     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

بنية ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        إذا (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    فراغ *  cache = nullptr;
    int64_t اندازه  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
بنية ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#که_تعریف_شوی USE_ACL_ګراف
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_ګراف_lru_cache ګراف_lru_cache;
    bool                      acl_ګراف_ډېرپېښ = true;
#د_که_پای
    bool                   async_ډېرپېښ;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrt_بهیر streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(صحيح device) : device(device), name("CANN" + std::منزل_string(device)) {
        ggml_cann_سېټ_device(device);
        description = aclrtGetSocName();

#که_تعریف_شوی USE_ACL_ګراف
        acl_ګراف_ډېرپېښ = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").ارزښت_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_ګراف_ډېرپېښ ? "GRAPH" : "EAGER",
                      acl_ګراف_ډېرپېښ ? "acl graph enabled" : "acl graph disabled");
#د_که_پای
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_سېټ_device(device);
        إذا (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        لكل (صحيح i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            إذا (streams[i] != nullptr) {
                ACL_CHECK(aclrt_Destroy_بهیر(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrt_بهیر بهیر(صحيح بهیر) {
        إذا (streams[بهیر] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrt_سېټ_Device(device));
            ACL_CHECK(aclrt_Create_بهیر(&streams[بهیر]));
        }
        أرجع streams[بهیر];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrt_بهیر بهیر() { أرجع بهیر(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    ساكن std::unique_ptr<ggml_cann_pool> new_pool_for_device(صحيح device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        إذا (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        أرجع *mem_pool;
    }
};

#د_که_پای  // CANN_COMMON_H
