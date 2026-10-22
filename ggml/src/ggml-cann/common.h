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

#اگر_غیر_تعریف_شدہ CANN_COMMON_H
#تعریف_کرو CANN_COMMON_H

#شامل "../ggml-impl.h"
#شامل "../include/ggml-cann.h"
#شامل "../include/ggml.h"

#شامل <acl/acl.h>
#شامل <unistd.h>

#شامل <atomic>
#شامل <condition_variable>
#شامل <cstdio>
#شامل <functional>
#شامل <iostream>
#شامل <list>
#شامل <map>
#شامل <memory>
#شامل <mutex>
#شامل <optional>
#شامل <string>
#شامل <thread>
#شامل <vector>

#تعریف_کرو میٹرکس_صف_PADDING    512
#تعریف_کرو GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] خالی ggml_cann_error(ثابت حرف * stmt, ثابت حرف * func, ثابت حرف * file, صحیح سطر, ثابت حرف * پیغام_2);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#تعریف_کرو ACL_CHECK_GEN(stmt, success, error_fn)                                \
    کرو {                                                                      \
        صحیح err_رمز = (stmt);                                                \
        اگر (err_رمز != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } جب (0);

#تعریف_کرو ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_پیغام)

/**
 * @brief Contains information about CANN devices.
 */
ساخت ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_گنتی;

    /**
     * @brief Information about a single CANN device.
     */
    ساخت cann_device_info {
        صحیح    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t کل_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

ثابت ggml_cann_device_info & ggml_cann_info();

خالی    ggml_cann_سیٹ_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(ثابت std::string & name);
bool                       parse_bool(ثابت std::string & قدر);
صحیح                        parse_integer(ثابت std::string & قدر);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
ساخت ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    مجازی ~ggml_cann_pool() = طےشدہ;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    مجازی خالی * alloc(size_t سائز, size_t * actual_سائز) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    مجازی خالی آزاد(خالی * ptr, size_t سائز) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
ساخت ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    خالی *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_سائز = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = طےشدہ;

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
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t سائز) : pool(&pool) { alloc(سائز); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        اگر (ptr != nullptr) {
            pool->آزاد(ptr, actual_سائز);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    خالی * alloc(size_t سائز) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(سائز, &this->actual_سائز);
        واپس ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    خالی * alloc(ggml_cann_pool & pool, size_t سائز) {
        this->pool = &pool;
        واپس alloc(سائز);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    خالی * get() { واپس ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & عامل=(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & عامل=(ggml_cann_pool_alloc &&) = delete;
};

#اگر_تعریف_شدہ USE_ACL_گراف
ساخت ggml_گراف_گرہ_properties {
    // dst tensor
    خالی *    گرہ_پتہ;
    ggml_type گرہ_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    خالی *    src_پتہ[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op گرہ_op;
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
    bool has_matching_properties(ggml_tensor * گرہ) {
        اگر (گرہ->ڈیٹا != this->گرہ_پتہ && گرہ->op != GGML_OP_VIEW) {
            واپس false;
        }

        اگر (گرہ->op != this->گرہ_op) {
            واپس false;
        }

        اگر (گرہ->type != this->گرہ_type) {
            واپس false;
        }

        برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
            اگر (گرہ->ne[i] != this->ne[i]) {
                واپس false;
            }
            اگر (گرہ->nb[i] != this->nb[i]) {
                واپس false;
            }
        }

        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (گرہ->src[i]) {
                اگر (گرہ->src[i]->ڈیٹا != this->src_پتہ[i] && گرہ->op != GGML_OP_VIEW) {
                    واپس false;
                }

                اگر (گرہ->src[i]->type != this->src_type[i]) {
                    واپس false;
                }

                برائے (صحیح d = 0; d < GGML_MAX_DIMS; d++) {
                    اگر (گرہ->src[i]->ne[d] != this->src_ne[i][d]) {
                        واپس false;
                    }
                    اگر (گرہ->src[i]->nb[d] != this->src_nb[i][d]) {
                        واپس false;
                    }
                }
            } ورنہ {
                اگر (this->src_پتہ[i] != nullptr) {
                    واپس false;
                }
            }
        }

        واپس memcmp(this->op_params, گرہ->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

ساخت ggml_cann_گراف {
    ~ggml_cann_گراف() {
        اگر (گراف != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(گراف));
        }
    }

    aclmdlRI گراف = nullptr;

    std::سمتیہ<ggml_گراف_گرہ_properties> ggml_گراف_properties;

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
    ساکن ggml_cann_گراف * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_گراف * new_گراف = new ggml_cann_گراف();
        new_گراف->ggml_گراف_properties.resize(cgraph->n_nodes);

        برائے (صحیح گرہ_idx = 0; گرہ_idx < cgraph->n_nodes; ++گرہ_idx) {
            ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];
            خودکار &        prop = new_گراف->ggml_گراف_properties[گرہ_idx];

            prop.گرہ_پتہ = گرہ->ڈیٹا;
            prop.گرہ_op      = گرہ->op;
            prop.گرہ_type    = گرہ->type;

            std::copy_n(گرہ->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(گرہ->nb, GGML_MAX_DIMS, prop.nb);

            برائے (صحیح src = 0; src < GGML_MAX_SRC; ++src) {
                اگر (گرہ->src[src]) {
                    prop.src_پتہ[src] = گرہ->src[src]->ڈیٹا;
                    prop.src_type[src]    = گرہ->src[src]->type;
                    std::copy_n(گرہ->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(گرہ->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } ورنہ {
                    prop.src_پتہ[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_گنتی;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            حافظہ_نقل(prop.op_params, گرہ->op_params, GGML_MAX_OP_PARAMS);
        }

        واپس new_گراف;
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
        اگر (this->ggml_گراف_properties.سائز() != static_cast<size_t>(cgraph->n_nodes)) {
            واپس false;
        }

        برائے (صحیح i = 0; i < cgraph->n_nodes; ++i) {
            اگر (!this->ggml_گراف_properties[i].has_matching_properties(cgraph->nodes[i])) {
                واپس false;
            }
        }

        واپس true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
ساخت ggml_cann_گراف_lru_cache {
    size_t استعداد;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_گراف *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_گراف_lru_cache() { استعداد = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").قدر_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    خالی ڈالو(ggml_cann_گراف * new_گرہ) {
        اگر (cache_list.سائز() >= استعداد) {
            ggml_cann_گراف * old = cache_list.عقب();
            cache_list.نکالو_عقب();
            delete old;  // free the old graph
        }
        cache_list.ڈالو_front(new_گرہ);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    خالی clear() {
        برائے (خودکار ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_گراف_lru_cache() { clear(); }

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
    bool ڈھونڈو_and_move_مقصد_front(ggml_cgraph * cgraph) {
        برائے (خودکار & گراف_ptr : this->cache_list) {
            اگر (گراف_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(گراف_ptr);
                cache_list.ڈالو_front(گراف_ptr);
                واپس true;
            }
        }
        واپس false;
    }
};
#اگر_ختم  // USE_ACL_GRAPH

ساخت ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        اگر (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        اگر (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        اگر (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        اگر (position_select_اشاریہ) {
            ACL_CHECK(aclrtFree(position_select_اشاریہ));
        }
        اگر (theta_scale_exp_host) {
            آزاد(theta_scale_exp_host);
        }
        اگر (position_select_اشاریہ_host) {
            آزاد(position_select_اشاریہ_host);
        }
        اگر (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_لمبائی,
               int64_t position_لمبائی,
               اعشاری   ext_ضربی_عدد,
               اعشاری   theta_scale,
               اعشاری   freq_scale,
               اعشاری   attn_ضربی_عدد,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_مستعمل,
               bool    is_imrope,
               صحیح     sections[4]) {
        واپس this->theta_scale_لمبائی == theta_scale_لمبائی && this->position_لمبائی == position_لمبائی &&
               this->ext_ضربی_عدد == ext_ضربی_عدد && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_ضربی_عدد == attn_ضربی_عدد && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_مستعمل == mrope_مستعمل && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    خالی سیٹ(int64_t theta_scale_لمبائی,
             int64_t position_لمبائی,
             اعشاری   ext_ضربی_عدد,
             اعشاری   theta_scale,
             اعشاری   freq_scale,
             اعشاری   attn_ضربی_عدد,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_مستعمل,
             bool    is_imrope,
             صحیح     sections[4]) {
        this->theta_scale_لمبائی = theta_scale_لمبائی;
        this->position_لمبائی    = position_لمبائی;
        this->ext_ضربی_عدد         = ext_ضربی_عدد;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_ضربی_عدد        = attn_ضربی_عدد;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_مستعمل         = mrope_مستعمل;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    خالی *  theta_scale_cache          = nullptr;
    اعشاری * theta_scale_exp_host       = nullptr;
    صحیح *   position_select_اشاریہ_host = nullptr;
    خالی *  position_select_اشاریہ      = nullptr;
    خالی *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    خالی *  sin_cache                  = nullptr;
    خالی *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_لمبائی         = 0;
    int64_t position_لمبائی            = 0;
    bool    cached                     = false;
    اعشاری   ext_ضربی_عدد                 = 0.0f;
    اعشاری   theta_scale                = 0.0f;
    اعشاری   freq_scale                 = 0.0f;
    اعشاری   attn_ضربی_عدد                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_مستعمل                 = false;
    صحیح     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

ساخت ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        اگر (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    خالی *  cache = nullptr;
    int64_t سائز  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
ساخت ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#اگر_تعریف_شدہ USE_ACL_گراف
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_گراف_lru_cache گراف_lru_cache;
    bool                      acl_گراف_منوال = true;
#اگر_ختم
    bool                   async_منوال;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrt_دھارا streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(صحیح device) : device(device), name("CANN" + std::مقصد_string(device)) {
        ggml_cann_سیٹ_device(device);
        description = aclrtGetSocName();

#اگر_تعریف_شدہ USE_ACL_گراف
        acl_گراف_منوال = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").قدر_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_گراف_منوال ? "GRAPH" : "EAGER",
                      acl_گراف_منوال ? "acl graph enabled" : "acl graph disabled");
#اگر_ختم
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_سیٹ_device(device);
        اگر (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        برائے (صحیح i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            اگر (streams[i] != nullptr) {
                ACL_CHECK(aclrt_Destroy_دھارا(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrt_دھارا دھارا(صحیح دھارا) {
        اگر (streams[دھارا] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrt_سیٹ_Device(device));
            ACL_CHECK(aclrt_Create_دھارا(&streams[دھارا]));
        }
        واپس streams[دھارا];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrt_دھارا دھارا() { واپس دھارا(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    ساکن std::unique_ptr<ggml_cann_pool> new_pool_for_device(صحیح device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        اگر (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        واپس *mem_pool;
    }
};

#اگر_ختم  // CANN_COMMON_H
