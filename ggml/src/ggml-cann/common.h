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

#ifndef CANN_COMMON_H
#define CANN_COMMON_H

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

#define MATRIX_ROW_PADDING    512
#define GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] تهی ggml_cann_error(ثابت نویسه * stmt, ثابت نویسه * func, ثابت نویسه * file, صحیح سطر, ثابت نویسه * پیام);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    انجامبده {                                                                      \
        صحیح err_code = (stmt);                                                \
        اگر (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } وقتی (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_پیام)

/**
 * @brief Contains information about CANN devices.
 */
ساختار ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_شمار;

    /**
     * @brief Information about a single CANN device.
     */
    ساختار cann_device_info {
        صحیح    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t total_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

ثابت ggml_cann_device_info & ggml_cann_info();

تهی    ggml_cann_set_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(ثابت std::string & name);
bool                       parse_bool(ثابت std::string & مقدار);
صحیح                        parse_integer(ثابت std::string & مقدار);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
ساختار ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    مجازی ~ggml_cann_pool() = پیشفرض;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    مجازی تهی * alloc(size_t size, size_t * actual_size) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    مجازی تهی آزاد(تهی * ptr, size_t size) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
ساختار ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    تهی *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_size = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = پیشفرض;

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
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t size) : pool(&pool) { alloc(size); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        اگر (ptr != nullptr) {
            pool->آزاد(ptr, actual_size);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    تهی * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(size, &this->actual_size);
        بازگشت ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    تهی * alloc(ggml_cann_pool & pool, size_t size) {
        this->pool = &pool;
        بازگشت alloc(size);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    تهی * get() { بازگشت ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & operator=(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & operator=(ggml_cann_pool_alloc &&) = delete;
};

#ifdef USE_ACL_GRAPH
ساختار ggml_graph_گره_properties {
    // dst tensor
    تهی *    گره_address;
    ggml_type گره_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    تهی *    src_address[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op گره_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / اندازه(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * گره) {
        اگر (گره->data != this->گره_address && گره->op != GGML_OP_VIEW) {
            بازگشت false;
        }

        اگر (گره->op != this->گره_op) {
            بازگشت false;
        }

        اگر (گره->type != this->گره_type) {
            بازگشت false;
        }

        برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
            اگر (گره->ne[i] != this->ne[i]) {
                بازگشت false;
            }
            اگر (گره->nb[i] != this->nb[i]) {
                بازگشت false;
            }
        }

        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (گره->src[i]) {
                اگر (گره->src[i]->data != this->src_address[i] && گره->op != GGML_OP_VIEW) {
                    بازگشت false;
                }

                اگر (گره->src[i]->type != this->src_type[i]) {
                    بازگشت false;
                }

                برای (صحیح d = 0; d < GGML_MAX_DIMS; d++) {
                    اگر (گره->src[i]->ne[d] != this->src_ne[i][d]) {
                        بازگشت false;
                    }
                    اگر (گره->src[i]->nb[d] != this->src_nb[i][d]) {
                        بازگشت false;
                    }
                }
            } وگرنه {
                اگر (this->src_address[i] != nullptr) {
                    بازگشت false;
                }
            }
        }

        بازگشت memcmp(this->op_params, گره->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

ساختار ggml_cann_graph {
    ~ggml_cann_graph() {
        اگر (graph != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(graph));
        }
    }

    aclmdlRI graph = nullptr;

    std::vector<ggml_graph_گره_properties> ggml_graph_properties;

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
    ایستا ggml_cann_graph * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_graph * new_graph = new ggml_cann_graph();
        new_graph->ggml_graph_properties.resize(cgraph->n_nodes);

        برای (صحیح گره_idx = 0; گره_idx < cgraph->n_nodes; ++گره_idx) {
            ggml_tensor * گره = cgraph->nodes[گره_idx];
            auto &        prop = new_graph->ggml_graph_properties[گره_idx];

            prop.گره_address = گره->data;
            prop.گره_op      = گره->op;
            prop.گره_type    = گره->type;

            std::copy_n(گره->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(گره->nb, GGML_MAX_DIMS, prop.nb);

            برای (صحیح src = 0; src < GGML_MAX_SRC; ++src) {
                اگر (گره->src[src]) {
                    prop.src_address[src] = گره->src[src]->data;
                    prop.src_type[src]    = گره->src[src]->type;
                    std::copy_n(گره->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(گره->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } وگرنه {
                    prop.src_address[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_شمار;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            memcpy(prop.op_params, گره->op_params, GGML_MAX_OP_PARAMS);
        }

        بازگشت new_graph;
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
        اگر (this->ggml_graph_properties.size() != static_cast<size_t>(cgraph->n_nodes)) {
            بازگشت false;
        }

        برای (صحیح i = 0; i < cgraph->n_nodes; ++i) {
            اگر (!this->ggml_graph_properties[i].has_matching_properties(cgraph->nodes[i])) {
                بازگشت false;
            }
        }

        بازگشت true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
ساختار ggml_cann_graph_lru_cache {
    size_t capacity;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_graph *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_graph_lru_cache() { capacity = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").مقدار_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    تهی افزودن(ggml_cann_graph * new_گره) {
        اگر (cache_list.size() >= capacity) {
            ggml_cann_graph * old = cache_list.back();
            cache_list.برداشت_back();
            delete old;  // free the old graph
        }
        cache_list.افزودن_front(new_گره);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    تهی clear() {
        برای (auto ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_graph_lru_cache() { clear(); }

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
    bool find_and_move_to_front(ggml_cgraph * cgraph) {
        برای (auto & graph_ptr : this->cache_list) {
            اگر (graph_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(graph_ptr);
                cache_list.افزودن_front(graph_ptr);
                بازگشت true;
            }
        }
        بازگشت false;
    }
};
#endif  // USE_ACL_GRAPH

ساختار ggml_cann_rope_cache {
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
        اگر (position_select_index) {
            ACL_CHECK(aclrtFree(position_select_index));
        }
        اگر (theta_scale_exp_host) {
            آزاد(theta_scale_exp_host);
        }
        اگر (position_select_index_host) {
            آزاد(position_select_index_host);
        }
        اگر (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_length,
               int64_t position_length,
               اعشاری   ext_factor,
               اعشاری   theta_scale,
               اعشاری   freq_scale,
               اعشاری   attn_factor,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_used,
               bool    is_imrope,
               صحیح     sections[4]) {
        بازگشت this->theta_scale_length == theta_scale_length && this->position_length == position_length &&
               this->ext_factor == ext_factor && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_factor == attn_factor && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_used == mrope_used && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    تهی set(int64_t theta_scale_length,
             int64_t position_length,
             اعشاری   ext_factor,
             اعشاری   theta_scale,
             اعشاری   freq_scale,
             اعشاری   attn_factor,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_used,
             bool    is_imrope,
             صحیح     sections[4]) {
        this->theta_scale_length = theta_scale_length;
        this->position_length    = position_length;
        this->ext_factor         = ext_factor;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_factor        = attn_factor;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_used         = mrope_used;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    تهی *  theta_scale_cache          = nullptr;
    اعشاری * theta_scale_exp_host       = nullptr;
    صحیح *   position_select_index_host = nullptr;
    تهی *  position_select_index      = nullptr;
    تهی *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    تهی *  sin_cache                  = nullptr;
    تهی *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_length         = 0;
    int64_t position_length            = 0;
    bool    cached                     = false;
    اعشاری   ext_factor                 = 0.0f;
    اعشاری   theta_scale                = 0.0f;
    اعشاری   freq_scale                 = 0.0f;
    اعشاری   attn_factor                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_used                 = false;
    صحیح     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

ساختار ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        اگر (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    تهی *  cache = nullptr;
    int64_t size  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
ساختار ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#ifdef USE_ACL_GRAPH
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_graph_lru_cache graph_lru_cache;
    bool                      acl_graph_mode = true;
#endif
    bool                   async_mode;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(صحیح device) : device(device), name("CANN" + std::to_string(device)) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();

#ifdef USE_ACL_GRAPH
        acl_graph_mode = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").مقدار_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_graph_mode ? "GRAPH" : "EAGER",
                      acl_graph_mode ? "acl graph enabled" : "acl graph disabled");
#endif
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        اگر (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        برای (صحیح i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            اگر (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrtStream stream(صحیح stream) {
        اگر (streams[stream] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrtSetDevice(device));
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        بازگشت streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrtStream stream() { بازگشت stream(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    ایستا std::unique_ptr<ggml_cann_pool> new_pool_for_device(صحیح device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        اگر (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        بازگشت *mem_pool;
    }
};

#endif  // CANN_COMMON_H
