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

#અન્તર્ભાવ "../ggml-impl.h"
#અન્તર્ભાવ "../include/ggml-cann.h"
#અન્તર્ભાવ "../include/ggml.h"

#અન્તર્ભાવ <acl/acl.h>
#અન્તર્ભાવ <unistd.h>

#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <condition_variable>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <functional>
#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <list>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <optional>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <thread>
#અન્તર્ભાવ <vector>

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
[[noreturn]] શૂન્ય ggml_cann_error(અચલ અક્ષર * stmt, અચલ અક્ષર * func, અચલ અક્ષર * file, પૂર્ણાંક લીટી, અચલ અક્ષર * સંદેશ);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    કરો {                                                                      \
        પૂર્ણાંક err_code = (stmt);                                                \
        જો (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } જ્યારે (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_સંદેશ)

/**
 * @brief Contains information about CANN devices.
 */
સંરચના ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_ગણતરી;

    /**
     * @brief Information about a single CANN device.
     */
    સંરચના cann_device_info {
        પૂર્ણાંક    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t total_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

અચલ ggml_cann_device_info & ggml_cann_info();

શૂન્ય    ggml_cann_set_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(અચલ std::string & name);
bool                       parse_bool(અચલ std::string & કિંમત);
પૂર્ણાંક                        parse_integer(અચલ std::string & કિંમત);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
સંરચના ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    આભાસી ~ggml_cann_pool() = મૂળભૂત;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    આભાસી શૂન્ય * alloc(size_t size, size_t * actual_size) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    આભાસી શૂન્ય મુક્ત(શૂન્ય * ptr, size_t size) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
સંરચના ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    શૂન્ય *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_size = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = મૂળભૂત;

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
        જો (ptr != nullptr) {
            pool->મુક્ત(ptr, actual_size);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    શૂન્ય * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(size, &this->actual_size);
        પાછા ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    શૂન્ય * alloc(ggml_cann_pool & pool, size_t size) {
        this->pool = &pool;
        પાછા alloc(size);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    શૂન્ય * get() { પાછા ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(અચલ ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & operator=(અચલ ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & operator=(ggml_cann_pool_alloc &&) = delete;
};

#ifdef USE_ACL_GRAPH
સંરચના ggml_graph_ગાંઠ_properties {
    // dst tensor
    શૂન્ય *    ગાંઠ_address;
    ggml_type ગાંઠ_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    શૂન્ય *    src_address[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op ગાંઠ_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / કદ(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * ગાંઠ) {
        જો (ગાંઠ->data != this->ગાંઠ_address && ગાંઠ->op != GGML_OP_VIEW) {
            પાછા false;
        }

        જો (ગાંઠ->op != this->ગાંઠ_op) {
            પાછા false;
        }

        જો (ગાંઠ->type != this->ગાંઠ_type) {
            પાછા false;
        }

        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
            જો (ગાંઠ->ne[i] != this->ne[i]) {
                પાછા false;
            }
            જો (ગાંઠ->nb[i] != this->nb[i]) {
                પાછા false;
            }
        }

        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            જો (ગાંઠ->src[i]) {
                જો (ગાંઠ->src[i]->data != this->src_address[i] && ગાંઠ->op != GGML_OP_VIEW) {
                    પાછા false;
                }

                જો (ગાંઠ->src[i]->type != this->src_type[i]) {
                    પાછા false;
                }

                માટે (પૂર્ણાંક d = 0; d < GGML_MAX_DIMS; d++) {
                    જો (ગાંઠ->src[i]->ne[d] != this->src_ne[i][d]) {
                        પાછા false;
                    }
                    જો (ગાંઠ->src[i]->nb[d] != this->src_nb[i][d]) {
                        પાછા false;
                    }
                }
            } નહીંતર {
                જો (this->src_address[i] != nullptr) {
                    પાછા false;
                }
            }
        }

        પાછા memcmp(this->op_params, ગાંઠ->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

સંરચના ggml_cann_graph {
    ~ggml_cann_graph() {
        જો (graph != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(graph));
        }
    }

    aclmdlRI graph = nullptr;

    std::vector<ggml_graph_ગાંઠ_properties> ggml_graph_properties;

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
    સ્થિર ggml_cann_graph * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_graph * new_graph = new ggml_cann_graph();
        new_graph->ggml_graph_properties.resize(cgraph->n_nodes);

        માટે (પૂર્ણાંક ગાંઠ_idx = 0; ગાંઠ_idx < cgraph->n_nodes; ++ગાંઠ_idx) {
            ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idx];
            auto &        prop = new_graph->ggml_graph_properties[ગાંઠ_idx];

            prop.ગાંઠ_address = ગાંઠ->data;
            prop.ગાંઠ_op      = ગાંઠ->op;
            prop.ગાંઠ_type    = ગાંઠ->type;

            std::copy_n(ગાંઠ->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(ગાંઠ->nb, GGML_MAX_DIMS, prop.nb);

            માટે (પૂર્ણાંક src = 0; src < GGML_MAX_SRC; ++src) {
                જો (ગાંઠ->src[src]) {
                    prop.src_address[src] = ગાંઠ->src[src]->data;
                    prop.src_type[src]    = ગાંઠ->src[src]->type;
                    std::copy_n(ગાંઠ->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(ગાંઠ->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } નહીંતર {
                    prop.src_address[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_ગણતરી;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            memcpy(prop.op_params, ગાંઠ->op_params, GGML_MAX_OP_PARAMS);
        }

        પાછા new_graph;
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
        જો (this->ggml_graph_properties.size() != static_cast<size_t>(cgraph->n_nodes)) {
            પાછા false;
        }

        માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; ++i) {
            જો (!this->ggml_graph_properties[i].has_matching_properties(cgraph->nodes[i])) {
                પાછા false;
            }
        }

        પાછા true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
સંરચના ggml_cann_graph_lru_cache {
    size_t capacity;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_graph *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_graph_lru_cache() { capacity = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").કિંમત_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    શૂન્ય ધકેલો(ggml_cann_graph * new_ગાંઠ) {
        જો (cache_list.size() >= capacity) {
            ggml_cann_graph * old = cache_list.back();
            cache_list.કાઢો_back();
            delete old;  // free the old graph
        }
        cache_list.ધકેલો_front(new_ગાંઠ);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    શૂન્ય clear() {
        માટે (auto ptr : cache_list) {
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
        માટે (auto & graph_ptr : this->cache_list) {
            જો (graph_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(graph_ptr);
                cache_list.ધકેલો_front(graph_ptr);
                પાછા true;
            }
        }
        પાછા false;
    }
};
#endif  // USE_ACL_GRAPH

સંરચના ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        જો (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        જો (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        જો (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        જો (position_select_index) {
            ACL_CHECK(aclrtFree(position_select_index));
        }
        જો (theta_scale_exp_host) {
            મુક્ત(theta_scale_exp_host);
        }
        જો (position_select_index_host) {
            મુક્ત(position_select_index_host);
        }
        જો (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_length,
               int64_t position_length,
               દશાંશ   ext_factor,
               દશાંશ   theta_scale,
               દશાંશ   freq_scale,
               દશાંશ   attn_factor,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_used,
               bool    is_imrope,
               પૂર્ણાંક     sections[4]) {
        પાછા this->theta_scale_length == theta_scale_length && this->position_length == position_length &&
               this->ext_factor == ext_factor && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_factor == attn_factor && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_used == mrope_used && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    શૂન્ય set(int64_t theta_scale_length,
             int64_t position_length,
             દશાંશ   ext_factor,
             દશાંશ   theta_scale,
             દશાંશ   freq_scale,
             દશાંશ   attn_factor,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_used,
             bool    is_imrope,
             પૂર્ણાંક     sections[4]) {
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
    શૂન્ય *  theta_scale_cache          = nullptr;
    દશાંશ * theta_scale_exp_host       = nullptr;
    પૂર્ણાંક *   position_select_index_host = nullptr;
    શૂન્ય *  position_select_index      = nullptr;
    શૂન્ય *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    શૂન્ય *  sin_cache                  = nullptr;
    શૂન્ય *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_length         = 0;
    int64_t position_length            = 0;
    bool    cached                     = false;
    દશાંશ   ext_factor                 = 0.0f;
    દશાંશ   theta_scale                = 0.0f;
    દશાંશ   freq_scale                 = 0.0f;
    દશાંશ   attn_factor                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_used                 = false;
    પૂર્ણાંક     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

સંરચના ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        જો (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    શૂન્ય *  cache = nullptr;
    int64_t size  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
સંરચના ggml_backend_cann_context {
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
    explicit ggml_backend_cann_context(પૂર્ણાંક device) : device(device), name("CANN" + std::to_string(device)) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();

#ifdef USE_ACL_GRAPH
        acl_graph_mode = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").કિંમત_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_graph_mode ? "GRAPH" : "EAGER",
                      acl_graph_mode ? "acl graph enabled" : "acl graph disabled");
#endif
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        જો (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        માટે (પૂર્ણાંક i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            જો (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrtStream stream(પૂર્ણાંક stream) {
        જો (streams[stream] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrtSetDevice(device));
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        પાછા streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrtStream stream() { પાછા stream(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    સ્થિર std::unique_ptr<ggml_cann_pool> new_pool_for_device(પૂર્ણાંક device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        જો (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        પાછા *mem_pool;
    }
};

#endif  // CANN_COMMON_H
