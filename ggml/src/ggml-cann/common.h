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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ CANN_COMMON_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CANN_COMMON_H

#ਅਨ੍ਤਰ੍ਭਾਵ "../ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../include/ggml-cann.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../include/ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <acl/acl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <condition_variable>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <functional>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <list>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING    512
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] ਖਾਲੀ ggml_cann_error(ਅਚਲ ਅੱਖਰ * stmt, ਅਚਲ ਅੱਖਰ * func, ਅਚਲ ਅੱਖਰ * file, ਪੂਰਨ ਸਤਰ, ਅਚਲ ਅੱਖਰ * ਸੁਨੇਹਾ_2);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ACL_CHECK_GEN(stmt, success, error_fn)                                \
    ਕਰੋ {                                                                      \
        ਪੂਰਨ err_ਕੋਡ = (stmt);                                                \
        ਜੇ (err_ਕੋਡ != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } ਜਦੋਂ (0);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_ਸੁਨੇਹਾ)

/**
 * @brief Contains information about CANN devices.
 */
ਬਣਤਰ ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_ਗਿਣਤੀ;

    /**
     * @brief Information about a single CANN device.
     */
    ਬਣਤਰ cann_device_info {
        ਪੂਰਨ    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t ਕੁੱਲ_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

ਅਚਲ ggml_cann_device_info & ggml_cann_info();

ਖਾਲੀ    ggml_cann_ਸੈੱਟ_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(ਅਚਲ std::string & name);
bool                       parse_bool(ਅਚਲ std::string & ਮੁੱਲ);
ਪੂਰਨ                        parse_integer(ਅਚਲ std::string & ਮੁੱਲ);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
ਬਣਤਰ ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    ਆਭਾਸੀ ~ggml_cann_pool() = ਮੂਲ;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    ਆਭਾਸੀ ਖਾਲੀ * alloc(size_t ਮਾਪ, size_t * actual_ਮਾਪ) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    ਆਭਾਸੀ ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t ਮਾਪ) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
ਬਣਤਰ ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    ਖਾਲੀ *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_ਮਾਪ = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = ਮੂਲ;

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
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t ਮਾਪ) : pool(&pool) { alloc(ਮਾਪ); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        ਜੇ (ptr != nullptr) {
            pool->ਮੁਕਤ(ptr, actual_ਮਾਪ);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    ਖਾਲੀ * alloc(size_t ਮਾਪ) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(ਮਾਪ, &this->actual_ਮਾਪ);
        ਵਾਪਸ ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    ਖਾਲੀ * alloc(ggml_cann_pool & pool, size_t ਮਾਪ) {
        this->pool = &pool;
        ਵਾਪਸ alloc(ਮਾਪ);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    ਖਾਲੀ * get() { ਵਾਪਸ ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(ਅਚਲ ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & ਸੰਚਾਲਕ=(ਅਚਲ ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & ਸੰਚਾਲਕ=(ggml_cann_pool_alloc &&) = delete;
};

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ USE_ACL_ਗ੍ਰਾਫ
ਬਣਤਰ ggml_ਗ੍ਰਾਫ_ਗੰਢ_properties {
    // dst tensor
    ਖਾਲੀ *    ਗੰਢ_ਪਤਾ;
    ggml_type ਗੰਢ_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    ਖਾਲੀ *    src_ਪਤਾ[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op ਗੰਢ_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * ਗੰਢ) {
        ਜੇ (ਗੰਢ->ਅੰਕੜੇ != this->ਗੰਢ_ਪਤਾ && ਗੰਢ->op != GGML_OP_VIEW) {
            ਵਾਪਸ false;
        }

        ਜੇ (ਗੰਢ->op != this->ਗੰਢ_op) {
            ਵਾਪਸ false;
        }

        ਜੇ (ਗੰਢ->type != this->ਗੰਢ_type) {
            ਵਾਪਸ false;
        }

        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            ਜੇ (ਗੰਢ->ne[i] != this->ne[i]) {
                ਵਾਪਸ false;
            }
            ਜੇ (ਗੰਢ->nb[i] != this->nb[i]) {
                ਵਾਪਸ false;
            }
        }

        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਜੇ (ਗੰਢ->src[i]) {
                ਜੇ (ਗੰਢ->src[i]->ਅੰਕੜੇ != this->src_ਪਤਾ[i] && ਗੰਢ->op != GGML_OP_VIEW) {
                    ਵਾਪਸ false;
                }

                ਜੇ (ਗੰਢ->src[i]->type != this->src_type[i]) {
                    ਵਾਪਸ false;
                }

                ਲਈ (ਪੂਰਨ d = 0; d < GGML_MAX_DIMS; d++) {
                    ਜੇ (ਗੰਢ->src[i]->ne[d] != this->src_ne[i][d]) {
                        ਵਾਪਸ false;
                    }
                    ਜੇ (ਗੰਢ->src[i]->nb[d] != this->src_nb[i][d]) {
                        ਵਾਪਸ false;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (this->src_ਪਤਾ[i] != nullptr) {
                    ਵਾਪਸ false;
                }
            }
        }

        ਵਾਪਸ memcmp(this->op_params, ਗੰਢ->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

ਬਣਤਰ ggml_cann_ਗ੍ਰਾਫ {
    ~ggml_cann_ਗ੍ਰਾਫ() {
        ਜੇ (ਗ੍ਰਾਫ != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(ਗ੍ਰਾਫ));
        }
    }

    aclmdlRI ਗ੍ਰਾਫ = nullptr;

    std::ਸਦਿਸ਼<ggml_ਗ੍ਰਾਫ_ਗੰਢ_properties> ggml_ਗ੍ਰਾਫ_properties;

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
    ਸਥਿਰ ggml_cann_ਗ੍ਰਾਫ * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_ਗ੍ਰਾਫ * new_ਗ੍ਰਾਫ = new ggml_cann_ਗ੍ਰਾਫ();
        new_ਗ੍ਰਾਫ->ggml_ਗ੍ਰਾਫ_properties.resize(cgraph->n_nodes);

        ਲਈ (ਪੂਰਨ ਗੰਢ_idx = 0; ਗੰਢ_idx < cgraph->n_nodes; ++ਗੰਢ_idx) {
            ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
            ਸਵੈਚਾਲਿਤ &        prop = new_ਗ੍ਰਾਫ->ggml_ਗ੍ਰਾਫ_properties[ਗੰਢ_idx];

            prop.ਗੰਢ_ਪਤਾ = ਗੰਢ->ਅੰਕੜੇ;
            prop.ਗੰਢ_op      = ਗੰਢ->op;
            prop.ਗੰਢ_type    = ਗੰਢ->type;

            std::copy_n(ਗੰਢ->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(ਗੰਢ->nb, GGML_MAX_DIMS, prop.nb);

            ਲਈ (ਪੂਰਨ src = 0; src < GGML_MAX_SRC; ++src) {
                ਜੇ (ਗੰਢ->src[src]) {
                    prop.src_ਪਤਾ[src] = ਗੰਢ->src[src]->ਅੰਕੜੇ;
                    prop.src_type[src]    = ਗੰਢ->src[src]->type;
                    std::copy_n(ਗੰਢ->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(ਗੰਢ->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } ਨਹੀਂਤਾਂ {
                    prop.src_ਪਤਾ[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_ਗਿਣਤੀ;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(prop.op_params, ਗੰਢ->op_params, GGML_MAX_OP_PARAMS);
        }

        ਵਾਪਸ new_ਗ੍ਰਾਫ;
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
        ਜੇ (this->ggml_ਗ੍ਰਾਫ_properties.ਮਾਪ() != static_cast<size_t>(cgraph->n_nodes)) {
            ਵਾਪਸ false;
        }

        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; ++i) {
            ਜੇ (!this->ggml_ਗ੍ਰਾਫ_properties[i].has_matching_properties(cgraph->nodes[i])) {
                ਵਾਪਸ false;
            }
        }

        ਵਾਪਸ true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
ਬਣਤਰ ggml_cann_ਗ੍ਰਾਫ_lru_cache {
    size_t ਸਮਰੱਥਾ;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_ਗ੍ਰਾਫ *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_ਗ੍ਰਾਫ_lru_cache() { ਸਮਰੱਥਾ = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").ਮੁੱਲ_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    ਖਾਲੀ ਧੱਕੋ(ggml_cann_ਗ੍ਰਾਫ * new_ਗੰਢ) {
        ਜੇ (cache_list.ਮਾਪ() >= ਸਮਰੱਥਾ) {
            ggml_cann_ਗ੍ਰਾਫ * old = cache_list.ਪਿੱਠ();
            cache_list.ਕੱਢੋ_ਪਿੱਠ();
            delete old;  // free the old graph
        }
        cache_list.ਧੱਕੋ_front(new_ਗੰਢ);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    ਖਾਲੀ clear() {
        ਲਈ (ਸਵੈਚਾਲਿਤ ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_ਗ੍ਰਾਫ_lru_cache() { clear(); }

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
    bool ਲੱਭੋ_and_move_ਮੰਜ਼ਿਲ_front(ggml_cgraph * cgraph) {
        ਲਈ (ਸਵੈਚਾਲਿਤ & ਗ੍ਰਾਫ_ptr : this->cache_list) {
            ਜੇ (ਗ੍ਰਾਫ_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(ਗ੍ਰਾਫ_ptr);
                cache_list.ਧੱਕੋ_front(ਗ੍ਰਾਫ_ptr);
                ਵਾਪਸ true;
            }
        }
        ਵਾਪਸ false;
    }
};
#ਜੇ_ਅੰਤ  // USE_ACL_GRAPH

ਬਣਤਰ ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        ਜੇ (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        ਜੇ (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        ਜੇ (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        ਜੇ (position_select_ਸੂਚਕਾਂਕ) {
            ACL_CHECK(aclrtFree(position_select_ਸੂਚਕਾਂਕ));
        }
        ਜੇ (theta_scale_exp_host) {
            ਮੁਕਤ(theta_scale_exp_host);
        }
        ਜੇ (position_select_ਸੂਚਕਾਂਕ_host) {
            ਮੁਕਤ(position_select_ਸੂਚਕਾਂਕ_host);
        }
        ਜੇ (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_ਲੰਮਾਈ,
               int64_t position_ਲੰਮਾਈ,
               ਦਸ਼ਮਲਵ   ext_ਗੁਣਕ,
               ਦਸ਼ਮਲਵ   theta_scale,
               ਦਸ਼ਮਲਵ   freq_scale,
               ਦਸ਼ਮਲਵ   attn_ਗੁਣਕ,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_ਵਰਤਿਆ,
               bool    is_imrope,
               ਪੂਰਨ     sections[4]) {
        ਵਾਪਸ this->theta_scale_ਲੰਮਾਈ == theta_scale_ਲੰਮਾਈ && this->position_ਲੰਮਾਈ == position_ਲੰਮਾਈ &&
               this->ext_ਗੁਣਕ == ext_ਗੁਣਕ && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_ਗੁਣਕ == attn_ਗੁਣਕ && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_ਵਰਤਿਆ == mrope_ਵਰਤਿਆ && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    ਖਾਲੀ ਸੈੱਟ(int64_t theta_scale_ਲੰਮਾਈ,
             int64_t position_ਲੰਮਾਈ,
             ਦਸ਼ਮਲਵ   ext_ਗੁਣਕ,
             ਦਸ਼ਮਲਵ   theta_scale,
             ਦਸ਼ਮਲਵ   freq_scale,
             ਦਸ਼ਮਲਵ   attn_ਗੁਣਕ,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_ਵਰਤਿਆ,
             bool    is_imrope,
             ਪੂਰਨ     sections[4]) {
        this->theta_scale_ਲੰਮਾਈ = theta_scale_ਲੰਮਾਈ;
        this->position_ਲੰਮਾਈ    = position_ਲੰਮਾਈ;
        this->ext_ਗੁਣਕ         = ext_ਗੁਣਕ;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_ਗੁਣਕ        = attn_ਗੁਣਕ;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_ਵਰਤਿਆ         = mrope_ਵਰਤਿਆ;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    ਖਾਲੀ *  theta_scale_cache          = nullptr;
    ਦਸ਼ਮਲਵ * theta_scale_exp_host       = nullptr;
    ਪੂਰਨ *   position_select_ਸੂਚਕਾਂਕ_host = nullptr;
    ਖਾਲੀ *  position_select_ਸੂਚਕਾਂਕ      = nullptr;
    ਖਾਲੀ *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    ਖਾਲੀ *  sin_cache                  = nullptr;
    ਖਾਲੀ *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_ਲੰਮਾਈ         = 0;
    int64_t position_ਲੰਮਾਈ            = 0;
    bool    cached                     = false;
    ਦਸ਼ਮਲਵ   ext_ਗੁਣਕ                 = 0.0f;
    ਦਸ਼ਮਲਵ   theta_scale                = 0.0f;
    ਦਸ਼ਮਲਵ   freq_scale                 = 0.0f;
    ਦਸ਼ਮਲਵ   attn_ਗੁਣਕ                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_ਵਰਤਿਆ                 = false;
    ਪੂਰਨ     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

ਬਣਤਰ ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        ਜੇ (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    ਖਾਲੀ *  cache = nullptr;
    int64_t ਮਾਪ  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
ਬਣਤਰ ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ USE_ACL_ਗ੍ਰਾਫ
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_ਗ੍ਰਾਫ_lru_cache ਗ੍ਰਾਫ_lru_cache;
    bool                      acl_ਗ੍ਰਾਫ_ਬਹੁਲਕ = true;
#ਜੇ_ਅੰਤ
    bool                   async_ਬਹੁਲਕ;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrt_ਧਾਰਾ streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(ਪੂਰਨ device) : device(device), name("CANN" + std::ਮੰਜ਼ਿਲ_string(device)) {
        ggml_cann_ਸੈੱਟ_device(device);
        description = aclrtGetSocName();

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ USE_ACL_ਗ੍ਰਾਫ
        acl_ਗ੍ਰਾਫ_ਬਹੁਲਕ = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").ਮੁੱਲ_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_ਗ੍ਰਾਫ_ਬਹੁਲਕ ? "GRAPH" : "EAGER",
                      acl_ਗ੍ਰਾਫ_ਬਹੁਲਕ ? "acl graph enabled" : "acl graph disabled");
#ਜੇ_ਅੰਤ
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_ਸੈੱਟ_device(device);
        ਜੇ (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            ਜੇ (streams[i] != nullptr) {
                ACL_CHECK(aclrt_Destroy_ਧਾਰਾ(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrt_ਧਾਰਾ ਧਾਰਾ(ਪੂਰਨ ਧਾਰਾ) {
        ਜੇ (streams[ਧਾਰਾ] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrt_ਸੈੱਟ_Device(device));
            ACL_CHECK(aclrt_Create_ਧਾਰਾ(&streams[ਧਾਰਾ]));
        }
        ਵਾਪਸ streams[ਧਾਰਾ];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrt_ਧਾਰਾ ਧਾਰਾ() { ਵਾਪਸ ਧਾਰਾ(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    ਸਥਿਰ std::unique_ptr<ggml_cann_pool> new_pool_for_device(ਪੂਰਨ device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        ਜੇ (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        ਵਾਪਸ *mem_pool;
    }
};

#ਜੇ_ਅੰਤ  // CANN_COMMON_H
