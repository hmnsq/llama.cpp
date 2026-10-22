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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ CANN_COMMON_H
#ವ್ಯಾಖ್ಯಾನಿಸು CANN_COMMON_H

#ಅನ್ತರ್ಭಾವ "../ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "../include/ggml-cann.h"
#ಅನ್ತರ್ಭಾವ "../include/ggml.h"

#ಅನ್ತರ್ಭಾವ <acl/acl.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <condition_variable>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <functional>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <list>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <thread>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿಸು ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING    512
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] ಶೂನ್ಯ ggml_cann_error(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * stmt, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * func, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * file, ಪೂರ್ಣಾಂಕ ಸಾಲು, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸಂದೇಶ_2);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#ವ್ಯಾಖ್ಯಾನಿಸು ACL_CHECK_GEN(stmt, success, error_fn)                                \
    ಮಾಡು {                                                                      \
        ಪೂರ್ಣಾಂಕ err_ಸಂಕೇತ = (stmt);                                                \
        ಆದರೆ (err_ಸಂಕೇತ != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } ಆಗ (0);

#ವ್ಯಾಖ್ಯಾನಿಸು ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_ಸಂದೇಶ)

/**
 * @brief Contains information about CANN devices.
 */
ರಚನೆ ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_ಎಣಿಕೆ;

    /**
     * @brief Information about a single CANN device.
     */
    ರಚನೆ cann_device_info {
        ಪೂರ್ಣಾಂಕ    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t ಒಟ್ಟು_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

ಸ್ಥಿರಾಂಕ ggml_cann_device_info & ggml_cann_info();

ಶೂನ್ಯ    ggml_cann_ಗಣ_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(ಸ್ಥಿರಾಂಕ std::string & name);
bool                       parse_bool(ಸ್ಥಿರಾಂಕ std::string & ಮೌಲ್ಯ);
ಪೂರ್ಣಾಂಕ                        parse_integer(ಸ್ಥಿರಾಂಕ std::string & ಮೌಲ್ಯ);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
ರಚನೆ ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    ವಾಸ್ತವಿಕ ~ggml_cann_pool() = ಪೂರ್ವನಿಯೋಜಿತ;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ * alloc(size_t ಗಾತ್ರ_2, size_t * actual_ಗಾತ್ರ) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t ಗಾತ್ರ_2) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
ರಚನೆ ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    ಶೂನ್ಯ *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_ಗಾತ್ರ = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = ಪೂರ್ವನಿಯೋಜಿತ;

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
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t ಗಾತ್ರ_2) : pool(&pool) { alloc(ಗಾತ್ರ_2); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        ಆದರೆ (ptr != nullptr) {
            pool->ಬಿಡುಗಡೆ(ptr, actual_ಗಾತ್ರ);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    ಶೂನ್ಯ * alloc(size_t ಗಾತ್ರ_2) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(ಗಾತ್ರ_2, &this->actual_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    ಶೂನ್ಯ * alloc(ggml_cann_pool & pool, size_t ಗಾತ್ರ_2) {
        this->pool = &pool;
        ಹಿಂತಿರುಗಿ alloc(ಗಾತ್ರ_2);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    ಶೂನ್ಯ * get() { ಹಿಂತಿರುಗಿ ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(ಸ್ಥಿರಾಂಕ ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & ಕ್ರಿಯಾಕಾರಕ=(ಸ್ಥಿರಾಂಕ ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & ಕ್ರಿಯಾಕಾರಕ=(ggml_cann_pool_alloc &&) = delete;
};

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ USE_ACL_ನಕ್ಷೆ
ರಚನೆ ggml_ನಕ್ಷೆ_ಗಂಟು_properties {
    // dst tensor
    ಶೂನ್ಯ *    ಗಂಟು_ವಿಳಾಸ;
    ggml_type ಗಂಟು_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    ಶೂನ್ಯ *    src_ವಿಳಾಸ[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op ಗಂಟು_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * ಗಂಟು) {
        ಆದರೆ (ಗಂಟು->ದತ್ತಾಂಶ != this->ಗಂಟು_ವಿಳಾಸ && ಗಂಟು->op != GGML_OP_VIEW) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಆದರೆ (ಗಂಟು->op != this->ಗಂಟು_op) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಆದರೆ (ಗಂಟು->type != this->ಗಂಟು_type) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            ಆದರೆ (ಗಂಟು->ne[i] != this->ne[i]) {
                ಹಿಂತಿರುಗಿ false;
            }
            ಆದರೆ (ಗಂಟು->nb[i] != this->nb[i]) {
                ಹಿಂತಿರುಗಿ false;
            }
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಆದರೆ (ಗಂಟು->src[i]) {
                ಆದರೆ (ಗಂಟು->src[i]->ದತ್ತಾಂಶ != this->src_ವಿಳಾಸ[i] && ಗಂಟು->op != GGML_OP_VIEW) {
                    ಹಿಂತಿರುಗಿ false;
                }

                ಆದರೆ (ಗಂಟು->src[i]->type != this->src_type[i]) {
                    ಹಿಂತಿರುಗಿ false;
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < GGML_MAX_DIMS; d++) {
                    ಆದರೆ (ಗಂಟು->src[i]->ne[d] != this->src_ne[i][d]) {
                        ಹಿಂತಿರುಗಿ false;
                    }
                    ಆದರೆ (ಗಂಟು->src[i]->nb[d] != this->src_nb[i][d]) {
                        ಹಿಂತಿರುಗಿ false;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (this->src_ವಿಳಾಸ[i] != nullptr) {
                    ಹಿಂತಿರುಗಿ false;
                }
            }
        }

        ಹಿಂತಿರುಗಿ memcmp(this->op_params, ಗಂಟು->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

ರಚನೆ ggml_cann_ನಕ್ಷೆ {
    ~ggml_cann_ನಕ್ಷೆ() {
        ಆದರೆ (ನಕ್ಷೆ != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(ನಕ್ಷೆ));
        }
    }

    aclmdlRI ನಕ್ಷೆ = nullptr;

    std::ಸದಿಶ<ggml_ನಕ್ಷೆ_ಗಂಟು_properties> ggml_ನಕ್ಷೆ_properties;

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
    ಸ್ಥಿರ ggml_cann_ನಕ್ಷೆ * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_ನಕ್ಷೆ * new_ನಕ್ಷೆ = new ggml_cann_ನಕ್ಷೆ();
        new_ನಕ್ಷೆ->ggml_ನಕ್ಷೆ_properties.resize(cgraph->n_nodes);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_idx = 0; ಗಂಟು_idx < cgraph->n_nodes; ++ಗಂಟು_idx) {
            ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
            ಸ್ವಯಂಚಾಲಿತ &        prop = new_ನಕ್ಷೆ->ggml_ನಕ್ಷೆ_properties[ಗಂಟು_idx];

            prop.ಗಂಟು_ವಿಳಾಸ = ಗಂಟು->ದತ್ತಾಂಶ;
            prop.ಗಂಟು_op      = ಗಂಟು->op;
            prop.ಗಂಟು_type    = ಗಂಟು->type;

            std::copy_n(ಗಂಟು->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(ಗಂಟು->nb, GGML_MAX_DIMS, prop.nb);

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ src = 0; src < GGML_MAX_SRC; ++src) {
                ಆದರೆ (ಗಂಟು->src[src]) {
                    prop.src_ವಿಳಾಸ[src] = ಗಂಟು->src[src]->ದತ್ತಾಂಶ;
                    prop.src_type[src]    = ಗಂಟು->src[src]->type;
                    std::copy_n(ಗಂಟು->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(ಗಂಟು->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    prop.src_ವಿಳಾಸ[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_ಎಣಿಕೆ;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            ಸ್ಮರಣೆ_ನಕಲು(prop.op_params, ಗಂಟು->op_params, GGML_MAX_OP_PARAMS);
        }

        ಹಿಂತಿರುಗಿ new_ನಕ್ಷೆ;
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
        ಆದರೆ (this->ggml_ನಕ್ಷೆ_properties.ಗಾತ್ರ_2() != static_cast<size_t>(cgraph->n_nodes)) {
            ಹಿಂತಿರುಗಿ false;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; ++i) {
            ಆದರೆ (!this->ggml_ನಕ್ಷೆ_properties[i].has_matching_properties(cgraph->nodes[i])) {
                ಹಿಂತಿರುಗಿ false;
            }
        }

        ಹಿಂತಿರುಗಿ true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
ರಚನೆ ggml_cann_ನಕ್ಷೆ_lru_cache {
    size_t ಸಾಮರ್ಥ್ಯ;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_ನಕ್ಷೆ *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_ನಕ್ಷೆ_lru_cache() { ಸಾಮರ್ಥ್ಯ = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").ಮೌಲ್ಯ_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    ಶೂನ್ಯ ತಳ್ಳು(ggml_cann_ನಕ್ಷೆ * new_ಗಂಟು) {
        ಆದರೆ (cache_list.ಗಾತ್ರ_2() >= ಸಾಮರ್ಥ್ಯ) {
            ggml_cann_ನಕ್ಷೆ * old = cache_list.ಹಿಂಭಾಗ();
            cache_list.ತೆಗೆ_ಹಿಂಭಾಗ();
            delete old;  // free the old graph
        }
        cache_list.ತಳ್ಳು_front(new_ಗಂಟು);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    ಶೂನ್ಯ clear() {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_ನಕ್ಷೆ_lru_cache() { clear(); }

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
    bool ಕಂಡುಹಿಡಿ_and_move_ಗಮ್ಯ_front(ggml_cgraph * cgraph) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ನಕ್ಷೆ_ptr : this->cache_list) {
            ಆದರೆ (ನಕ್ಷೆ_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(ನಕ್ಷೆ_ptr);
                cache_list.ತಳ್ಳು_front(ನಕ್ಷೆ_ptr);
                ಹಿಂತಿರುಗಿ true;
            }
        }
        ಹಿಂತಿರುಗಿ false;
    }
};
#ಆದರೆ_ಅಂತ್ಯ  // USE_ACL_GRAPH

ರಚನೆ ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        ಆದರೆ (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        ಆದರೆ (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        ಆದರೆ (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        ಆದರೆ (position_select_ಸೂಚ್ಯಂಕ) {
            ACL_CHECK(aclrtFree(position_select_ಸೂಚ್ಯಂಕ));
        }
        ಆದರೆ (theta_scale_exp_host) {
            ಬಿಡುಗಡೆ(theta_scale_exp_host);
        }
        ಆದರೆ (position_select_ಸೂಚ್ಯಂಕ_host) {
            ಬಿಡುಗಡೆ(position_select_ಸೂಚ್ಯಂಕ_host);
        }
        ಆದರೆ (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_ಉದ್ದಳತೆ,
               int64_t position_ಉದ್ದಳತೆ,
               ದಶಮಾಂಶ   ext_ಗುಣಕ,
               ದಶಮಾಂಶ   theta_scale,
               ದಶಮಾಂಶ   freq_scale,
               ದಶಮಾಂಶ   attn_ಗುಣಕ,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_ಬಳಸಿದ,
               bool    is_imrope,
               ಪೂರ್ಣಾಂಕ     sections[4]) {
        ಹಿಂತಿರುಗಿ this->theta_scale_ಉದ್ದಳತೆ == theta_scale_ಉದ್ದಳತೆ && this->position_ಉದ್ದಳತೆ == position_ಉದ್ದಳತೆ &&
               this->ext_ಗುಣಕ == ext_ಗುಣಕ && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_ಗುಣಕ == attn_ಗುಣಕ && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_ಬಳಸಿದ == mrope_ಬಳಸಿದ && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    ಶೂನ್ಯ ಗಣ(int64_t theta_scale_ಉದ್ದಳತೆ,
             int64_t position_ಉದ್ದಳತೆ,
             ದಶಮಾಂಶ   ext_ಗುಣಕ,
             ದಶಮಾಂಶ   theta_scale,
             ದಶಮಾಂಶ   freq_scale,
             ದಶಮಾಂಶ   attn_ಗುಣಕ,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_ಬಳಸಿದ,
             bool    is_imrope,
             ಪೂರ್ಣಾಂಕ     sections[4]) {
        this->theta_scale_ಉದ್ದಳತೆ = theta_scale_ಉದ್ದಳತೆ;
        this->position_ಉದ್ದಳತೆ    = position_ಉದ್ದಳತೆ;
        this->ext_ಗುಣಕ         = ext_ಗುಣಕ;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_ಗುಣಕ        = attn_ಗುಣಕ;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_ಬಳಸಿದ         = mrope_ಬಳಸಿದ;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    ಶೂನ್ಯ *  theta_scale_cache          = nullptr;
    ದಶಮಾಂಶ * theta_scale_exp_host       = nullptr;
    ಪೂರ್ಣಾಂಕ *   position_select_ಸೂಚ್ಯಂಕ_host = nullptr;
    ಶೂನ್ಯ *  position_select_ಸೂಚ್ಯಂಕ      = nullptr;
    ಶೂನ್ಯ *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    ಶೂನ್ಯ *  sin_cache                  = nullptr;
    ಶೂನ್ಯ *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_ಉದ್ದಳತೆ         = 0;
    int64_t position_ಉದ್ದಳತೆ            = 0;
    bool    cached                     = false;
    ದಶಮಾಂಶ   ext_ಗುಣಕ                 = 0.0f;
    ದಶಮಾಂಶ   theta_scale                = 0.0f;
    ದಶಮಾಂಶ   freq_scale                 = 0.0f;
    ದಶಮಾಂಶ   attn_ಗುಣಕ                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_ಬಳಸಿದ                 = false;
    ಪೂರ್ಣಾಂಕ     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

ರಚನೆ ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        ಆದರೆ (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    ಶೂನ್ಯ *  cache = nullptr;
    int64_t ಗಾತ್ರ_2  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
ರಚನೆ ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ USE_ACL_ನಕ್ಷೆ
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_ನಕ್ಷೆ_lru_cache ನಕ್ಷೆ_lru_cache;
    bool                      acl_ನಕ್ಷೆ_ಬಹುಲಕ = true;
#ಆದರೆ_ಅಂತ್ಯ
    bool                   async_ಬಹುಲಕ;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrt_ಹರಿವು streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(ಪೂರ್ಣಾಂಕ device) : device(device), name("CANN" + std::ಗಮ್ಯ_string(device)) {
        ggml_cann_ಗಣ_device(device);
        description = aclrtGetSocName();

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ USE_ACL_ನಕ್ಷೆ
        acl_ನಕ್ಷೆ_ಬಹುಲಕ = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").ಮೌಲ್ಯ_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_ನಕ್ಷೆ_ಬಹುಲಕ ? "GRAPH" : "EAGER",
                      acl_ನಕ್ಷೆ_ಬಹುಲಕ ? "acl graph enabled" : "acl graph disabled");
#ಆದರೆ_ಅಂತ್ಯ
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_ಗಣ_device(device);
        ಆದರೆ (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            ಆದರೆ (streams[i] != nullptr) {
                ACL_CHECK(aclrt_Destroy_ಹರಿವು(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrt_ಹರಿವು ಹರಿವು(ಪೂರ್ಣಾಂಕ ಹರಿವು) {
        ಆದರೆ (streams[ಹರಿವು] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrt_ಗಣ_Device(device));
            ACL_CHECK(aclrt_Create_ಹರಿವು(&streams[ಹರಿವು]));
        }
        ಹಿಂತಿರುಗಿ streams[ಹರಿವು];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrt_ಹರಿವು ಹರಿವು() { ಹಿಂತಿರುಗಿ ಹರಿವು(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    ಸ್ಥಿರ std::unique_ptr<ggml_cann_pool> new_pool_for_device(ಪೂರ್ಣಾಂಕ device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        ಆದರೆ (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        ಹಿಂತಿರುಗಿ *mem_pool;
    }
};

#ಆದರೆ_ಅಂತ್ಯ  // CANN_COMMON_H
