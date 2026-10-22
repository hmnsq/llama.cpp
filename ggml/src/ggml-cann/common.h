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

#నిర్వచితం_కాకపోతే CANN_COMMON_H
#నిర్వచించు CANN_COMMON_H

#అన్తర్భావ "../ggml-impl.h"
#అన్తర్భావ "../include/ggml-cann.h"
#అన్తర్భావ "../include/ggml.h"

#అన్తర్భావ <acl/acl.h>
#అన్తర్భావ <unistd.h>

#అన్తర్భావ <atomic>
#అన్తర్భావ <condition_variable>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <functional>
#అన్తర్భావ <iostream>
#అన్తర్భావ <list>
#అన్తర్భావ <map>
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <optional>
#అన్తర్భావ <string>
#అన్తర్భావ <thread>
#అన్తర్భావ <vector>

#నిర్వచించు మాతృక_అడ్డువరుస_PADDING    512
#నిర్వచించు GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] శూన్య ggml_cann_error(స్థిరరాశి అక్షరం * stmt, స్థిరరాశి అక్షరం * func, స్థిరరాశి అక్షరం * file, పూర్ణాంకం పంక్తి, స్థిరరాశి అక్షరం * సందేశం_2);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#నిర్వచించు ACL_CHECK_GEN(stmt, success, error_fn)                                \
    చేయి {                                                                      \
        పూర్ణాంకం err_సంకేతం = (stmt);                                                \
        ఒకవేళ (err_సంకేతం != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } అయితే (0);

#నిర్వచించు ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_సందేశం)

/**
 * @brief Contains information about CANN devices.
 */
నిర్మాణం ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_లెక్క;

    /**
     * @brief Information about a single CANN device.
     */
    నిర్మాణం cann_device_info {
        పూర్ణాంకం    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t మొత్తం_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

స్థిరరాశి ggml_cann_device_info & ggml_cann_info();

శూన్య    ggml_cann_సమితి_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(స్థిరరాశి std::string & name);
bool                       parse_bool(స్థిరరాశి std::string & విలువ);
పూర్ణాంకం                        parse_integer(స్థిరరాశి std::string & విలువ);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
నిర్మాణం ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    మాయా ~ggml_cann_pool() = అప్రమేయ;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    మాయా శూన్య * alloc(size_t కొలత, size_t * actual_కొలత) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    మాయా శూన్య విడుదల(శూన్య * ptr, size_t కొలత) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
నిర్మాణం ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    శూన్య *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_కొలత = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = అప్రమేయ;

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
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t కొలత) : pool(&pool) { alloc(కొలత); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        ఒకవేళ (ptr != nullptr) {
            pool->విడుదల(ptr, actual_కొలత);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    శూన్య * alloc(size_t కొలత) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(కొలత, &this->actual_కొలత);
        తిరిగి ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    శూన్య * alloc(ggml_cann_pool & pool, size_t కొలత) {
        this->pool = &pool;
        తిరిగి alloc(కొలత);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    శూన్య * get() { తిరిగి ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(స్థిరరాశి ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & క్రియాకారకం=(స్థిరరాశి ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & క్రియాకారకం=(ggml_cann_pool_alloc &&) = delete;
};

#నిర్వచితమైతే USE_ACL_లేఖాచిత్రం
నిర్మాణం ggml_లేఖాచిత్రం_కణుపు_properties {
    // dst tensor
    శూన్య *    కణుపు_చిరునామా;
    ggml_type కణుపు_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    శూన్య *    src_చిరునామా[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op కణుపు_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / పరిమాణం(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * కణుపు) {
        ఒకవేళ (కణుపు->దత్తాంశం != this->కణుపు_చిరునామా && కణుపు->op != GGML_OP_VIEW) {
            తిరిగి false;
        }

        ఒకవేళ (కణుపు->op != this->కణుపు_op) {
            తిరిగి false;
        }

        ఒకవేళ (కణుపు->type != this->కణుపు_type) {
            తిరిగి false;
        }

        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
            ఒకవేళ (కణుపు->ne[i] != this->ne[i]) {
                తిరిగి false;
            }
            ఒకవేళ (కణుపు->nb[i] != this->nb[i]) {
                తిరిగి false;
            }
        }

        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            ఒకవేళ (కణుపు->src[i]) {
                ఒకవేళ (కణుపు->src[i]->దత్తాంశం != this->src_చిరునామా[i] && కణుపు->op != GGML_OP_VIEW) {
                    తిరిగి false;
                }

                ఒకవేళ (కణుపు->src[i]->type != this->src_type[i]) {
                    తిరిగి false;
                }

                కొరకు (పూర్ణాంకం d = 0; d < GGML_MAX_DIMS; d++) {
                    ఒకవేళ (కణుపు->src[i]->ne[d] != this->src_ne[i][d]) {
                        తిరిగి false;
                    }
                    ఒకవేళ (కణుపు->src[i]->nb[d] != this->src_nb[i][d]) {
                        తిరిగి false;
                    }
                }
            } లేకపోతే {
                ఒకవేళ (this->src_చిరునామా[i] != nullptr) {
                    తిరిగి false;
                }
            }
        }

        తిరిగి memcmp(this->op_params, కణుపు->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

నిర్మాణం ggml_cann_లేఖాచిత్రం {
    ~ggml_cann_లేఖాచిత్రం() {
        ఒకవేళ (లేఖాచిత్రం != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(లేఖాచిత్రం));
        }
    }

    aclmdlRI లేఖాచిత్రం = nullptr;

    std::సదిశ<ggml_లేఖాచిత్రం_కణుపు_properties> ggml_లేఖాచిత్రం_properties;

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
    స్థిర ggml_cann_లేఖాచిత్రం * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_లేఖాచిత్రం * new_లేఖాచిత్రం = new ggml_cann_లేఖాచిత్రం();
        new_లేఖాచిత్రం->ggml_లేఖాచిత్రం_properties.resize(cgraph->n_nodes);

        కొరకు (పూర్ణాంకం కణుపు_idx = 0; కణుపు_idx < cgraph->n_nodes; ++కణుపు_idx) {
            ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx];
            స్వయంచాలక &        prop = new_లేఖాచిత్రం->ggml_లేఖాచిత్రం_properties[కణుపు_idx];

            prop.కణుపు_చిరునామా = కణుపు->దత్తాంశం;
            prop.కణుపు_op      = కణుపు->op;
            prop.కణుపు_type    = కణుపు->type;

            std::copy_n(కణుపు->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(కణుపు->nb, GGML_MAX_DIMS, prop.nb);

            కొరకు (పూర్ణాంకం src = 0; src < GGML_MAX_SRC; ++src) {
                ఒకవేళ (కణుపు->src[src]) {
                    prop.src_చిరునామా[src] = కణుపు->src[src]->దత్తాంశం;
                    prop.src_type[src]    = కణుపు->src[src]->type;
                    std::copy_n(కణుపు->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(కణుపు->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } లేకపోతే {
                    prop.src_చిరునామా[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_లెక్క;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            జ్ఞాపకం_నకలు(prop.op_params, కణుపు->op_params, GGML_MAX_OP_PARAMS);
        }

        తిరిగి new_లేఖాచిత్రం;
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
        ఒకవేళ (this->ggml_లేఖాచిత్రం_properties.కొలత() != static_cast<size_t>(cgraph->n_nodes)) {
            తిరిగి false;
        }

        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; ++i) {
            ఒకవేళ (!this->ggml_లేఖాచిత్రం_properties[i].has_matching_properties(cgraph->nodes[i])) {
                తిరిగి false;
            }
        }

        తిరిగి true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
నిర్మాణం ggml_cann_లేఖాచిత్రం_lru_cache {
    size_t సామర్థ్యం;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_లేఖాచిత్రం *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_లేఖాచిత్రం_lru_cache() { సామర్థ్యం = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").విలువ_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    శూన్య నెట్టు(ggml_cann_లేఖాచిత్రం * new_కణుపు) {
        ఒకవేళ (cache_list.కొలత() >= సామర్థ్యం) {
            ggml_cann_లేఖాచిత్రం * old = cache_list.వెనుక();
            cache_list.తీయి_వెనుక();
            delete old;  // free the old graph
        }
        cache_list.నెట్టు_front(new_కణుపు);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    శూన్య clear() {
        కొరకు (స్వయంచాలక ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_లేఖాచిత్రం_lru_cache() { clear(); }

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
    bool కనుగొను_and_move_గమ్యం_front(ggml_cgraph * cgraph) {
        కొరకు (స్వయంచాలక & లేఖాచిత్రం_ptr : this->cache_list) {
            ఒకవేళ (లేఖాచిత్రం_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(లేఖాచిత్రం_ptr);
                cache_list.నెట్టు_front(లేఖాచిత్రం_ptr);
                తిరిగి true;
            }
        }
        తిరిగి false;
    }
};
#అయితే_ముగింపు  // USE_ACL_GRAPH

నిర్మాణం ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        ఒకవేళ (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        ఒకవేళ (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        ఒకవేళ (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        ఒకవేళ (position_select_సూచీ) {
            ACL_CHECK(aclrtFree(position_select_సూచీ));
        }
        ఒకవేళ (theta_scale_exp_host) {
            విడుదల(theta_scale_exp_host);
        }
        ఒకవేళ (position_select_సూచీ_host) {
            విడుదల(position_select_సూచీ_host);
        }
        ఒకవేళ (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_నిడివి,
               int64_t position_నిడివి,
               దశాంశం   ext_గుణకం,
               దశాంశం   theta_scale,
               దశాంశం   freq_scale,
               దశాంశం   attn_గుణకం,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_వాడిన,
               bool    is_imrope,
               పూర్ణాంకం     sections[4]) {
        తిరిగి this->theta_scale_నిడివి == theta_scale_నిడివి && this->position_నిడివి == position_నిడివి &&
               this->ext_గుణకం == ext_గుణకం && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_గుణకం == attn_గుణకం && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_వాడిన == mrope_వాడిన && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    శూన్య సమితి(int64_t theta_scale_నిడివి,
             int64_t position_నిడివి,
             దశాంశం   ext_గుణకం,
             దశాంశం   theta_scale,
             దశాంశం   freq_scale,
             దశాంశం   attn_గుణకం,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_వాడిన,
             bool    is_imrope,
             పూర్ణాంకం     sections[4]) {
        this->theta_scale_నిడివి = theta_scale_నిడివి;
        this->position_నిడివి    = position_నిడివి;
        this->ext_గుణకం         = ext_గుణకం;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_గుణకం        = attn_గుణకం;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_వాడిన         = mrope_వాడిన;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    శూన్య *  theta_scale_cache          = nullptr;
    దశాంశం * theta_scale_exp_host       = nullptr;
    పూర్ణాంకం *   position_select_సూచీ_host = nullptr;
    శూన్య *  position_select_సూచీ      = nullptr;
    శూన్య *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    శూన్య *  sin_cache                  = nullptr;
    శూన్య *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_నిడివి         = 0;
    int64_t position_నిడివి            = 0;
    bool    cached                     = false;
    దశాంశం   ext_గుణకం                 = 0.0f;
    దశాంశం   theta_scale                = 0.0f;
    దశాంశం   freq_scale                 = 0.0f;
    దశాంశం   attn_గుణకం                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_వాడిన                 = false;
    పూర్ణాంకం     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

నిర్మాణం ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        ఒకవేళ (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    శూన్య *  cache = nullptr;
    int64_t కొలత  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
నిర్మాణం ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#నిర్వచితమైతే USE_ACL_లేఖాచిత్రం
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_లేఖాచిత్రం_lru_cache లేఖాచిత్రం_lru_cache;
    bool                      acl_లేఖాచిత్రం_బహుళకం = true;
#అయితే_ముగింపు
    bool                   async_బహుళకం;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrt_ప్రవాహం streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(పూర్ణాంకం device) : device(device), name("CANN" + std::గమ్యం_string(device)) {
        ggml_cann_సమితి_device(device);
        description = aclrtGetSocName();

#నిర్వచితమైతే USE_ACL_లేఖాచిత్రం
        acl_లేఖాచిత్రం_బహుళకం = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").విలువ_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_లేఖాచిత్రం_బహుళకం ? "GRAPH" : "EAGER",
                      acl_లేఖాచిత్రం_బహుళకం ? "acl graph enabled" : "acl graph disabled");
#అయితే_ముగింపు
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_సమితి_device(device);
        ఒకవేళ (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        కొరకు (పూర్ణాంకం i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            ఒకవేళ (streams[i] != nullptr) {
                ACL_CHECK(aclrt_Destroy_ప్రవాహం(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrt_ప్రవాహం ప్రవాహం(పూర్ణాంకం ప్రవాహం) {
        ఒకవేళ (streams[ప్రవాహం] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrt_సమితి_Device(device));
            ACL_CHECK(aclrt_Create_ప్రవాహం(&streams[ప్రవాహం]));
        }
        తిరిగి streams[ప్రవాహం];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrt_ప్రవాహం ప్రవాహం() { తిరిగి ప్రవాహం(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    స్థిర std::unique_ptr<ggml_cann_pool> new_pool_for_device(పూర్ణాంకం device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        ఒకవేళ (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        తిరిగి *mem_pool;
    }
};

#అయితే_ముగింపు  // CANN_COMMON_H
