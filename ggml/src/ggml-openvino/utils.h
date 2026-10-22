#అన్తర్భావ "ggml-decoder.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-openvino-extra.h"

#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <functional>
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <openvino/runtime/core.hpp>
#అన్తర్భావ <openvino/runtime/infer_request.hpp>
#అన్తర్భావ <string>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <utility>
#అన్తర్భావ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
నిర్మాణం లేఖాచిత్రం_కీలకం {
    పూర్ణాంకం n_nodes;
    పూర్ణాంకం n_leaves;
    std::string మొదటి_కణుపు_name;
    std::string చివరి_కణుపు_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::సదిశ<std::string> ఆదానం_srcs;

    లేఖాచిత్రం_కీలకం(స్థిరరాశి ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        ఒకవేళ (n_nodes > 0) {
            మొదటి_కణుపు_name = cgraph->nodes[0]->name;
            చివరి_కణుపు_name = cgraph->nodes[n_nodes - 1]->name;
        }

        స్థిర స్థిరరాశి bool నిండిన_కీలకం = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        ఒకవేళ (!నిండిన_కీలకం && !include_inputs) {
            తిరిగి;
        }

        std::unordered_సమితి<స్థిరరాశి ggml_tensor *> కణుపు_సమితి;
        కణుపు_సమితి.reserve(cgraph->n_nodes);
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            కణుపు_సమితి.చొప్పించు(cgraph->nodes[i]);
        }

        కొరకు (పూర్ణాంకం కణుపు_idx = 0; కణుపు_idx < cgraph->n_nodes; కణుపు_idx++) {
            స్థిరరాశి ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx];
            కొరకు (పూర్ణాంకం src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                స్థిరరాశి ggml_tensor * src = కణుపు->src[src_idx];
                ఒకవేళ (src == nullptr || src->బఫరు == nullptr || కణుపు_సమితి.లెక్క(src) || src->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS) {
                    కొనసాగు;
                }

                ఆదానం_srcs.నెట్టు_వెనుక(std::గమ్యం_string(కణుపు_idx) + ":" + std::గమ్యం_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(కణుపు));
            }
        }
    }

    bool క్రియాకారకం==(స్థిరరాశి లేఖాచిత్రం_కీలకం & other) స్థిరరాశి {
        తిరిగి n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               మొదటి_కణుపు_name == other.మొదటి_కణుపు_name &&
               చివరి_కణుపు_name == other.చివరి_కణుపు_name && ఆదానం_srcs == other.ఆదానం_srcs;
    }
};

నిర్మాణం లేఖాచిత్రం_కీలకం_hash {
    size_t క్రియాకారకం()(స్థిరరాశి లేఖాచిత్రం_కీలకం & కీలకం) స్థిరరాశి {
        size_t hash = std::hash<పూర్ణాంకం>{}(కీలకం.n_nodes);
        hash ^= std::hash<పూర్ణాంకం>{}(కీలకం.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        ఒకవేళ (కీలకం.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(కీలకం.మొదటి_కణుపు_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(కీలకం.చివరి_కణుపు_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        కొరకు (స్థిరరాశి స్వయంచాలక & s : కీలకం.ఆదానం_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        తిరిగి hash;
    }
};

నిర్మాణం decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

నిర్మాణం ov_compiled_లేఖాచిత్రం {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::సదిశ<std::string> ఆదానం_పేర్లు;
    std::సదిశ<std::string> output_పేర్లు;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
నిర్మాణం ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_లేఖాచిత్రం> graphs;
    size_t backend_లెక్క = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
నిర్మాణం ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<లేఖాచిత్రం_కీలకం, std::shared_ptr<decoder_runtime_ctx>, లేఖాచిత్రం_కీలకం_hash> decoder_cache;
    std::unordered_map<లేఖాచిత్రం_కీలకం, std::shared_ptr<ov::InferRequest>, లేఖాచిత్రం_కీలకం_hash> infer_request_cache;
    std::unordered_map<లేఖాచిత్రం_కీలకం, std::shared_ptr<ov::InferRequest>, లేఖాచిత్రం_కీలకం_hash> infer_request_cache_prefill;
    std::unordered_map<లేఖాచిత్రం_కీలకం, std::సదిశ<std::string>, లేఖాచిత్రం_కీలకం_hash> ov_ఆదానం_పేర్లు_cache;
    std::unordered_map<లేఖాచిత్రం_కీలకం, std::సదిశ<std::string>, లేఖాచిత్రం_కీలకం_hash> ov_output_పేర్లు_cache;
    size_t stateful_kv_కొలత;
    std::map<std::string, std::string> kv_state_ఆదానం_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    నిర్మాణం compute_లేఖాచిత్రం {
        ggml_cgraph లేఖాచిత్రం;
        std::సదిశ<ggml_tensor *> nodes;
    };
    std::unordered_map<స్థిరరాశి ggml_cgraph *, compute_లేఖాచిత్రం> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_కొలత(0) {}

    శూన్య clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_ఆదానం_పేర్లు_cache.clear();
        ov_output_పేర్లు_cache.clear();
        kv_state_ఆదానం_name_map.clear();
        stateful_kv_కొలత = 0;
    }

    శూన్య clear_caches() {
        std::తాళం_guard<std::mutex> తాళం(ctx_mutex);
        clear_caches_locked();
    }
};

గణన ggml_స్థితి ov_లేఖాచిత్రం_compute(నిర్మాణం ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(స్థిరరాశి శూన్య * దత్తాంశం, size_t కొలత);

bool save_ggml_tensor_దత్తాంశం_గమ్యం_txt(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి std::string & file_మార్గం);

శూన్య print_ఆదానం_tensor_info(స్థిరరాశి std::string & name, స్థిరరాశి ov::Tensor & tensor);

శూన్య print_output_tensor_info(స్థిరరాశి std::string & name, స్థిరరాశి ov::Tensor & tensor, స్థిరరాశి శూన్య * output_dst);

మూస <typename T>
std::సదిశ<T> pad_ఆదానం(స్థిరరాశి T * దత్తాంశం,
                         size_t అడ్డువరుసలు,
                         size_t నిలువుల_సంఖ్య_2,
                         size_t padded_అడ్డువరుసలు,
                         size_t padded_నిలువుల_సంఖ్య,
                         T pad_విలువ) {
    std::సదిశ<T> padded(padded_అడ్డువరుసలు * padded_నిలువుల_సంఖ్య, pad_విలువ);

    కొరకు (size_t i = 0; i < std::min(అడ్డువరుసలు, padded_అడ్డువరుసలు); ++i) {
        కొరకు (size_t j = 0; j < std::min(నిలువుల_సంఖ్య_2, padded_నిలువుల_సంఖ్య); ++j) {
            padded[i * padded_నిలువుల_సంఖ్య + j] = దత్తాంశం[i * నిలువుల_సంఖ్య_2 + j];
        }
    }

    తిరిగి padded;
}

మూస <typename T>
std::సదిశ<T> pad_ఆదానం(స్థిరరాశి ggml_tensor * tensor, size_t padded_అడ్డువరుసలు, size_t padded_నిలువుల_సంఖ్య, T pad_విలువ) {
    తిరిగి pad_ఆదానం<T>(reinterpret_cast<స్థిరరాశి T *>(tensor->దత్తాంశం),
                        static_cast<size_t>(tensor->ne[1]),  //అడ్డువరుసలు
                        static_cast<size_t>(tensor->ne[0]),  //నిలువుల_సంఖ్య
                        padded_అడ్డువరుసలు, padded_నిలువుల_సంఖ్య, pad_విలువ);
}

స్థిరరాశి ggml_tensor * get_inp_pos_tensor(నిర్మాణం ggml_cgraph * cgraph);

int64_t get_inp_pos_n_టోకెన్లు(నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * inp_pos);

bool get_is_prefill(నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * inp_pos);

bool is_naive(నిర్మాణం ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(నిర్మాణం ggml_cgraph * cgraph);
