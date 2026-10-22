#अन्तर्भाव "ggml-decoder.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino-extra.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <functional>
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <openvino/runtime/core.hpp>
#अन्तर्भाव <openvino/runtime/infer_request.hpp>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
संरचना graph_कुंजी {
    पूर्णाङ्क n_nodes;
    पूर्णाङ्क n_leaves;
    std::string first_गाँठ_name;
    std::string last_गाँठ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> निवेश_srcs;

    graph_कुंजी(ध्रुव ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        यदि (n_nodes > 0) {
            first_गाँठ_name = cgraph->nodes[0]->name;
            last_गाँठ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        स्थिर ध्रुव bool full_कुंजी = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        यदि (!full_कुंजी && !include_inputs) {
            प्रत्यागम;
        }

        std::unordered_set<ध्रुव ggml_tensor *> गाँठ_set;
        गाँठ_set.reserve(cgraph->n_nodes);
        प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            गाँठ_set.जोड़ो(cgraph->nodes[i]);
        }

        प्रति (पूर्णाङ्क गाँठ_idx = 0; गाँठ_idx < cgraph->n_nodes; गाँठ_idx++) {
            ध्रुव ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx];
            प्रति (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ध्रुव ggml_tensor * src = गाँठ->src[src_idx];
                यदि (src == nullptr || src->buffer == nullptr || गाँठ_set.गिनती(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    अनुवर्त;
                }

                निवेश_srcs.धकेलो_back(std::to_string(गाँठ_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(गाँठ));
            }
        }
    }

    bool चालक==(ध्रुव graph_कुंजी & other) ध्रुव {
        प्रत्यागम n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_गाँठ_name == other.first_गाँठ_name &&
               last_गाँठ_name == other.last_गाँठ_name && निवेश_srcs == other.निवेश_srcs;
    }
};

संरचना graph_कुंजी_hash {
    माप_प्रकार चालक()(ध्रुव graph_कुंजी & कुंजी) ध्रुव {
        माप_प्रकार hash = std::hash<पूर्णाङ्क>{}(कुंजी.n_nodes);
        hash ^= std::hash<पूर्णाङ्क>{}(कुंजी.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        यदि (कुंजी.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(कुंजी.first_गाँठ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(कुंजी.last_गाँठ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        प्रति (ध्रुव स्वतः & s : कुंजी.निवेश_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        प्रत्यागम hash;
    }
};

संरचना decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

संरचना ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> निवेश_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
संरचना ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    माप_प्रकार backend_गिनती = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
संरचना ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_कुंजी, std::shared_ptr<decoder_runtime_ctx>, graph_कुंजी_hash> decoder_cache;
    std::unordered_map<graph_कुंजी, std::shared_ptr<ov::InferRequest>, graph_कुंजी_hash> infer_request_cache;
    std::unordered_map<graph_कुंजी, std::shared_ptr<ov::InferRequest>, graph_कुंजी_hash> infer_request_cache_prefill;
    std::unordered_map<graph_कुंजी, std::vector<std::string>, graph_कुंजी_hash> ov_निवेश_names_cache;
    std::unordered_map<graph_कुंजी, std::vector<std::string>, graph_कुंजी_hash> ov_output_names_cache;
    माप_प्रकार stateful_kv_size;
    std::map<std::string, std::string> kv_state_निवेश_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    संरचना compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<ध्रुव ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    शून्य clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_निवेश_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_निवेश_name_map.clear();
        stateful_kv_size = 0;
    }

    शून्य clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

गणना ggml_status ov_graph_compute(संरचना ggml_cgraph * cgraph, ggml_backend_t backend);

माप_प्रकार checksum(ध्रुव शून्य * data, माप_प्रकार size);

bool save_ggml_tensor_data_to_txt(ध्रुव ggml_tensor * tensor, ध्रुव std::string & file_path);

शून्य print_निवेश_tensor_info(ध्रुव std::string & name, ध्रुव ov::Tensor & tensor);

शून्य print_output_tensor_info(ध्रुव std::string & name, ध्रुव ov::Tensor & tensor, ध्रुव शून्य * output_dst);

ढाँचा <typename T>
std::vector<T> pad_निवेश(ध्रुव T * data,
                         माप_प्रकार rows,
                         माप_प्रकार cols,
                         माप_प्रकार padded_rows,
                         माप_प्रकार padded_cols,
                         T pad_मूल्य) {
    std::vector<T> padded(padded_rows * padded_cols, pad_मूल्य);

    प्रति (माप_प्रकार i = 0; i < std::min(rows, padded_rows); ++i) {
        प्रति (माप_प्रकार j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    प्रत्यागम padded;
}

ढाँचा <typename T>
std::vector<T> pad_निवेश(ध्रुव ggml_tensor * tensor, माप_प्रकार padded_rows, माप_प्रकार padded_cols, T pad_मूल्य) {
    प्रत्यागम pad_निवेश<T>(reinterpret_cast<ध्रुव T *>(tensor->data),
                        static_cast<माप_प्रकार>(tensor->ne[1]),  // rows
                        static_cast<माप_प्रकार>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_मूल्य);
}

ध्रुव ggml_tensor * get_inp_pos_tensor(संरचना ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(संरचना ggml_cgraph * cgraph, ध्रुव ggml_tensor * inp_pos);

bool get_is_prefill(संरचना ggml_cgraph * cgraph, ध्रुव ggml_tensor * inp_pos);

bool is_naive(संरचना ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(संरचना ggml_cgraph * cgraph);
