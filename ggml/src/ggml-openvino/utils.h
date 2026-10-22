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
संरचना graph_कुञ्चिका {
    पूर्णाङ्क n_nodes;
    पूर्णाङ्क n_leaves;
    std::string first_ग्रन्थिः_name;
    std::string last_ग्रन्थिः_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> निवेशः_srcs;

    graph_कुञ्चिका(ध्रुव ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        सचे (n_nodes > 0) {
            first_ग्रन्थिः_name = cgraph->nodes[0]->name;
            last_ग्रन्थिः_name = cgraph->nodes[n_nodes - 1]->name;
        }

        स्थिर ध्रुव bool full_कुञ्चिका = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        सचे (!full_कुञ्चिका && !include_inputs) {
            पटिनिवत्त;
        }

        std::unordered_set<ध्रुव ggml_tensor *> ग्रन्थिः_set;
        ग्रन्थिः_set.reserve(cgraph->n_nodes);
        अत्थाय (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            ग्रन्थिः_set.निवेशय(cgraph->nodes[i]);
        }

        अत्थाय (पूर्णाङ्क ग्रन्थिः_idx = 0; ग्रन्थिः_idx < cgraph->n_nodes; ग्रन्थिः_idx++) {
            ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];
            अत्थाय (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ध्रुव ggml_tensor * src = ग्रन्थिः->src[src_idx];
                सचे (src == nullptr || src->buffer == nullptr || ग्रन्थिः_set.गणनम्(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    अनुवर्त;
                }

                निवेशः_srcs.प्रक्षिप_back(std::to_string(ग्रन्थिः_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(ग्रन्थिः));
            }
        }
    }

    bool operator==(ध्रुव graph_कुञ्चिका & other) ध्रुव {
        पटिनिवत्त n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_ग्रन्थिः_name == other.first_ग्रन्थिः_name &&
               last_ग्रन्थिः_name == other.last_ग्रन्थिः_name && निवेशः_srcs == other.निवेशः_srcs;
    }
};

संरचना graph_कुञ्चिका_hash {
    size_t operator()(ध्रुव graph_कुञ्चिका & कुञ्चिका) ध्रुव {
        size_t hash = std::hash<पूर्णाङ्क>{}(कुञ्चिका.n_nodes);
        hash ^= std::hash<पूर्णाङ्क>{}(कुञ्चिका.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        सचे (कुञ्चिका.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(कुञ्चिका.first_ग्रन्थिः_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(कुञ्चिका.last_ग्रन्थिः_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        अत्थाय (ध्रुव auto & s : कुञ्चिका.निवेशः_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        पटिनिवत्त hash;
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
    std::vector<std::string> निवेशः_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
संरचना ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_गणनम् = 0;
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
    std::unordered_map<graph_कुञ्चिका, std::shared_ptr<decoder_runtime_ctx>, graph_कुञ्चिका_hash> decoder_cache;
    std::unordered_map<graph_कुञ्चिका, std::shared_ptr<ov::InferRequest>, graph_कुञ्चिका_hash> infer_request_cache;
    std::unordered_map<graph_कुञ्चिका, std::shared_ptr<ov::InferRequest>, graph_कुञ्चिका_hash> infer_request_cache_prefill;
    std::unordered_map<graph_कुञ्चिका, std::vector<std::string>, graph_कुञ्चिका_hash> ov_निवेशः_names_cache;
    std::unordered_map<graph_कुञ्चिका, std::vector<std::string>, graph_कुञ्चिका_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_निवेशः_name_map;
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
        ov_निवेशः_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_निवेशः_name_map.clear();
        stateful_kv_size = 0;
    }

    शून्य clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

गणना ggml_status ov_graph_compute(संरचना ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ध्रुव शून्य * data, size_t size);

bool save_ggml_tensor_data_to_txt(ध्रुव ggml_tensor * tensor, ध्रुव std::string & file_path);

शून्य print_निवेशः_tensor_info(ध्रुव std::string & name, ध्रुव ov::Tensor & tensor);

शून्य print_output_tensor_info(ध्रुव std::string & name, ध्रुव ov::Tensor & tensor, ध्रुव शून्य * output_dst);

template <typename T>
std::vector<T> pad_निवेशः(ध्रुव T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_मानम्) {
    std::vector<T> padded(padded_rows * padded_cols, pad_मानम्);

    अत्थाय (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        अत्थाय (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    पटिनिवत्त padded;
}

template <typename T>
std::vector<T> pad_निवेशः(ध्रुव ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_मानम्) {
    पटिनिवत्त pad_निवेशः<T>(reinterpret_cast<ध्रुव T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_मानम्);
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
