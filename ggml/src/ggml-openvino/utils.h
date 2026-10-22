#അന്തര്ഭാവ "ggml-decoder.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-openvino-extra.h"

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <functional>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <openvino/runtime/core.hpp>
#അന്തര്ഭാവ <openvino/runtime/infer_request.hpp>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <utility>
#അന്തര്ഭാവ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
ഘടന graph_താക്കോൽ {
    പൂർണ്ണസംഖ്യ n_nodes;
    പൂർണ്ണസംഖ്യ n_leaves;
    std::string first_കണ്ണി_name;
    std::string last_കണ്ണി_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> നിവേശം_srcs;

    graph_താക്കോൽ(സ്ഥിരം ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        എങ്കിൽ (n_nodes > 0) {
            first_കണ്ണി_name = cgraph->nodes[0]->name;
            last_കണ്ണി_name = cgraph->nodes[n_nodes - 1]->name;
        }

        സ്ഥിതം സ്ഥിരം bool full_താക്കോൽ = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        എങ്കിൽ (!full_താക്കോൽ && !include_inputs) {
            തിരികെ;
        }

        std::unordered_set<സ്ഥിരം ggml_tensor *> കണ്ണി_set;
        കണ്ണി_set.reserve(cgraph->n_nodes);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
            കണ്ണി_set.തിരുകുക(cgraph->nodes[i]);
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ കണ്ണി_idx = 0; കണ്ണി_idx < cgraph->n_nodes; കണ്ണി_idx++) {
            സ്ഥിരം ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_idx];
            വേണ്ടി (പൂർണ്ണസംഖ്യ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                സ്ഥിരം ggml_tensor * src = കണ്ണി->src[src_idx];
                എങ്കിൽ (src == nullptr || src->buffer == nullptr || കണ്ണി_set.എണ്ണം(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    തുടരുക;
                }

                നിവേശം_srcs.തള്ളുക_back(std::to_string(കണ്ണി_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(കണ്ണി));
            }
        }
    }

    bool operator==(സ്ഥിരം graph_താക്കോൽ & other) സ്ഥിരം {
        തിരികെ n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_കണ്ണി_name == other.first_കണ്ണി_name &&
               last_കണ്ണി_name == other.last_കണ്ണി_name && നിവേശം_srcs == other.നിവേശം_srcs;
    }
};

ഘടന graph_താക്കോൽ_hash {
    size_t operator()(സ്ഥിരം graph_താക്കോൽ & താക്കോൽ) സ്ഥിരം {
        size_t hash = std::hash<പൂർണ്ണസംഖ്യ>{}(താക്കോൽ.n_nodes);
        hash ^= std::hash<പൂർണ്ണസംഖ്യ>{}(താക്കോൽ.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        എങ്കിൽ (താക്കോൽ.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(താക്കോൽ.first_കണ്ണി_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(താക്കോൽ.last_കണ്ണി_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        വേണ്ടി (സ്ഥിരം auto & s : താക്കോൽ.നിവേശം_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        തിരികെ hash;
    }
};

ഘടന decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

ഘടന ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> നിവേശം_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
ഘടന ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_എണ്ണം = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
ഘടന ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_താക്കോൽ, std::shared_ptr<decoder_runtime_ctx>, graph_താക്കോൽ_hash> decoder_cache;
    std::unordered_map<graph_താക്കോൽ, std::shared_ptr<ov::InferRequest>, graph_താക്കോൽ_hash> infer_request_cache;
    std::unordered_map<graph_താക്കോൽ, std::shared_ptr<ov::InferRequest>, graph_താക്കോൽ_hash> infer_request_cache_prefill;
    std::unordered_map<graph_താക്കോൽ, std::vector<std::string>, graph_താക്കോൽ_hash> ov_നിവേശം_names_cache;
    std::unordered_map<graph_താക്കോൽ, std::vector<std::string>, graph_താക്കോൽ_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_നിവേശം_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    ഘടന compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<സ്ഥിരം ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    ശൂന്യം clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_നിവേശം_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_നിവേശം_name_map.clear();
        stateful_kv_size = 0;
    }

    ശൂന്യം clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

എണ്ണൽ ggml_status ov_graph_compute(ഘടന ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(സ്ഥിരം ശൂന്യം * data, size_t size);

bool save_ggml_tensor_data_to_txt(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം std::string & file_path);

ശൂന്യം print_നിവേശം_tensor_info(സ്ഥിരം std::string & name, സ്ഥിരം ov::Tensor & tensor);

ശൂന്യം print_output_tensor_info(സ്ഥിരം std::string & name, സ്ഥിരം ov::Tensor & tensor, സ്ഥിരം ശൂന്യം * output_dst);

template <typename T>
std::vector<T> pad_നിവേശം(സ്ഥിരം T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_മൂല്യം) {
    std::vector<T> padded(padded_rows * padded_cols, pad_മൂല്യം);

    വേണ്ടി (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        വേണ്ടി (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    തിരികെ padded;
}

template <typename T>
std::vector<T> pad_നിവേശം(സ്ഥിരം ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_മൂല്യം) {
    തിരികെ pad_നിവേശം<T>(reinterpret_cast<സ്ഥിരം T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_മൂല്യം);
}

സ്ഥിരം ggml_tensor * get_inp_pos_tensor(ഘടന ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(ഘടന ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * inp_pos);

bool get_is_prefill(ഘടന ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * inp_pos);

bool is_naive(ഘടന ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(ഘടന ggml_cgraph * cgraph);
