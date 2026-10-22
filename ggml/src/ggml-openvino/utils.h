#અન્તર્ભાવ "ggml-decoder.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-openvino-extra.h"

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <functional>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <openvino/runtime/core.hpp>
#અન્તર્ભાવ <openvino/runtime/infer_request.hpp>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <unordered_set>
#અન્તર્ભાવ <utility>
#અન્તર્ભાવ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
સંરચના graph_ચાવી {
    પૂર્ણાંક n_nodes;
    પૂર્ણાંક n_leaves;
    std::string first_ગાંઠ_name;
    std::string last_ગાંઠ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> આગત_srcs;

    graph_ચાવી(અચલ ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        જો (n_nodes > 0) {
            first_ગાંઠ_name = cgraph->nodes[0]->name;
            last_ગાંઠ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        સ્થિર અચલ bool full_ચાવી = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        જો (!full_ચાવી && !include_inputs) {
            પાછા;
        }

        std::unordered_set<અચલ ggml_tensor *> ગાંઠ_set;
        ગાંઠ_set.reserve(cgraph->n_nodes);
        માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
            ગાંઠ_set.ઉમેરો(cgraph->nodes[i]);
        }

        માટે (પૂર્ણાંક ગાંઠ_idx = 0; ગાંઠ_idx < cgraph->n_nodes; ગાંઠ_idx++) {
            અચલ ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idx];
            માટે (પૂર્ણાંક src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                અચલ ggml_tensor * src = ગાંઠ->src[src_idx];
                જો (src == nullptr || src->buffer == nullptr || ગાંઠ_set.ગણતરી(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    ચાલુ;
                }

                આગત_srcs.ધકેલો_back(std::to_string(ગાંઠ_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(ગાંઠ));
            }
        }
    }

    bool operator==(અચલ graph_ચાવી & other) અચલ {
        પાછા n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_ગાંઠ_name == other.first_ગાંઠ_name &&
               last_ગાંઠ_name == other.last_ગાંઠ_name && આગત_srcs == other.આગત_srcs;
    }
};

સંરચના graph_ચાવી_hash {
    size_t operator()(અચલ graph_ચાવી & ચાવી) અચલ {
        size_t hash = std::hash<પૂર્ણાંક>{}(ચાવી.n_nodes);
        hash ^= std::hash<પૂર્ણાંક>{}(ચાવી.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        જો (ચાવી.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(ચાવી.first_ગાંઠ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(ચાવી.last_ગાંઠ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        માટે (અચલ auto & s : ચાવી.આગત_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        પાછા hash;
    }
};

સંરચના decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

સંરચના ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> આગત_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
સંરચના ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_ગણતરી = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
સંરચના ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_ચાવી, std::shared_ptr<decoder_runtime_ctx>, graph_ચાવી_hash> decoder_cache;
    std::unordered_map<graph_ચાવી, std::shared_ptr<ov::InferRequest>, graph_ચાવી_hash> infer_request_cache;
    std::unordered_map<graph_ચાવી, std::shared_ptr<ov::InferRequest>, graph_ચાવી_hash> infer_request_cache_prefill;
    std::unordered_map<graph_ચાવી, std::vector<std::string>, graph_ચાવી_hash> ov_આગત_names_cache;
    std::unordered_map<graph_ચાવી, std::vector<std::string>, graph_ચાવી_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_આગત_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    સંરચના compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<અચલ ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    શૂન્ય clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_આગત_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_આગત_name_map.clear();
        stateful_kv_size = 0;
    }

    શૂન્ય clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

ગણના ggml_status ov_graph_compute(સંરચના ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(અચલ શૂન્ય * data, size_t size);

bool save_ggml_tensor_data_to_txt(અચલ ggml_tensor * tensor, અચલ std::string & file_path);

શૂન્ય print_આગત_tensor_info(અચલ std::string & name, અચલ ov::Tensor & tensor);

શૂન્ય print_output_tensor_info(અચલ std::string & name, અચલ ov::Tensor & tensor, અચલ શૂન્ય * output_dst);

template <typename T>
std::vector<T> pad_આગત(અચલ T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_કિંમત) {
    std::vector<T> padded(padded_rows * padded_cols, pad_કિંમત);

    માટે (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        માટે (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    પાછા padded;
}

template <typename T>
std::vector<T> pad_આગત(અચલ ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_કિંમત) {
    પાછા pad_આગત<T>(reinterpret_cast<અચલ T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_કિંમત);
}

અચલ ggml_tensor * get_inp_pos_tensor(સંરચના ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(સંરચના ggml_cgraph * cgraph, અચલ ggml_tensor * inp_pos);

bool get_is_prefill(સંરચના ggml_cgraph * cgraph, અચલ ggml_tensor * inp_pos);

bool is_naive(સંરચના ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(સંરચના ggml_cgraph * cgraph);
