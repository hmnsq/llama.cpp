#அந்தர்பாவ "ggml-decoder.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-openvino-extra.h"

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <functional>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <openvino/runtime/core.hpp>
#அந்தர்பாவ <openvino/runtime/infer_request.hpp>
#அந்தர்பாவ <string>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <utility>
#அந்தர்பாவ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
கட்டமைப்பு graph_சாவி {
    முழு n_nodes;
    முழு n_leaves;
    std::string first_முனை_name;
    std::string last_முனை_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> உள்ளீடு_srcs;

    graph_சாவி(மாறிலி ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        எனில் (n_nodes > 0) {
            first_முனை_name = cgraph->nodes[0]->name;
            last_முனை_name = cgraph->nodes[n_nodes - 1]->name;
        }

        நிலையான மாறிலி bool full_சாவி = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        எனில் (!full_சாவி && !include_inputs) {
            திரும்பு;
        }

        std::unordered_set<மாறிலி ggml_tensor *> முனை_set;
        முனை_set.reserve(cgraph->n_nodes);
        க்கு (முழு i = 0; i < cgraph->n_nodes; i++) {
            முனை_set.செருகு(cgraph->nodes[i]);
        }

        க்கு (முழு முனை_idx = 0; முனை_idx < cgraph->n_nodes; முனை_idx++) {
            மாறிலி ggml_tensor * முனை = cgraph->nodes[முனை_idx];
            க்கு (முழு src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                மாறிலி ggml_tensor * src = முனை->src[src_idx];
                எனில் (src == nullptr || src->buffer == nullptr || முனை_set.மொத்தம்(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    தொடர்;
                }

                உள்ளீடு_srcs.தள்ளு_back(std::to_string(முனை_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(முனை));
            }
        }
    }

    bool operator==(மாறிலி graph_சாவி & other) மாறிலி {
        திரும்பு n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_முனை_name == other.first_முனை_name &&
               last_முனை_name == other.last_முனை_name && உள்ளீடு_srcs == other.உள்ளீடு_srcs;
    }
};

கட்டமைப்பு graph_சாவி_hash {
    size_t operator()(மாறிலி graph_சாவி & சாவி) மாறிலி {
        size_t hash = std::hash<முழு>{}(சாவி.n_nodes);
        hash ^= std::hash<முழு>{}(சாவி.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        எனில் (சாவி.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(சாவி.first_முனை_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(சாவி.last_முனை_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        க்கு (மாறிலி auto & s : சாவி.உள்ளீடு_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        திரும்பு hash;
    }
};

கட்டமைப்பு decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

கட்டமைப்பு ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> உள்ளீடு_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
கட்டமைப்பு ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_மொத்தம் = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
கட்டமைப்பு ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_சாவி, std::shared_ptr<decoder_runtime_ctx>, graph_சாவி_hash> decoder_cache;
    std::unordered_map<graph_சாவி, std::shared_ptr<ov::InferRequest>, graph_சாவி_hash> infer_request_cache;
    std::unordered_map<graph_சாவி, std::shared_ptr<ov::InferRequest>, graph_சாவி_hash> infer_request_cache_prefill;
    std::unordered_map<graph_சாவி, std::vector<std::string>, graph_சாவி_hash> ov_உள்ளீடு_names_cache;
    std::unordered_map<graph_சாவி, std::vector<std::string>, graph_சாவி_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_உள்ளீடு_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    கட்டமைப்பு compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<மாறிலி ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    வெறுமை clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_உள்ளீடு_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_உள்ளீடு_name_map.clear();
        stateful_kv_size = 0;
    }

    வெறுமை clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

எண்ணிக்கை ggml_status ov_graph_compute(கட்டமைப்பு ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(மாறிலி வெறுமை * data, size_t size);

bool save_ggml_tensor_data_to_txt(மாறிலி ggml_tensor * tensor, மாறிலி std::string & file_path);

வெறுமை print_உள்ளீடு_tensor_info(மாறிலி std::string & name, மாறிலி ov::Tensor & tensor);

வெறுமை print_output_tensor_info(மாறிலி std::string & name, மாறிலி ov::Tensor & tensor, மாறிலி வெறுமை * output_dst);

template <typename T>
std::vector<T> pad_உள்ளீடு(மாறிலி T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_மதிப்பு) {
    std::vector<T> padded(padded_rows * padded_cols, pad_மதிப்பு);

    க்கு (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        க்கு (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    திரும்பு padded;
}

template <typename T>
std::vector<T> pad_உள்ளீடு(மாறிலி ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_மதிப்பு) {
    திரும்பு pad_உள்ளீடு<T>(reinterpret_cast<மாறிலி T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_மதிப்பு);
}

மாறிலி ggml_tensor * get_inp_pos_tensor(கட்டமைப்பு ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி ggml_tensor * inp_pos);

bool get_is_prefill(கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி ggml_tensor * inp_pos);

bool is_naive(கட்டமைப்பு ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(கட்டமைப்பு ggml_cgraph * cgraph);
