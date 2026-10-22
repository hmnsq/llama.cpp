#ضمن "ggml-decoder.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"

#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <functional>
#ضمن <memory>
#ضمن <mutex>
#ضمن <openvino/runtime/core.hpp>
#ضمن <openvino/runtime/infer_request.hpp>
#ضمن <string>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <utility>
#ضمن <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
بنية graph_مفتاح {
    صحيح n_nodes;
    صحيح n_leaves;
    std::string first_عقدة_name;
    std::string last_عقدة_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> إدخال_srcs;

    graph_مفتاح(ثابت ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        إذا (n_nodes > 0) {
            first_عقدة_name = cgraph->nodes[0]->name;
            last_عقدة_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ساكن ثابت bool full_مفتاح = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        إذا (!full_مفتاح && !include_inputs) {
            إرجاع;
        }

        std::unordered_set<ثابت ggml_tensor *> عقدة_set;
        عقدة_set.reserve(cgraph->n_nodes);
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            عقدة_set.أدرج(cgraph->nodes[i]);
        }

        لكل (صحيح عقدة_idx = 0; عقدة_idx < cgraph->n_nodes; عقدة_idx++) {
            ثابت ggml_tensor * عقدة = cgraph->nodes[عقدة_idx];
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ثابت ggml_tensor * src = عقدة->src[src_idx];
                إذا (src == nullptr || src->buffer == nullptr || عقدة_set.تعداد_2(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    استمر;
                }

                إدخال_srcs.ادفع_back(std::to_string(عقدة_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(عقدة));
            }
        }
    }

    bool operator==(ثابت graph_مفتاح & other) ثابت {
        إرجاع n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_عقدة_name == other.first_عقدة_name &&
               last_عقدة_name == other.last_عقدة_name && إدخال_srcs == other.إدخال_srcs;
    }
};

بنية graph_مفتاح_hash {
    size_t operator()(ثابت graph_مفتاح & مفتاح) ثابت {
        size_t hash = std::hash<صحيح>{}(مفتاح.n_nodes);
        hash ^= std::hash<صحيح>{}(مفتاح.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        إذا (مفتاح.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(مفتاح.first_عقدة_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(مفتاح.last_عقدة_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        لكل (ثابت auto & s : مفتاح.إدخال_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        إرجاع hash;
    }
};

بنية decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

بنية ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> إدخال_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
بنية ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_تعداد = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
بنية ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_مفتاح, std::shared_ptr<decoder_runtime_ctx>, graph_مفتاح_hash> decoder_cache;
    std::unordered_map<graph_مفتاح, std::shared_ptr<ov::InferRequest>, graph_مفتاح_hash> infer_request_cache;
    std::unordered_map<graph_مفتاح, std::shared_ptr<ov::InferRequest>, graph_مفتاح_hash> infer_request_cache_prefill;
    std::unordered_map<graph_مفتاح, std::vector<std::string>, graph_مفتاح_hash> ov_إدخال_names_cache;
    std::unordered_map<graph_مفتاح, std::vector<std::string>, graph_مفتاح_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_إدخال_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    بنية compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<ثابت ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    فراغ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_إدخال_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_إدخال_name_map.clear();
        stateful_kv_size = 0;
    }

    فراغ clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

تعداد ggml_status ov_graph_compute(بنية ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ثابت فراغ * data, size_t size);

bool save_ggml_tensor_data_to_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_path);

فراغ print_إدخال_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor);

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst);

template <typename T>
std::vector<T> pad_إدخال(ثابت T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_قيمة) {
    std::vector<T> padded(padded_rows * padded_cols, pad_قيمة);

    لكل (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        لكل (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    إرجاع padded;
}

template <typename T>
std::vector<T> pad_إدخال(ثابت ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_قيمة) {
    إرجاع pad_إدخال<T>(reinterpret_cast<ثابت T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_قيمة);
}

ثابت ggml_tensor * get_inp_pos_tensor(بنية ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool get_is_prefill(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool is_naive(بنية ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(بنية ggml_cgraph * cgraph);
