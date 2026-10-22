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
بنية گراف_کلید {
    صحيح n_nodes;
    صحيح n_leaves;
    std::string پہلا_گرہ_name;
    std::string آخری_گرہ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::سمتیہ<std::string> داخلہ_srcs;

    گراف_کلید(ثابت ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        إذا (n_nodes > 0) {
            پہلا_گرہ_name = cgraph->nodes[0]->name;
            آخری_گرہ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ساكن ثابت bool بھریا_کلید = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        إذا (!بھریا_کلید && !include_inputs) {
            أرجع;
        }

        std::unordered_سیٹ<ثابت ggml_tensor *> گرہ_سیٹ;
        گرہ_سیٹ.reserve(cgraph->n_nodes);
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            گرہ_سیٹ.داخل_کرو(cgraph->nodes[i]);
        }

        لكل (صحيح گرہ_idx = 0; گرہ_idx < cgraph->n_nodes; گرہ_idx++) {
            ثابت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ثابت ggml_tensor * src = گرہ->src[src_idx];
                إذا (src == nullptr || src->بفر == nullptr || گرہ_سیٹ.گنتی(src) || src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                    استمر;
                }

                داخلہ_srcs.ڈالو_پٹھ(std::منزل_string(گرہ_idx) + ":" + std::منزل_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(گرہ));
            }
        }
    }

    bool عامل==(ثابت گراف_کلید & other) ثابت {
        أرجع n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               پہلا_گرہ_name == other.پہلا_گرہ_name &&
               آخری_گرہ_name == other.آخری_گرہ_name && داخلہ_srcs == other.داخلہ_srcs;
    }
};

بنية گراف_کلید_hash {
    size_t عامل()(ثابت گراف_کلید & کلید) ثابت {
        size_t hash = std::hash<صحيح>{}(کلید.n_nodes);
        hash ^= std::hash<صحيح>{}(کلید.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        إذا (کلید.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(کلید.پہلا_گرہ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(کلید.آخری_گرہ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        لكل (ثابت خودکار & s : کلید.داخلہ_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        أرجع hash;
    }
};

بنية decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

بنية ov_compiled_گراف {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::سمتیہ<std::string> داخلہ_ناں;
    std::سمتیہ<std::string> output_ناں;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
بنية ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_گراف> graphs;
    size_t backend_گنتی = 0;
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
    std::unordered_map<گراف_کلید, std::shared_ptr<decoder_runtime_ctx>, گراف_کلید_hash> decoder_cache;
    std::unordered_map<گراف_کلید, std::shared_ptr<ov::InferRequest>, گراف_کلید_hash> infer_request_cache;
    std::unordered_map<گراف_کلید, std::shared_ptr<ov::InferRequest>, گراف_کلید_hash> infer_request_cache_prefill;
    std::unordered_map<گراف_کلید, std::سمتیہ<std::string>, گراف_کلید_hash> ov_داخلہ_ناں_cache;
    std::unordered_map<گراف_کلید, std::سمتیہ<std::string>, گراف_کلید_hash> ov_output_ناں_cache;
    size_t stateful_kv_ناپ;
    std::map<std::string, std::string> kv_state_داخلہ_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    بنية compute_گراف {
        ggml_cgraph گراف;
        std::سمتیہ<ggml_tensor *> nodes;
    };
    std::unordered_map<ثابت ggml_cgraph *, compute_گراف> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_ناپ(0) {}

    فراغ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_داخلہ_ناں_cache.clear();
        ov_output_ناں_cache.clear();
        kv_state_داخلہ_name_map.clear();
        stateful_kv_ناپ = 0;
    }

    فراغ clear_caches() {
        std::جندرا_guard<std::mutex> جندرا(ctx_mutex);
        clear_caches_locked();
    }
};

تعداد ggml_حالت ov_گراف_compute(بنية ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ثابت فراغ * انکڑے, size_t ناپ);

bool save_ggml_tensor_انکڑے_منزل_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_رستہ);

فراغ print_داخلہ_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor);

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst);

سانچا <typename T>
std::سمتیہ<T> pad_داخلہ(ثابت T * انکڑے,
                         size_t صفاں,
                         size_t خانے_گنتی_2,
                         size_t padded_صفاں,
                         size_t padded_خانے_گنتی,
                         T pad_قدر) {
    std::سمتیہ<T> padded(padded_صفاں * padded_خانے_گنتی, pad_قدر);

    لكل (size_t i = 0; i < std::min(صفاں, padded_صفاں); ++i) {
        لكل (size_t j = 0; j < std::min(خانے_گنتی_2, padded_خانے_گنتی); ++j) {
            padded[i * padded_خانے_گنتی + j] = انکڑے[i * خانے_گنتی_2 + j];
        }
    }

    أرجع padded;
}

سانچا <typename T>
std::سمتیہ<T> pad_داخلہ(ثابت ggml_tensor * tensor, size_t padded_صفاں, size_t padded_خانے_گنتی, T pad_قدر) {
    أرجع pad_داخلہ<T>(reinterpret_cast<ثابت T *>(tensor->انکڑے),
                        static_cast<size_t>(tensor->ne[1]),  //صفاں
                        static_cast<size_t>(tensor->ne[0]),  //خانے_گنتی
                        padded_صفاں, padded_خانے_گنتی, pad_قدر);
}

ثابت ggml_tensor * get_inp_pos_tensor(بنية ggml_cgraph * cgraph);

int64_t get_inp_pos_n_ٹوکن(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool get_is_prefill(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool is_naive(بنية ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(بنية ggml_cgraph * cgraph);
