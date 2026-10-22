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
بنية ګراف_کلی {
    صحيح n_nodes;
    صحيح n_leaves;
    std::string لومړی_غوټه_name;
    std::string وروستی_غوټه_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::ویکټور<std::string> ننوت_srcs;

    ګراف_کلی(ثابت ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        إذا (n_nodes > 0) {
            لومړی_غوټه_name = cgraph->nodes[0]->name;
            وروستی_غوټه_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ساكن ثابت bool ډک_کلی = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        إذا (!ډک_کلی && !include_inputs) {
            أرجع;
        }

        std::unordered_سېټ<ثابت ggml_tensor *> غوټه_سېټ;
        غوټه_سېټ.reserve(cgraph->n_nodes);
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            غوټه_سېټ.ورننباسه(cgraph->nodes[i]);
        }

        لكل (صحيح غوټه_idx = 0; غوټه_idx < cgraph->n_nodes; غوټه_idx++) {
            ثابت ggml_tensor * غوټه = cgraph->nodes[غوټه_idx];
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ثابت ggml_tensor * src = غوټه->src[src_idx];
                إذا (src == nullptr || src->بفر == nullptr || غوټه_سېټ.شمېر(src) || src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                    استمر;
                }

                ننوت_srcs.ټېل_شا(std::منزل_string(غوټه_idx) + ":" + std::منزل_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(غوټه));
            }
        }
    }

    bool چلوونکی==(ثابت ګراف_کلی & other) ثابت {
        أرجع n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               لومړی_غوټه_name == other.لومړی_غوټه_name &&
               وروستی_غوټه_name == other.وروستی_غوټه_name && ننوت_srcs == other.ننوت_srcs;
    }
};

بنية ګراف_کلی_hash {
    size_t چلوونکی()(ثابت ګراف_کلی & کلی) ثابت {
        size_t hash = std::hash<صحيح>{}(کلی.n_nodes);
        hash ^= std::hash<صحيح>{}(کلی.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        إذا (کلی.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(کلی.لومړی_غوټه_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(کلی.وروستی_غوټه_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        لكل (ثابت خپلکاری & s : کلی.ننوت_srcs) {
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

بنية ov_compiled_ګراف {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::ویکټور<std::string> ننوت_نومونه;
    std::ویکټور<std::string> output_نومونه;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
بنية ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_ګراف> graphs;
    size_t backend_شمېر = 0;
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
    std::unordered_map<ګراف_کلی, std::shared_ptr<decoder_runtime_ctx>, ګراف_کلی_hash> decoder_cache;
    std::unordered_map<ګراف_کلی, std::shared_ptr<ov::InferRequest>, ګراف_کلی_hash> infer_request_cache;
    std::unordered_map<ګراف_کلی, std::shared_ptr<ov::InferRequest>, ګراف_کلی_hash> infer_request_cache_prefill;
    std::unordered_map<ګراف_کلی, std::ویکټور<std::string>, ګراف_کلی_hash> ov_ننوت_نومونه_cache;
    std::unordered_map<ګراف_کلی, std::ویکټور<std::string>, ګراف_کلی_hash> ov_output_نومونه_cache;
    size_t stateful_kv_اندازه;
    std::map<std::string, std::string> kv_state_ننوت_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    بنية compute_ګراف {
        ggml_cgraph ګراف;
        std::ویکټور<ggml_tensor *> nodes;
    };
    std::unordered_map<ثابت ggml_cgraph *, compute_ګراف> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_اندازه(0) {}

    فراغ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_ننوت_نومونه_cache.clear();
        ov_output_نومونه_cache.clear();
        kv_state_ننوت_name_map.clear();
        stateful_kv_اندازه = 0;
    }

    فراغ clear_caches() {
        std::قلف_guard<std::mutex> قلف(ctx_mutex);
        clear_caches_locked();
    }
};

تعداد ggml_حالت ov_ګراف_compute(بنية ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ثابت فراغ * معلومات, size_t اندازه);

bool save_ggml_tensor_معلومات_منزل_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_لار);

فراغ print_ننوت_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor);

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst);

کينډۍ <typename T>
std::ویکټور<T> pad_ننوت(ثابت T * معلومات,
                         size_t لیکې,
                         size_t ستون_شمېر_2,
                         size_t padded_لیکې,
                         size_t padded_ستون_شمېر,
                         T pad_ارزښت) {
    std::ویکټور<T> padded(padded_لیکې * padded_ستون_شمېر, pad_ارزښت);

    لكل (size_t i = 0; i < std::min(لیکې, padded_لیکې); ++i) {
        لكل (size_t j = 0; j < std::min(ستون_شمېر_2, padded_ستون_شمېر); ++j) {
            padded[i * padded_ستون_شمېر + j] = معلومات[i * ستون_شمېر_2 + j];
        }
    }

    أرجع padded;
}

کينډۍ <typename T>
std::ویکټور<T> pad_ننوت(ثابت ggml_tensor * tensor, size_t padded_لیکې, size_t padded_ستون_شمېر, T pad_ارزښت) {
    أرجع pad_ننوت<T>(reinterpret_cast<ثابت T *>(tensor->معلومات),
                        static_cast<size_t>(tensor->ne[1]),  //لیکې
                        static_cast<size_t>(tensor->ne[0]),  //ستون_شمېر
                        padded_لیکې, padded_ستون_شمېر, pad_ارزښت);
}

ثابت ggml_tensor * get_inp_pos_tensor(بنية ggml_cgraph * cgraph);

int64_t get_inp_pos_n_نښې(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool get_is_prefill(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool is_naive(بنية ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(بنية ggml_cgraph * cgraph);
