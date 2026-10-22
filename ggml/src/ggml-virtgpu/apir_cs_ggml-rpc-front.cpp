#ಅನ್ತರ್ಭಾವ "backend/shared/apir_cs_rpc.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

apir_rpc_tensor apir_serialize_tensor(ಧ್ರುವ ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ಯದಿ (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } ಅನ್ಯಥಾ {
        result.buffer = 0;
    }
    ಪ್ರತಿ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    ಪ್ರತಿ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    ಪ್ರತಿ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    ಯದಿ (tensor->data) {
        ಯದಿ (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ಪ್ರತ್ಯಾಗಮ result;
}

ಶೂನ್ಯ apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    ಯದಿ (tensor == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (visited.find(tensor) != visited.end()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    visited.insert(tensor);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

ಶೂನ್ಯ apir_serialize_graph(ಧ್ರುವ ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    ಪ್ರತಿ (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    ಪೂರ್ಣಾಙ್ಕ      output_size =
        ಪರಿಮಾಣ(uint32_t) + n_nodes * ಪರಿಮಾಣ(uint64_t) + ಪರಿಮಾಣ(uint32_t) + n_tensors * ಪರಿಮಾಣ(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, ಪರಿಮಾಣ(n_nodes));
    ಪ್ರತಿ (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + ಪರಿಮಾಣ(n_nodes) + i * ಪರಿಮಾಣ(uint64_t), &cgraph->nodes[i], ಪರಿಮಾಣ(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + ಪರಿಮಾಣ(n_nodes) + n_nodes * ಪರಿಮಾಣ(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + ಪರಿಮಾಣ(n_nodes) + n_nodes * ಪರಿಮಾಣ(uint64_t) + ಪರಿಮಾಣ(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * ಪರಿಮಾಣ(apir_rpc_tensor));
}
