#ضمن "backend/shared/apir_cs_rpc.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-remoting.h"

#ضمن <cinttypes>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

apir_rpc_tensor apir_serialize_tensor(ثابت ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    𐡄𐡍 (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } وإلا {
        result.buffer = 0;
    }
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    لكل (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    𐡄𐡍 (tensor->data) {
        𐡄𐡍 (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    أرجع result;
}

فراغ apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    𐡄𐡍 (tensor == nullptr) {
        أرجع;
    }
    𐡄𐡍 (visited.find(tensor) != visited.end()) {
        أرجع;
    }
    visited.insert(tensor);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

فراغ apir_serialize_graph(ثابت ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    صحيح      output_size =
        حجم(uint32_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, حجم(n_nodes));
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + حجم(n_nodes) + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + حجم(n_nodes) + n_nodes * حجم(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + حجم(n_nodes) + n_nodes * حجم(uint64_t) + حجم(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * حجم(apir_rpc_tensor));
}
