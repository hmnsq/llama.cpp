#अन्तर्भाव "backend/shared/apir_cs_rpc.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <cinttypes>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <vector>

apir_rpc_tensor apir_serialize_tensor(स्थिरांक ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    जर (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } अन्यथा {
        result.buffer = 0;
    }
    साठी (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    साठी (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / आकार(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    साठी (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    जर (tensor->data) {
        जर (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    परत result;
}

शून्य apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    जर (tensor == nullptr) {
        परत;
    }
    जर (visited.find(tensor) != visited.end()) {
        परत;
    }
    visited.insert(tensor);
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

शून्य apir_serialize_graph(स्थिरांक ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    साठी (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    पूर्णांक      output_size =
        आकार(uint32_t) + n_nodes * आकार(uint64_t) + आकार(uint32_t) + n_tensors * आकार(apir_rpc_tensor);
    output.resize(output_size, 0);
    स_नकल(output.data(), &n_nodes, आकार(n_nodes));
    साठी (uint32_t i = 0; i < n_nodes; i++) {
        स_नकल(output.data() + आकार(n_nodes) + i * आकार(uint64_t), &cgraph->nodes[i], आकार(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + आकार(n_nodes) + n_nodes * आकार(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + आकार(n_nodes) + n_nodes * आकार(uint64_t) + आकार(uint32_t));
    स_नकल(out_tensors, tensors.data(), n_tensors * आकार(apir_rpc_tensor));
}
