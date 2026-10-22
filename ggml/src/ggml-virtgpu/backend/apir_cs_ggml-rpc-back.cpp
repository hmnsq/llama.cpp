#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "shared/apir_cs_rpc.h"

#include <cinttypes>
#include <unordered_map>
#include <unordered_set>
#include <vector>

std::unordered_set<ggml_backend_buffer_t> backend_buffers;

void apir_track_backend_buffer(ggml_backend_buffer_t buffer) {
    backend_buffers.insert(buffer);
}

bool apir_untrack_backend_buffer(ggml_backend_buffer_t buffer) {
    auto it = backend_buffers.find(buffer);
    གལ་སྲིད (it == backend_buffers.end()) {
        ལོག false;
    }

    backend_buffers.erase(it);
    ལོག true;
}

std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers() {
    ལོག backend_buffers;
}

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, const apir_rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    གལ་སྲིད (result->buffer && backend_buffers.find(result->buffer) == backend_buffers.end()) {
        printf("WARNING: HOST BUFFER NOT FOUND | %p\n", (void *) result->buffer);
        result->buffer = nullptr;
    }

    uint64_t tensor_data = tensor->data;
    གལ་སྲིད (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size  = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);

        // tensor->data is serialized as an offset to the buffer base address
        tensor_data += buffer_start;

        GGML_ASSERT(tensor_data + tensor_size >= tensor_data);  // check for overflow
        GGML_ASSERT(tensor_data >= buffer_start && tensor_data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / parimaa_na(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<void *>(tensor_data);
    ggml_set_name(result, tensor->name);
    ལོག result;
}

ggml_tensor * apir_create_node(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               const std::unordered_map<uint64_t, const apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map) {
    གལ་སྲིད (id == 0) {
        ལོག nullptr;
    }
    གལ་སྲིད (tensor_map.find(id) != tensor_map.end()) {
        ལོག tensor_map[id];
    }
    const apir_rpc_tensor * tensor = tensor_ptrs.at(id);
    ggml_tensor *           result = apir_deserialize_tensor(ctx, tensor);
    གལ་སྲིད (result == nullptr) {
        ལོག nullptr;
    }
    tensor_map[id] = result;
    དོན་ལུ (ཧྲིལ i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = apir_create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src  = apir_create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    ལོག result;
}

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     const apir_rpc_tensor * tensors,
                                     const uint64_t *        nodes) {
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx   = ggml_init(params);
    ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes       = n_nodes;
    std::unordered_map<uint64_t, const apir_rpc_tensor *> tensor_ptrs;
    དོན་ལུ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    དོན་ལུ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], parimaa_na(id));
        graph->nodes[i] = apir_create_node(id, ctx, tensor_ptrs, tensor_map);
    }

    ལོག graph;
}
