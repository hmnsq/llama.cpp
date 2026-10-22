#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-rpc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "transport.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "log.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <chrono>
#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <queue>
#ਅਨ੍ਤਰ੍ਭਾਵ <condition_variable>
#ਅਨ੍ਤਰ੍ਭਾਵ <future>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <filesystem>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>

ਨਾਮਾਕਾਸ਼ fs = std::filesystem;

// macro for nicer error messages on server crash
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RPC_ਸਥਿਤੀ_ASSERT(x) ਜੇ (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#ਹਦਾਇਤ pack(ਧੱਕੋ, 1)
// ggml_tensor is serialized into rpc_tensor
ਬਣਤਰ rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t ਬਫਰ;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t ਅੰਕੜੇ;
    ਅੱਖਰ name[GGML_MAX_NAME];

    int32_t use_ਗਿਣਤੀ;
};

static_assert(ਆਕਾਰ(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
ਗਿਣਤੀ rpc_cmd {
    RPC_CMD_ALLOC_ਬਫਰ = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_ਮਾਪ,
    RPC_CMD_ਬਫਰ_GET_ਅਧਾਰ,
    RPC_CMD_FREE_ਬਫਰ,
    RPC_CMD_ਬਫਰ_CLEAR,
    RPC_CMD_ਸੈੱਟ_TENSOR,
    RPC_CMD_ਸੈੱਟ_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_ਗ੍ਰਾਫ_COMPUTE,
    RPC_CMD_GET_DEVICE_ਸਮ੍ਰਿਤੀ,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_ਮਾਪ,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_ਗਿਣਤੀ,
    RPC_CMD_ਗ੍ਰਾਫ_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_ਸੈੱਟ_TENSOR_2_D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_ਗਿਣਤੀ,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * rpc_cmd_name(ਗਿਣਤੀ rpc_cmd cmd) {
    ਚੋਣ (cmd) {
        ਹਾਲਤ RPC_CMD_ALLOC_ਬਫਰ:      ਵਾਪਸ "ALLOC_BUFFER";
        ਹਾਲਤ RPC_CMD_GET_ALIGNMENT:     ਵਾਪਸ "GET_ALIGNMENT";
        ਹਾਲਤ RPC_CMD_GET_MAX_ਮਾਪ:      ਵਾਪਸ "GET_MAX_SIZE";
        ਹਾਲਤ RPC_CMD_ਬਫਰ_GET_ਅਧਾਰ:   ਵਾਪਸ "BUFFER_GET_BASE";
        ਹਾਲਤ RPC_CMD_FREE_ਬਫਰ:       ਵਾਪਸ "FREE_BUFFER";
        ਹਾਲਤ RPC_CMD_ਬਫਰ_CLEAR:      ਵਾਪਸ "BUFFER_CLEAR";
        ਹਾਲਤ RPC_CMD_ਸੈੱਟ_TENSOR:        ਵਾਪਸ "SET_TENSOR";
        ਹਾਲਤ RPC_CMD_ਸੈੱਟ_TENSOR_HASH:   ਵਾਪਸ "SET_TENSOR_HASH";
        ਹਾਲਤ RPC_CMD_GET_TENSOR:        ਵਾਪਸ "GET_TENSOR";
        ਹਾਲਤ RPC_CMD_COPY_TENSOR:       ਵਾਪਸ "COPY_TENSOR";
        ਹਾਲਤ RPC_CMD_ਗ੍ਰਾਫ_COMPUTE:     ਵਾਪਸ "GRAPH_COMPUTE";
        ਹਾਲਤ RPC_CMD_GET_DEVICE_ਸਮ੍ਰਿਤੀ: ਵਾਪਸ "GET_DEVICE_MEMORY";
        ਹਾਲਤ RPC_CMD_INIT_TENSOR:       ਵਾਪਸ "INIT_TENSOR";
        ਹਾਲਤ RPC_CMD_GET_ALLOC_ਮਾਪ:    ਵਾਪਸ "GET_ALLOC_SIZE";
        ਹਾਲਤ RPC_CMD_HELLO:             ਵਾਪਸ "HELLO";
        ਹਾਲਤ RPC_CMD_DEVICE_ਗਿਣਤੀ:      ਵਾਪਸ "DEVICE_COUNT";
        ਹਾਲਤ RPC_CMD_ਗ੍ਰਾਫ_RECOMPUTE:   ਵਾਪਸ "GRAPH_RECOMPUTE";
        ਹਾਲਤ RPC_CMD_MEMSET_TENSOR:     ਵਾਪਸ "MEMSET_TENSOR";
        ਹਾਲਤ RPC_CMD_ਸੈੱਟ_TENSOR_2_D:     ਵਾਪਸ "SET_TENSOR_2D";
        ਹਾਲਤ RPC_CMD_GET_TENSOR_2D:     ਵਾਪਸ "GET_TENSOR_2D";
        ਹਾਲਤ RPC_CMD_COMM_INIT:         ਵਾਪਸ "COMM_INIT";
        ਹਾਲਤ RPC_CMD_COMM_ALLREDUCE:    ਵਾਪਸ "COMM_ALLREDUCE";
        ਹਾਲਤ RPC_CMD_COMM_FREE:         ਵਾਪਸ "COMM_FREE";
        ਹਾਲਤ RPC_CMD_NONE:              ਵਾਪਸ "NONE";
        ਮੂਲ:                        ਵਾਪਸ "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ਅਚਲ size_t HASH_ਹੱਦ = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ਅਚਲ size_t ਗ੍ਰਾਫ_CACHE_MAX = 1024;

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_ਮਾਪ];
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_ਮਾਪ];
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_device_ਗਿਣਤੀ_rsp {
    uint32_t device_ਗਿਣਤੀ;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_rsp {
    uint64_t alloc_ਮਾਪ;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_init_tensor_req {
    rpc_tensor tensor;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_req {
    uint32_t device;
    uint64_t size;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_rsp {
    uint64_t remote_ptr;
    uint64_t remote_ਮਾਪ;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_alignment_req {
    uint32_t device;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_alignment_rsp {
    uint64_t alignment;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_req {
    uint32_t device;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_rsp {
    uint64_t max_size;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_req {
    uint64_t remote_ptr;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_rsp {
    uint64_t ਅਧਾਰ_ptr;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_free_ਬਫਰ_req {
    uint64_t remote_ptr;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_ਬਫਰ_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t ਵਿੱਥ;
    uint64_t size;
    uint8_t value;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t ਵਿੱਥ;
    uint64_t hash;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_rsp {
    uint8_t ਨਤੀਜਾ;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_tensor_req {
    rpc_tensor tensor;
    uint64_t ਵਿੱਥ;
    uint64_t size;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_copy_tensor_rsp {
    uint8_t ਨਤੀਜਾ;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_req {
    uint32_t device;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_rsp {
    uint64_t free_mem;
    uint64_t ਕੁੱਲ_mem;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_ਗ੍ਰਾਫ_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t ਵਿੱਥ;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t ਪੋਰਟ;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    ਅੱਖਰ     host[64];  // rank > 0: rank 0's host
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_comm_init_rsp {
    uint8_t ok;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ਬਣਤਰ rpc_ਸੁਨੇਹਾ_comm_free_req {
    uint32_t device;
};

#ਹਦਾਇਤ pack(ਕੱਢੋ)

// RPC data structures

ਸਥਿਰ ggml_guid_t ggml_backend_rpc_guid() {
    ਸਥਿਰ ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ਵਾਪਸ &guid;
}

ਬਣਤਰ ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ਬਣਤਰ ggml_backend_rpc_ਬਫਰ_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

ਵਰਗ rpc_dispatcher;
ਬਣਤਰ ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ਬਣਤਰ ggml_backend_rpc_ਬਫਰ_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    ਖਾਲੀ                            * ਅਧਾਰ_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ਸਥਿਰ uint64_t fnv_hash(ਅਚਲ uint8_t * ਅੰਕੜੇ, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ਅਚਲ uint64_t fnv_ਅਭਾਜ = 0x100000001b3ULL;

    ਲਈ (size_t i = 0; i < len; ++i) {
        hash ^= ਅੰਕੜੇ[i];
        hash *= fnv_ਅਭਾਜ;
    }
    ਵਾਪਸ hash;
}

ਸਥਿਰ ਤਾਰਕਿਕ send_ਸੁਨੇਹਾ(socket_ptr sock, ਅਚਲ ਖਾਲੀ * ਸੁਨੇਹਾ_2, size_t ਸੁਨੇਹਾ_ਮਾਪ) {
    ਜੇ (!sock->send_ਅੰਕੜੇ(&ਸੁਨੇਹਾ_ਮਾਪ, ਆਕਾਰ(ਸੁਨੇਹਾ_ਮਾਪ))) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!sock->send_ਅੰਕੜੇ(ਸੁਨੇਹਾ_2, ਸੁਨੇਹਾ_ਮਾਪ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ sock->flush();
}

ਸਥਿਰ ਤਾਰਕਿਕ recv_ਸੁਨੇਹਾ(socket_ptr sock, ਖਾਲੀ * ਸੁਨੇਹਾ_2, size_t ਸੁਨੇਹਾ_ਮਾਪ) {
    uint64_t size;
    ਜੇ (!sock->recv_ਅੰਕੜੇ(&size, ਆਕਾਰ(size))) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (size != ਸੁਨੇਹਾ_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ sock->recv_ਅੰਕੜੇ(ਸੁਨੇਹਾ_2, ਸੁਨੇਹਾ_ਮਾਪ);
}

ਸਥਿਰ ਤਾਰਕਿਕ recv_ਸੁਨੇਹਾ(socket_ptr sock, std::vector<uint8_t> & ਨਿਵੇਸ਼) {
    uint64_t size;
    ਜੇ (!sock->recv_ਅੰਕੜੇ(&size, ਆਕਾਰ(size))) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਕੋਸ਼ਿਸ਼ {
        ਨਿਵੇਸ਼.resize(size);
    } ਫੜੋ (ਅਚਲ std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ sock->recv_ਅੰਕੜੇ(ਨਿਵੇਸ਼.ਅੰਕੜੇ(), size);
}

ਸਥਿਰ ਤਾਰਕਿਕ parse_endpoint(ਅਚਲ std::string & endpoint, std::string & host, ਪੂਰਨ & ਪੋਰਟ) {
    size_t pos = endpoint.find(':');
    ਜੇ (pos == std::string::npos) {
        ਵਾਪਸ ਝੂਠ;
    }
    host = endpoint.substr(0, pos);
    ਕੋਸ਼ਿਸ਼ {
        ਪੋਰਟ = std::stoi(endpoint.substr(pos + 1));
    } ਫੜੋ (...) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ਸਥਿਰ ਤਾਰਕਿਕ send_rpc_cmd(socket_ptr sock, ਗਿਣਤੀ rpc_cmd cmd, ਅਚਲ ਖਾਲੀ * ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ) {
    uint8_t cmd_byte = cmd;
    ਜੇ (!sock->send_ਅੰਕੜੇ(&cmd_byte, ਆਕਾਰ(cmd_byte))) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!sock->send_ਅੰਕੜੇ(&ਨਿਵੇਸ਼_ਮਾਪ, ਆਕਾਰ(ਨਿਵੇਸ਼_ਮਾਪ))) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!sock->send_ਅੰਕੜੇ(ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_ਮਾਪ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ਸਥਿਰ ਤਾਰਕਿਕ send_rpc_cmd(socket_ptr sock, ਗਿਣਤੀ rpc_cmd cmd, ਅਚਲ ਖਾਲੀ * ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ, ਖਾਲੀ * output, size_t output_ਮਾਪ) {
    ਜੇ (!send_rpc_cmd(sock, cmd, ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_ਮਾਪ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    uint64_t ਬਾਹਰਲਾ_ਮਾਪ;
    ਜੇ (!sock->recv_ਅੰਕੜੇ(&ਬਾਹਰਲਾ_ਮਾਪ, ਆਕਾਰ(ਬਾਹਰਲਾ_ਮਾਪ))) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ਬਾਹਰਲਾ_ਮਾਪ != output_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!sock->recv_ਅੰਕੜੇ(output, output_ਮਾਪ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ਸਥਿਰ constexpr ਸਵੈਚਾਲਿਤ RPC_BUSY_SPIN_IDLE_ਸਮਾਂ = std::chrono::milliseconds(100);

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rpc_cpu_relax() {
#ਜੇ defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ ਅਸਥਿਰ("yield" ::: "memory");
#ਨਹੀਂਤਾਂ
    std::this_thread::yield();
#ਜੇ_ਅੰਤ
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ਸਥਿਰ ਤਾਰਕਿਕ negotiate_hello(ਅਚਲ std::shared_ptr<socket_t> & sock) {
    rpc_ਸੁਨੇਹਾ_hello_req request = {};
    rpc_ਸੁਨੇਹਾ_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    ਤਾਰਕਿਕ ਸਥਿਤੀ = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, ਆਕਾਰ(request), &response, ਆਕਾਰ(response));
    RPC_ਸਥਿਤੀ_ASSERT(ਸਥਿਤੀ);

    ਜੇ (response.major != RPC_PROTO_MAJOR_ਸੰਸਕਰਣ || response.minor > RPC_PROTO_MINOR_ਸੰਸਕਰਣ) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ਵਾਪਸ ਝੂਠ;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ਵਾਪਸ ਸੱਚ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਵਰਗ ਸੁਨੇਹਾ_ਕਤਾਰ {
ਜਨਤਕ:
    ਸੁਨੇਹਾ_ਕਤਾਰ() {}

    ਤਾਰਕਿਕ ਧੱਕੋ(ਅਚਲ T &value) {
        std::unique_lock<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (interrupted) {
            ਵਾਪਸ ਝੂਠ;
        }
        queue.ਧੱਕੋ(value);
        cvar.notify_all();
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ ਕੱਢੋ(T* ਬਾਹਰਲਾ) {
        std::unique_lock<std::mutex> ਤਾਲਾ(mutex);
        cvar.wait(ਤਾਲਾ, [ਇਹ] { ਵਾਪਸ !queue.ਖਾਲੀ_ਥਾਵਾਂ() || interrupted; });
        ਜੇ (interrupted) {
            ਵਾਪਸ ਝੂਠ;
        }
        *ਬਾਹਰਲਾ = queue.front();
        queue.ਕੱਢੋ();
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ try_ਕੱਢੋ(T* ਬਾਹਰਲਾ) {
        std::unique_lock<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (interrupted || queue.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਵਾਪਸ ਝੂਠ;
        }
        *ਬਾਹਰਲਾ = queue.front();
        queue.ਕੱਢੋ();
        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ ਰੁਕਾਵਟ() {
        std::unique_lock<std::mutex> ਤਾਲਾ(mutex);
        interrupted = ਸੱਚ;
        ਤਾਲਾ.unlock();
        cvar.notify_all();
    }

ਨਿਜੀ:
    ਤਾਰਕਿਕ interrupted = ਝੂਠ;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

ਵਰਗ rpc_dispatcher {
ਜਨਤਕ:
    rpc_dispatcher() {
    }

    ਖਾਲੀ send(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ);
    ਖਾਲੀ send(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ, ਖਾਲੀ * output, size_t output_ਮਾਪ);
    ਖਾਲੀ send_async(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ);
    ਖਾਲੀ send_async(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ, ਖਾਲੀ * output, size_t output_ਮਾਪ);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    ਖਾਲੀ event_free(ggml_backend_event_t event);
    ਖਾਲੀ event_synchronize(ggml_backend_event_t event);
    ਖਾਲੀ event_ਰਿਕਾਰਡ(ggml_backend_event_t event);
    ਖਾਲੀ synchronize();
    ਖਾਲੀ busy_spin_acquire();
    ਖਾਲੀ busy_spin_release();
    ਖਾਲੀ ਗ੍ਰਾਫ_compute(uint32_t device, ਅਚਲ ggml_cgraph * cgraph);

    ਖਾਲੀ start(ਅਚਲ std::string & endpoint);
    ਖਾਲੀ work();

    ~rpc_dispatcher();

ਨਿਜੀ:
    ਬਣਤਰ rpc_ਸੁਨੇਹਾ {
        rpc_cmd                       cmd;
        std::shared_ptr<ਅਚਲ ਖਾਲੀ>   ਨਿਵੇਸ਼;
        size_t                        ਨਿਵੇਸ਼_ਮਾਪ;
        ਖਾਲੀ                        * output;
        size_t                        output_ਮਾਪ;
        std::promise<ਖਾਲੀ>            completion;
    };
    ਵਰਤੋਂ rpc_ਸੁਨੇਹਾ_ptr   = std::shared_ptr<rpc_ਸੁਨੇਹਾ>;
    ਵਰਤੋਂ rpc_ਸੁਨੇਹਾ_ਕਤਾਰ = ਸੁਨੇਹਾ_ਕਤਾਰ<rpc_ਸੁਨੇਹਾ_ptr>;
    ਬਣਤਰ rpc_event {
        rpc_ਸੁਨੇਹਾ_ptr              ਸੁਨੇਹਾ_2;
        std::shared_future<ਖਾਲੀ> sf;
    };
    std::mutex ਗ੍ਰਾਫ_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> ਗ੍ਰਾਫ_uids;
    rpc_ਸੁਨੇਹਾ_ਕਤਾਰ    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ਸਥਿਰ ਖਾਲੀ rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

ਖਾਲੀ rpc_dispatcher::send(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ) {
    ਸਵੈਚਾਲਿਤ ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ਸੁਨੇਹਾ_2->cmd = cmd;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼ = ਨਿਵੇਸ਼;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼_ਮਾਪ = ਨਿਵੇਸ਼_ਮਾਪ;
    ਸੁਨੇਹਾ_2->output = ਖਾਲੀ_ਸੂਚਕ;
    ਸੁਨੇਹਾ_2->output_ਮਾਪ = 0;
    GGML_ASSERT(queue.ਧੱਕੋ(ਸੁਨੇਹਾ_2));
    ਸਵੈਚਾਲਿਤ future = ਸੁਨੇਹਾ_2->completion.get_future();
    future.wait();
}

ਖਾਲੀ rpc_dispatcher::send_async(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ) {
    ਸਵੈਚਾਲਿਤ ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ਸੁਨੇਹਾ_2->cmd = cmd;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼ = ਨਿਵੇਸ਼;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼_ਮਾਪ = ਨਿਵੇਸ਼_ਮਾਪ;
    ਸੁਨੇਹਾ_2->output = ਖਾਲੀ_ਸੂਚਕ;
    ਸੁਨੇਹਾ_2->output_ਮਾਪ = 0;
    GGML_ASSERT(queue.ਧੱਕੋ(ਸੁਨੇਹਾ_2));
}

ਖਾਲੀ rpc_dispatcher::send(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ, ਖਾਲੀ * output, size_t output_ਮਾਪ) {
    ਸਵੈਚਾਲਿਤ ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ਸੁਨੇਹਾ_2->cmd = cmd;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼ = ਨਿਵੇਸ਼;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼_ਮਾਪ = ਨਿਵੇਸ਼_ਮਾਪ;
    ਸੁਨੇਹਾ_2->output = output;
    ਸੁਨੇਹਾ_2->output_ਮਾਪ = output_ਮਾਪ;
    GGML_ASSERT(queue.ਧੱਕੋ(ਸੁਨੇਹਾ_2));
    ਸਵੈਚਾਲਿਤ future = ਸੁਨੇਹਾ_2->completion.get_future();
    future.wait();
}

ਖਾਲੀ rpc_dispatcher::send_async(ਗਿਣਤੀ rpc_cmd cmd, std::shared_ptr<ਅਚਲ ਖਾਲੀ> ਨਿਵੇਸ਼, size_t ਨਿਵੇਸ਼_ਮਾਪ, ਖਾਲੀ * output, size_t output_ਮਾਪ) {
    ਸਵੈਚਾਲਿਤ ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ਸੁਨੇਹਾ_2->cmd = cmd;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼ = ਨਿਵੇਸ਼;
    ਸੁਨੇਹਾ_2->ਨਿਵੇਸ਼_ਮਾਪ = ਨਿਵੇਸ਼_ਮਾਪ;
    ਸੁਨੇਹਾ_2->output = output;
    ਸੁਨੇਹਾ_2->output_ਮਾਪ = output_ਮਾਪ;
    GGML_ASSERT(queue.ਧੱਕੋ(ਸੁਨੇਹਾ_2));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = ਨਵਾਂ rpc_event;
    ev->ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ev->ਸੁਨੇਹਾ_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->ਸੁਨੇਹਾ_2->completion.get_future().share();
    GGML_ASSERT(queue.ਧੱਕੋ(ev->ਸੁਨੇਹਾ_2));
    ਵਾਪਸ ਨਵਾਂ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

ਖਾਲੀ rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ਮਿਟਾਓ ev;
}

ਖਾਲੀ rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

ਖਾਲੀ rpc_dispatcher::event_ਰਿਕਾਰਡ(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ev->ਸੁਨੇਹਾ_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->ਸੁਨੇਹਾ_2->completion.get_future().share();
    GGML_ASSERT(queue.ਧੱਕੋ(ev->ਸੁਨੇਹਾ_2));
}

ਖਾਲੀ rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    ਸਵੈਚਾਲਿਤ ਸੁਨੇਹਾ_2 = std::make_shared<rpc_ਸੁਨੇਹਾ>();
    ਸੁਨੇਹਾ_2->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ਧੱਕੋ(ਸੁਨੇਹਾ_2));
    ਸੁਨੇਹਾ_2->completion.get_future().wait();
}

ਖਾਲੀ rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

ਖਾਲੀ rpc_dispatcher::busy_spin_release() {
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪਿਛਲਾ = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(ਪਿਛਲਾ > 0);
}

ਖਾਲੀ rpc_dispatcher::start(ਅਚਲ std::string & endpoint) {
    std::string host;
    ਪੂਰਨ ਪੋਰਟ;
    ਜੇ (!parse_endpoint(endpoint, host, ਪੋਰਟ)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    ਜੇ (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), ਪੋਰਟ);
    ਜੇ (sock == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    ਜੇ (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = ਸੱਚ;
    thread = std::thread(rpc_dispatcher_trampoline, ਇਹ);
}

ਖਾਲੀ rpc_dispatcher::work() {
    ਸਵੈਚਾਲਿਤ ਆਖਰੀ_cmd = std::chrono::steady_clock::now();
    ਜਦੋਂ (running) {
        rpc_ਸੁਨੇਹਾ_ptr ਸੁਨੇਹਾ_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ਅਚਲ ਤਾਰਕਿਕ spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - ਆਖਰੀ_cmd < RPC_BUSY_SPIN_IDLE_ਸਮਾਂ;
        ਜੇ (spin) {
            ਜੇ (!queue.try_ਕੱਢੋ(&ਸੁਨੇਹਾ_ptr)) {
                rpc_cpu_relax();
                ਜਾਰੀ;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (!queue.ਕੱਢੋ(&ਸੁਨੇਹਾ_ptr)) {
            ਰੋਕੋ;
        }
        ਜੇ (ਸੁਨੇਹਾ_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(ਸੁਨੇਹਾ_ptr->cmd),
                     rpc_log_format_ਮਾਪ(ਸੁਨੇਹਾ_ptr->ਨਿਵੇਸ਼_ਮਾਪ).c_str(), rpc_log_format_ਮਾਪ(ਸੁਨੇਹਾ_ptr->output_ਮਾਪ).c_str());
            ਅਚਲ ਤਾਰਕਿਕ trace = rpc_log_debug_level() >= 3;
            ਅਚਲ ਸਵੈਚਾਲਿਤ t0 = std::chrono::steady_clock::now();
            ਜੇ (ਸੁਨੇਹਾ_ptr->output) {
                ਤਾਰਕਿਕ ਸਥਿਤੀ = send_rpc_cmd(sock, ਸੁਨੇਹਾ_ptr->cmd, ਸੁਨੇਹਾ_ptr->ਨਿਵੇਸ਼.get(), ਸੁਨੇਹਾ_ptr->ਨਿਵੇਸ਼_ਮਾਪ, ਸੁਨੇਹਾ_ptr->output, ਸੁਨੇਹਾ_ptr->output_ਮਾਪ);
                RPC_ਸਥਿਤੀ_ASSERT(ਸਥਿਤੀ);
            } ਨਹੀਂਤਾਂ {
                ਤਾਰਕਿਕ ਸਥਿਤੀ = send_rpc_cmd(sock, ਸੁਨੇਹਾ_ptr->cmd, ਸੁਨੇਹਾ_ptr->ਨਿਵੇਸ਼.get(), ਸੁਨੇਹਾ_ptr->ਨਿਵੇਸ਼_ਮਾਪ);
                RPC_ਸਥਿਤੀ_ASSERT(ਸਥਿਤੀ);
            }
            ਜੇ (trace) {
                ਅਚਲ std::chrono::duration<ਦੁੱਗਣਾ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(ਸੁਨੇਹਾ_ptr->cmd), elapsed.count());
            }
        } ਨਹੀਂਤਾਂ {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        ਸੁਨੇਹਾ_ptr->completion.set_value();
        ਆਖਰੀ_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = ਝੂਠ;
    queue.ਰੁਕਾਵਟ();
    sock = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (thread.joinable()) {
        thread.join();
    }
}

ਸਥਿਰ std::shared_ptr<rpc_dispatcher> get_dispatcher(ਅਚਲ std::string & endpoint) {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
    ਸਥਿਰ std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    ਸਵੈਚਾਲਿਤ it = dispatchers.find(endpoint);
    ਜੇ (it != dispatchers.end()) {
        ਜੇ (ਸਵੈਚਾਲਿਤ dispatcher = it->second.ਤਾਲਾ()) {
            ਵਾਪਸ dispatcher;
        }
    }

    ਸਵੈਚਾਲਿਤ dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ਵਾਪਸ dispatcher;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_free_ਬਫਰ_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_ਬਫਰ, request, ਆਕਾਰ(*request));
    ਮਿਟਾਓ ctx;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_rpc_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    ਜੇ (ctx->ਅਧਾਰ_ptr != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ctx->ਅਧਾਰ_ptr;
    }
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_rsp response;
    ctx->dispatcher->send(RPC_CMD_ਬਫਰ_GET_ਅਧਾਰ, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
    ctx->ਅਧਾਰ_ptr = reinterpret_cast<ਖਾਲੀ *>(response.ਅਧਾਰ_ptr);
    ਵਾਪਸ ctx->ਅਧਾਰ_ptr;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_rpc(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ਬਫਰ->iface.free_ਬਫਰ == ggml_backend_rpc_ਬਫਰ_free_ਬਫਰ;
}

ਸਥਿਰ rpc_tensor serialize_tensor(ਅਚਲ ggml_tensor * tensor, ਅਚਲ rpc_dispatcher * dispatcher = ਖਾਲੀ_ਸੂਚਕ) {
    rpc_tensor ਨਤੀਜਾ;
    ਜੇ (!tensor) {
        memset(&ਨਤੀਜਾ, 0, ਆਕਾਰ(ਨਤੀਜਾ));
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    ਨਤੀਜਾ.id = reinterpret_cast<uint64_t>(tensor);
    ਨਤੀਜਾ.type = tensor->type;
    ਜੇ (tensor->ਬਫਰ && ggml_backend_ਬਫਰ_is_rpc(tensor->ਬਫਰ)) {
        ggml_backend_ਬਫਰ_t ਬਫਰ = tensor->ਬਫਰ;
        ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        ਜੇ (ctx != ਖਾਲੀ_ਸੂਚਕ && (dispatcher == ਖਾਲੀ_ਸੂਚਕ || ctx->dispatcher.get() == dispatcher)) {
            ਨਤੀਜਾ.ਬਫਰ = ctx->remote_ptr;
            ਨਤੀਜਾ.ਅੰਕੜੇ = reinterpret_cast<uint64_t>(tensor->ਅੰਕੜੇ);
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ.ਬਫਰ = 0;
            ਨਤੀਜਾ.ਅੰਕੜੇ = 0;
        }
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ.ਬਫਰ = 0;
        ਨਤੀਜਾ.ਅੰਕੜੇ   = 0;
    }
    ਲਈ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ਨਤੀਜਾ.ne[i] = tensor->ne[i];
        ਨਤੀਜਾ.nb[i] = tensor->nb[i];
    }
    ਨਤੀਜਾ.op = tensor->op;
    ਲਈ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t); i++) {
        ਨਤੀਜਾ.op_params[i] = tensor->op_params[i];
    }
    ਨਤੀਜਾ.flags = tensor->flags;
    ਲਈ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ਨਤੀਜਾ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ਨਤੀਜਾ.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    ਨਤੀਜਾ.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(ਨਤੀਜਾ.name, 0, ਆਕਾਰ(ਨਤੀਜਾ.name));
    ਨਤੀਜਾ.use_ਗਿਣਤੀ = 0;

    snprintf(ਨਤੀਜਾ.name, GGML_MAX_NAME, "%s", tensor->name);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_rpc_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    ਜੇ (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == ਖਾਲੀ_ਸੂਚਕ)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, ਆਕਾਰ(*request));
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_memset_tensor(
        ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->ਵਿੱਥ = ਵਿੱਥ;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ,
             rpc_log_format_ਮਾਪ(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, ਆਕਾਰ(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ਸਥਿਰ std::shared_ptr<uint8_t> serialize_ਸੈੱਟ_tensor(ਅਚਲ rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t ਵਿੱਥ, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size, size_t & ਨਿਵੇਸ਼_ਮਾਪ) {
    ਨਿਵੇਸ਼_ਮਾਪ = ਆਕਾਰ(rpc_tensor) + ਆਕਾਰ(cache_flag) + ਆਕਾਰ(ਵਿੱਥ) + size;
    uint8_t * ਨਿਵੇਸ਼ = ਨਵਾਂ uint8_t[ਨਿਵੇਸ਼_ਮਾਪ]();
    uint8_t * p = ਨਿਵੇਸ਼;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(p, &rpc_tensor, ਆਕਾਰ(rpc_tensor)); p += ਆਕਾਰ(rpc_tensor);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(p, &cache_flag, ਆਕਾਰ(cache_flag)); p += ਆਕਾਰ(cache_flag);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(p, &ਵਿੱਥ,     ਆਕਾਰ(ਵਿੱਥ));     p += ਆਕਾਰ(ਵਿੱਥ);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(p, ਅੰਕੜੇ, size);
    ਵਾਪਸ std::shared_ptr<uint8_t>(ਨਿਵੇਸ਼, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ਸਥਿਰ ਤਾਰਕਿਕ rpc_use_hash_cache(ਅਚਲ ggml_tensor * tensor, size_t size) {
    ਵਾਪਸ size > HASH_ਹੱਦ && tensor->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ,
             rpc_log_format_ਮਾਪ(size).c_str());
    ਜੇ (rpc_use_hash_cache(tensor, size)) {
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->ਵਿੱਥ = ਵਿੱਥ;
        request->hash = fnv_hash((ਅਚਲ uint8_t*)ਅੰਕੜੇ, size);
        rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_ਸੈੱਟ_TENSOR_HASH, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ਨਤੀਜਾ ? "hit" : "miss");
        ਜੇ (response.ਨਤੀਜਾ) {
            // the server has the same data, no need to send it
            ਵਾਪਸ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ਨਿਵੇਸ਼_ਮਾਪ;
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = serialize_ਸੈੱਟ_tensor(rpc_tensor, cache_flag, ਵਿੱਥ, ਅੰਕੜੇ, size, ਨਿਵੇਸ਼_ਮਾਪ);
    ctx->dispatcher->send(RPC_CMD_ਸੈੱਟ_TENSOR, ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_ਮਾਪ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_ਸੈੱਟ_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ,
        size_t ਵਿੱਥ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ, rpc_log_format_ਮਾਪ(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t ਨਿਵੇਸ਼_ਮਾਪ = ਆਕਾਰ(rpc_tensor) + 4*ਆਕਾਰ(uint64_t) + size*n_copies;
    uint8_t * ਨਿਵੇਸ਼ = ਨਵਾਂ uint8_t[ਨਿਵੇਸ਼_ਮਾਪ]();
    uint8_t * dest = ਨਿਵੇਸ਼;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest, &rpc_tensor, ਆਕਾਰ(rpc_tensor));
    dest += ਆਕਾਰ(rpc_tensor);
    uint64_t ਸਿਰਲੇਖ[4] = { ਵਿੱਥ, size, n_copies, stride_tensor };
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest, ਸਿਰਲੇਖ, ਆਕਾਰ(ਸਿਰਲੇਖ));
    dest += ਆਕਾਰ(ਸਿਰਲੇਖ);
    ਲਈ (size_t i = 0; i < n_copies; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest + i*size, (ਅਚਲ ਅੱਖਰ *)ਅੰਕੜੇ + i*stride_ਅੰਕੜੇ, size);
    }
    std::shared_ptr<uint8_t> ਨਿਵੇਸ਼_ptr(ਨਿਵੇਸ਼, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_ਸੈੱਟ_TENSOR_2_D, ਨਿਵੇਸ਼_ptr, ਨਿਵੇਸ਼_ਮਾਪ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_get_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ,
        size_t ਵਿੱਥ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ, rpc_log_format_ਮਾਪ(size).c_str(), n_copies, stride_tensor);
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->ਵਿੱਥ   = ਵਿੱਥ;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    ਜੇ (stride_ਅੰਕੜੇ == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ਆਕਾਰ(*request), ਅੰਕੜੇ, size*n_copies);
    } ਨਹੀਂਤਾਂ {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ਆਕਾਰ(*request), packed.ਅੰਕੜੇ(), packed.size());
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *)ਅੰਕੜੇ + i*stride_ਅੰਕੜੇ, packed.ਅੰਕੜੇ() + i*size, size);
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ,
             rpc_log_format_ਮਾਪ(size).c_str());
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->ਵਿੱਥ = ਵਿੱਥ;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, ਆਕਾਰ(*request), ਅੰਕੜੇ, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_rpc_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਜੇ (ggml_backend_ਬਫਰ_is_rpc(src->ਬਫਰ)) {
        // check if src and dst are on the same server
        ggml_backend_ਬਫਰ_t src_ਬਫਰ = src->ਬਫਰ;
        ggml_backend_rpc_ਬਫਰ_context * src_ctx = (ggml_backend_rpc_ਬਫਰ_context *)src_ਬਫਰ->context;
        ggml_backend_ਬਫਰ_t dst_ਬਫਰ = dst->ਬਫਰ;
        ggml_backend_rpc_ਬਫਰ_context * dst_ctx = (ggml_backend_rpc_ਬਫਰ_context *)dst_ਬਫਰ->context;
        ਜੇ (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ਵਾਪਸ ਝੂਠ;
        }
        ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_ਸੁਨੇਹਾ_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
        ਵਾਪਸ response.ਨਤੀਜਾ;
    }
    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_backend_rpc_ਬਫਰ_context * ctx = (ggml_backend_rpc_ਬਫਰ_context *)ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_ਬਫਰ_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_ਬਫਰ_CLEAR, request, ਆਕਾਰ(*request));
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_rpc_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_rpc_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_rpc_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_ਬਫਰ_ਸੈੱਟ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_ਬਫਰ_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_rpc_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = (ggml_backend_rpc_ਬਫਰ_type_context *)buft->context;
    ਵਾਪਸ buft_ctx->name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_rpc_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = (ggml_backend_rpc_ਬਫਰ_type_context *)buft->context;
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_rsp response;

    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_ਬਫਰ, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
    ਜੇ (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_ਮਾਪ(request->size).c_str(),
                response.remote_ptr, rpc_log_format_ਮਾਪ(response.remote_ਮਾਪ).c_str());
        ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_ਬਫਰ_init(buft,
            ggml_backend_rpc_ਬਫਰ_interface,
            ਨਵਾਂ ggml_backend_rpc_ਬਫਰ_context{dispatcher, ਖਾਲੀ_ਸੂਚਕ, response.remote_ptr},
            response.remote_ਮਾਪ);
        ਵਾਪਸ ਬਫਰ;
    } ਨਹੀਂਤਾਂ {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_ਮਾਪ(request->size).c_str());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਸਥਿਰ size_t get_alignment(ਅਚਲ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_alignment_req>();
    request->device = device;
    rpc_ਸੁਨੇਹਾ_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_ਮਾਪ(response.alignment).c_str());
    ਵਾਪਸ response.alignment;
}

ਸਥਿਰ size_t ggml_backend_rpc_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = (ggml_backend_rpc_ਬਫਰ_type_context *)buft->context;
    ਵਾਪਸ buft_ctx->alignment;
}

ਸਥਿਰ size_t get_max_ਮਾਪ(ਅਚਲ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_req>();
    request->device = device;
    rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_ਮਾਪ, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_ਮਾਪ(response.max_size).c_str());
    ਵਾਪਸ response.max_size;
}

ਸਥਿਰ size_t ggml_backend_rpc_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = (ggml_backend_rpc_ਬਫਰ_type_context *)buft->context;
    ਵਾਪਸ buft_ctx->max_size;
}

ਸਥਿਰ size_t ggml_backend_rpc_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    ਤਾਰਕਿਕ rpc_get = ਝੂਠ;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == ਖਾਲੀ_ਸੂਚਕ);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_ਮਾਪ_may_expand(tensor->op);

    ਜੇ (rpc_get) {
        ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = (ggml_backend_rpc_ਬਫਰ_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ਅਚਲ size_t min_ਮਾਪ = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ਬਣਤਰ alloc_ਮਾਪ_cache_ਕੁੰਜੀ {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_ਮਾਪ_cache_ਕੁੰਜੀ ਕੁੰਜੀ = {};
        ਕੁੰਜੀ.device = buft_ctx->device;
        ਕੁੰਜੀ.type = tensor->type;
        ਕੁੰਜੀ.op = tensor->op;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਕੁੰਜੀ.op_params, tensor->op_params, ਆਕਾਰ(ਕੁੰਜੀ.op_params));
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            ਕੁੰਜੀ.ne[i] = (uint32_t)tensor->ne[i];
            ਕੁੰਜੀ.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ਅਚਲ uint8_t *)&ਕੁੰਜੀ, ਆਕਾਰ(ਕੁੰਜੀ));
        cache_hash = fnv_hash((ਅਚਲ uint8_t *)buft_ctx->endpoint.ਅੰਕੜੇ(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ਸਥਿਰ std::mutex cache_mutex;
        ਸਥਿਰ std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> ਤਾਲਾ(cache_mutex);
            ਸਵੈਚਾਲਿਤ it = cache.find(cache_hash);
            ਜੇ (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_ਮਾਪ(it->second).c_str());
                ਵਾਪਸ std::max<size_t>(it->second, min_ਮਾਪ);
            }
        }

        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_rsp response;
        ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_ਮਾਪ, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));

        {
            std::lock_guard<std::mutex> ਤਾਲਾ(cache_mutex);
            cache[cache_hash] = response.alloc_ਮਾਪ;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_ਮਾਪ(response.alloc_ਮਾਪ).c_str());

        ਵਾਪਸ std::max<size_t>(response.alloc_ਮਾਪ, min_ਮਾਪ);
    }

    ਵਾਪਸ ggml_nbytes(tensor);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_i ggml_backend_rpc_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_ਬਫਰ_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_rpc_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_rpc_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ਨਿਰਾ,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ਵਾਪਸ rpc_ctx->name.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ਮਿਟਾਓ rpc_ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਸੈੱਟ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ,
             rpc_log_format_ਮਾਪ(size).c_str());
    ਜੇ (rpc_use_hash_cache(tensor, size)) {
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->ਵਿੱਥ = ਵਿੱਥ;
        request->hash = fnv_hash((ਅਚਲ uint8_t*)ਅੰਕੜੇ, size);
        rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_ਸੈੱਟ_TENSOR_HASH, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ਨਤੀਜਾ ? "hit" : "miss");
        ਜੇ (response.ਨਤੀਜਾ) {
            // the server has the same data, no need to send it
            ਵਾਪਸ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ਨਿਵੇਸ਼_ਮਾਪ;
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = serialize_ਸੈੱਟ_tensor(rpc_tensor, cache_flag, ਵਿੱਥ, ਅੰਕੜੇ, size, ਨਿਵੇਸ਼_ਮਾਪ);
    ctx->dispatcher->send_async(RPC_CMD_ਸੈੱਟ_TENSOR, ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_ਮਾਪ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ਵਿੱਥ,
             rpc_log_format_ਮਾਪ(size).c_str());
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->ਵਿੱਥ = ਵਿੱਥ;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, ਆਕਾਰ(*request), ਅੰਕੜੇ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_tensor(ggml_tensor * tensor, ਅਚਲ ggml_cgraph * cgraph, ਅਚਲ rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & ਦੇਖਿਆ) {
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    ਜੇ (ਦੇਖਿਆ.find(tensor) != ਦੇਖਿਆ.end()) {
        ਵਾਪਸ;
    }
    ਦੇਖਿਆ.insert(tensor);
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੋੜੋ_tensor(tensor->src[i], cgraph, dispatcher, tensors, ਦੇਖਿਆ);
    }
    ਜੋੜੋ_tensor(tensor->view_src, cgraph, dispatcher, tensors, ਦੇਖਿਆ);
    rpc_tensor ਨਤੀਜਾ = serialize_tensor(tensor, dispatcher);
    ਅਚਲ size_t hash_pos = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, tensor);
    ਜੇ (hash_pos != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, hash_pos)) {
        ਨਤੀਜਾ.use_ਗਿਣਤੀ = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(ਨਤੀਜਾ);
}

ਸਥਿਰ uint8_t * serialize_ਗ੍ਰਾਫ(uint32_t device, ਅਚਲ ggml_cgraph * cgraph, ਅਚਲ rpc_dispatcher * dispatcher, size_t * output_ਮਾਪ) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> ਦੇਖਿਆ;
    ਲਈ (uint32_t i = 0; i < n_nodes; i++) {
        ਜੋੜੋ_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, ਦੇਖਿਆ);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_ਮਾਪ = 2*ਆਕਾਰ(uint32_t) + ਆਕਾਰ(uint64_t) + n_nodes * ਆਕਾਰ(uint64_t) + ਆਕਾਰ(uint32_t) + n_tensors * ਆਕਾਰ(rpc_tensor);
    uint8_t * output = ਨਵਾਂ uint8_t[*output_ਮਾਪ]();
    uint8_t * dest = output;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest, &device, ਆਕਾਰ(device));
    dest += ਆਕਾਰ(device);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest, &cgraph->uid, ਆਕਾਰ(cgraph->uid));
    dest += ਆਕਾਰ(cgraph->uid);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest, &n_nodes, ਆਕਾਰ(n_nodes));
    dest += ਆਕਾਰ(n_nodes);
    ਲਈ (uint32_t i = 0; i < n_nodes; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest + i * ਆਕਾਰ(uint64_t), &cgraph->nodes[i], ਆਕਾਰ(uint64_t));
    }
    dest += n_nodes * ਆਕਾਰ(uint64_t);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dest, &n_tensors, ਆਕਾਰ(n_tensors));
    dest += ਆਕਾਰ(n_tensors);
    rpc_tensor * ਬਾਹਰਲਾ_tensors = (rpc_tensor *)dest;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ_tensors, tensors.ਅੰਕੜੇ(), n_tensors * ਆਕਾਰ(rpc_tensor));
    ਵਾਪਸ output;
}

ਖਾਲੀ rpc_dispatcher::ਗ੍ਰਾਫ_compute(uint32_t device, ਅਚਲ ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> ਤਾਲਾ(ਗ੍ਰਾਫ_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    ਸਵੈਚਾਲਿਤ & device_ਗ੍ਰਾਫ_uids = ਗ੍ਰਾਫ_uids[device];
    ਤਾਰਕਿਕ reuse = cgraph->uid != 0 && device_ਗ੍ਰਾਫ_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    ਜੇ (reuse) {
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_ਗ੍ਰਾਫ_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_ਗ੍ਰਾਫ_RECOMPUTE, request, ਆਕਾਰ(*request));
    } ਨਹੀਂਤਾਂ {
        ਜੇ (cgraph->uid != 0) {
            ਜੇ (device_ਗ੍ਰਾਫ_uids.size() >= ਗ੍ਰਾਫ_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_ਗ੍ਰਾਫ_uids.size());
                device_ਗ੍ਰਾਫ_uids.clear();
            }
            device_ਗ੍ਰਾਫ_uids.insert(cgraph->uid);
        }
        size_t ਨਿਵੇਸ਼_ਮਾਪ = 0;
        uint8_t * ਨਿਵੇਸ਼ = serialize_ਗ੍ਰਾਫ(device, cgraph, ਇਹ, &ਨਿਵੇਸ਼_ਮਾਪ);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_ਮਾਪ(ਨਿਵੇਸ਼_ਮਾਪ).c_str());
        std::shared_ptr<uint8_t> ਨਿਵੇਸ਼_ptr(ਨਿਵੇਸ਼, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_ਗ੍ਰਾਫ_COMPUTE, ਨਿਵੇਸ਼_ptr, ਨਿਵੇਸ਼_ਮਾਪ);
    }
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_rpc_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->ਗ੍ਰਾਫ_compute(rpc_ctx->device, cgraph);
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_ਰਿਕਾਰਡ(event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_ਅਣਵਰਤਿਆ(backend);
    GGML_ਅਣਵਰਤਿਆ(event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_ਸੈੱਟ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ,
        size_t ਵਿੱਥ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    ggml_backend_tensor_ਸੈੱਟ_2d(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ,
        size_t ਵਿੱਥ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    ggml_backend_tensor_get_2d(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_ਸੈੱਟ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_rpc_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_rpc_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

ggml_backend_ਬਫਰ_type_t ggml_backend_rpc_ਬਫਰ_type(ਅਚਲ ਅੱਖਰ * endpoint, uint32_t device) {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ਸਥਿਰ std::unordered_map<std::string, ggml_backend_ਬਫਰ_type_t> buft_map;
    ਸਵੈਚਾਲਿਤ it = buft_map.find(buft_name);
    ਜੇ (it != buft_map.end()) {
        ਵਾਪਸ it->second;
    }
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_ਮਾਪ(dispatcher, device);
    ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = ਨਵਾਂ ggml_backend_rpc_ਬਫਰ_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    ਸਵੈਚਾਲਿਤ reg = ggml_backend_rpc_ਜੋੜੋ_server(endpoint);
    ggml_backend_ਬਫਰ_type_t buft = ਨਵਾਂ ggml_backend_ਬਫਰ_type {
        /* .iface   = */ ggml_backend_rpc_ਬਫਰ_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_ਮਾਪ(alignment).c_str(), rpc_log_format_ਮਾਪ(max_size).c_str());
    ਵਾਪਸ buft;
}

ggml_backend_t ggml_backend_rpc_init(ਅਚਲ ਅੱਖਰ * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = ਨਵਾਂ ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    ਸਵੈਚਾਲਿਤ reg = ggml_backend_rpc_ਜੋੜੋ_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = ਨਵਾਂ ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ਵਾਪਸ backend;
}

ਤਾਰਕਿਕ ggml_backend_is_rpc(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

ਖਾਲੀ ggml_backend_rpc_get_device_ਸਮ੍ਰਿਤੀ(ਅਚਲ ਅੱਖਰ * endpoint, uint32_t device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(endpoint);
    ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_req>();
    request->device = device;
    rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_ਸਮ੍ਰਿਤੀ, request, ਆਕਾਰ(*request), &response, ਆਕਾਰ(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_ਮਾਪ(response.free_mem).c_str(), rpc_log_format_ਮਾਪ(response.ਕੁੱਲ_mem).c_str());
    *ਮੁਕਤ = response.free_mem;
    *ਕੁੱਲ = response.ਕੁੱਲ_mem;
}

// RPC server-side implementation

ਵਰਗ rpc_server {
ਜਨਤਕ:
    rpc_server(std::vector<ggml_backend_t> all_backends, ਅਚਲ ਅੱਖਰ * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_ਹਾਲਤਾਂ.resize(backends.size());
    }
    ~rpc_server();

    ਖਾਲੀ hello(rpc_ਸੁਨੇਹਾ_hello_rsp & response);
    ਤਾਰਕਿਕ alloc_ਬਫਰ(ਅਚਲ rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_req & request, rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_rsp & response);
    ਤਾਰਕਿਕ get_alignment(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_alignment_req & request, rpc_ਸੁਨੇਹਾ_get_alignment_rsp & response);
    ਤਾਰਕਿਕ get_max_ਮਾਪ(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_req & request, rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_rsp & response);
    ਤਾਰਕਿਕ ਬਫਰ_get_ਅਧਾਰ(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_req & request, rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_rsp & response);
    ਤਾਰਕਿਕ free_ਬਫਰ(ਅਚਲ rpc_ਸੁਨੇਹਾ_free_ਬਫਰ_req & request);
    ਤਾਰਕਿਕ ਬਫਰ_clear(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਬਫਰ_clear_req & request);
    ਤਾਰਕਿਕ memset_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_memset_tensor_req & request);
    ਤਾਰਕਿਕ ਸੈੱਟ_tensor(ਅਚਲ std::vector<uint8_t> & ਨਿਵੇਸ਼);
    ਤਾਰਕਿਕ ਸੈੱਟ_tensor_2d(ਅਚਲ std::vector<uint8_t> & ਨਿਵੇਸ਼);
    ਤਾਰਕਿਕ ਸੈੱਟ_tensor_hash(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_req & request, rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_rsp & response);
    ਤਾਰਕਿਕ get_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_tensor_req & request, std::vector<uint8_t> & response);
    ਤਾਰਕਿਕ get_tensor_2d(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    ਤਾਰਕਿਕ copy_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_copy_tensor_req & request, rpc_ਸੁਨੇਹਾ_copy_tensor_rsp & response);
    ਤਾਰਕਿਕ ਗ੍ਰਾਫ_compute(ਅਚਲ std::vector<uint8_t> & ਨਿਵੇਸ਼);
    ਤਾਰਕਿਕ ਗ੍ਰਾਫ_recompute(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਗ੍ਰਾਫ_recompute_req & request);
    ਤਾਰਕਿਕ comm_init(ਅਚਲ rpc_ਸੁਨੇਹਾ_comm_init_req & request, rpc_ਸੁਨੇਹਾ_comm_init_rsp & response);
    ਤਾਰਕਿਕ comm_allreduce(ਅਚਲ rpc_ਸੁਨੇਹਾ_comm_allreduce_req & request);
    ਤਾਰਕਿਕ comm_free(ਅਚਲ rpc_ਸੁਨੇਹਾ_comm_free_req & request);
    ਤਾਰਕਿਕ init_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_init_tensor_req & request);
    ਤਾਰਕਿਕ get_alloc_ਮਾਪ(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_req & request, rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_rsp & response);
    ਤਾਰਕਿਕ get_device_ਸਮ੍ਰਿਤੀ(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_req & request, rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_rsp & response);

    ਬਣਤਰ stored_ਗ੍ਰਾਫ {
        std::vector<uint8_t>   ਬਫਰ;
        ggml_cgraph          * graph;
    };

ਨਿਜੀ:
    ਖਾਲੀ sync_all_backends();
    ਤਾਰਕਿਕ get_cached_file(uint64_t hash, std::vector<uint8_t> & ਅੰਕੜੇ);
    ggml_tensor * deserialize_tensor(ਬਣਤਰ ggml_context * ctx, ਅਚਲ rpc_tensor * tensor);
    ggml_tensor * create_ਗੰਢ(uint64_t id,
                              ਬਣਤਰ ggml_context * ctx,
                              ਅਚਲ std::unordered_map<uint64_t, ਅਚਲ rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ਬਣਤਰ ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ਬਣਤਰ comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_ਬਫਰ_ptr scratch;
        size_t                  scratch_ਮਾਪ = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ਅਚਲ ਅੱਖਰ * cache_dir;
    std::unordered_set<ggml_backend_ਬਫਰ_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_ਗ੍ਰਾਫ>> stored_graphs;
    std::vector<comm_state> comm_ਹਾਲਤਾਂ;
};

ਖਾਲੀ rpc_server::hello(rpc_ਸੁਨੇਹਾ_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_ਸੰਸਕਰਣ;
    response.minor = RPC_PROTO_MINOR_ਸੰਸਕਰਣ;
    response.patch = RPC_PROTO_PATCH_ਸੰਸਕਰਣ;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

ਤਾਰਕਿਕ rpc_server::get_alloc_ਮਾਪ(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_req & request, rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_rsp & response) {
    uint32_t dev_id = request.device;
    ਜੇ (dev_id >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_ਬਫਰ_type_t buft;
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ਵਾਪਸ ਝੂਠ;
    }
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ);
    ਜੇ (tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        //No buffer allocated.
        buft = ggml_backend_get_default_ਬਫਰ_type(backends[dev_id]);
    } ਨਹੀਂਤਾਂ {
        buft = tensor->ਬਫਰ->buft;
    }

    response.alloc_ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(buft, tensor);

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::alloc_ਬਫਰ(ਅਚਲ rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_req & request, rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_rsp & response) {
    uint32_t dev_id = request.device;
    ਜੇ (dev_id >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_ਬਫਰ_type_t buft = ggml_backend_get_default_ਬਫਰ_type(backends[dev_id]);
    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_buft_alloc_ਬਫਰ(buft, request.size);
    response.remote_ptr = 0;
    response.remote_ਮਾਪ = 0;
    ਜੇ (ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
        response.remote_ptr = reinterpret_cast<uint64_t>(ਬਫਰ);
        response.remote_ਮਾਪ = ਬਫਰ->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_ਮਾਪ(request.size).c_str(), response.remote_ptr,
                rpc_log_format_ਮਾਪ(response.remote_ਮਾਪ).c_str());
        buffers.insert(ਬਫਰ);
    } ਨਹੀਂਤਾਂ {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_ਮਾਪ(request.size).c_str());
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::get_alignment(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_alignment_req & request, rpc_ਸੁਨੇਹਾ_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    ਜੇ (dev_id >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_ਬਫਰ_type_t buft = ggml_backend_get_default_ਬਫਰ_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_ਮਾਪ(alignment).c_str());
    response.alignment = alignment;
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::get_max_ਮਾਪ(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_req & request, rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_rsp & response) {
    uint32_t dev_id = request.device;
    ਜੇ (dev_id >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_ਬਫਰ_type_t buft = ggml_backend_get_default_ਬਫਰ_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_ਮਾਪ(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_ਮਾਪ(max_size).c_str());
    response.max_size = max_size;
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::ਬਫਰ_get_ਅਧਾਰ(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_req & request, rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_ਬਫਰ_t ਬਫਰ = reinterpret_cast<ggml_backend_ਬਫਰ_t>(request.remote_ptr);
    ਜੇ (buffers.find(ਬਫਰ) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    ਖਾਲੀ * ਅਧਾਰ = ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ਬਫਰ);
    response.ਅਧਾਰ_ptr = reinterpret_cast<uint64_t>(ਅਧਾਰ);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::free_ਬਫਰ(ਅਚਲ rpc_ਸੁਨੇਹਾ_free_ਬਫਰ_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_ਬਫਰ_t ਬਫਰ = reinterpret_cast<ggml_backend_ਬਫਰ_t>(request.remote_ptr);
    ਜੇ (buffers.find(ਬਫਰ) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    ਲਈ (ਸਵੈਚਾਲਿਤ & sgs : stored_graphs) {
        ਲਈ (ਸਵੈਚਾਲਿਤ & sg : sgs) {
            sg.second.graph = ਖਾਲੀ_ਸੂਚਕ;
        }
    }
    ggml_backend_ਬਫਰ_free(ਬਫਰ);
    buffers.erase(ਬਫਰ);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::ਬਫਰ_clear(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਬਫਰ_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_ਬਫਰ_t ਬਫਰ = reinterpret_cast<ggml_backend_ਬਫਰ_t>(request.remote_ptr);
    ਜੇ (buffers.find(ਬਫਰ) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_ਬਫਰ_clear(ਬਫਰ, request.value);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::memset_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_memset_tensor_req & request) {
    sync_all_backends();
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint64_t tensor_ਮਾਪ = ggml_nbytes(tensor);
    ਜੇ (request.ਵਿੱਥ > tensor_ਮਾਪ || request.size > tensor_ਮਾਪ - request.ਵਿੱਥ) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.ਵਿੱਥ, request.size, tensor_ਮਾਪ);
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint64_t ਬਫਰ_ਸ਼ੁਰੂ = (uint64_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ);
    ਅਚਲ uint64_t ਬਫਰ_ਮਾਪ = ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);
    ਜੇ (request.tensor.ਅੰਕੜੇ < ਬਫਰ_ਸ਼ੁਰੂ) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ uint64_t ਅੰਕੜੇ_ਵਿੱਥ = request.tensor.ਅੰਕੜੇ - ਬਫਰ_ਸ਼ੁਰੂ;
    ਜੇ (ਅੰਕੜੇ_ਵਿੱਥ > ਬਫਰ_ਮਾਪ ||
        request.ਵਿੱਥ > ਬਫਰ_ਮਾਪ - ਅੰਕੜੇ_ਵਿੱਥ ||
        request.size > ਬਫਰ_ਮਾਪ - ਅੰਕੜੇ_ਵਿੱਥ - request.ਵਿੱਥ) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (tensor->ਬਫਰ->iface.memset_tensor == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ਖਾਲੀ *) tensor->ਬਫਰ, tensor->ਅੰਕੜੇ, request.ਵਿੱਥ,
            rpc_log_format_ਮਾਪ(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.ਵਿੱਥ, request.size);
    ਵਾਪਸ ਸੱਚ;
}

ggml_tensor * rpc_server::deserialize_tensor(ਬਣਤਰ ggml_context * ctx, ਅਚਲ rpc_tensor * tensor) {
    // Validate tensor type before using it
    ਜੇ (tensor->type >= GGML_TYPE_ਗਿਣਤੀ) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    ਜੇ (ggml_blck_ਮਾਪ((ਗਿਣਤੀ ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_tensor * ਨਤੀਜਾ = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    ਜੇ (ਨਤੀਜਾ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਲਈ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ਨਤੀਜਾ->nb[i] = tensor->nb[i];
    }
    ਨਤੀਜਾ->ਬਫਰ = reinterpret_cast<ggml_backend_ਬਫਰ_t>(tensor->ਬਫਰ);
    ਜੇ (ਨਤੀਜਾ->ਬਫਰ && buffers.find(ਨਤੀਜਾ->ਬਫਰ) == buffers.end()) {
        ਨਤੀਜਾ->ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (ਨਤੀਜਾ->ਬਫਰ && ggml_nelements(ਨਤੀਜਾ) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_ਮਾਪ = (uint64_t) ggml_nbytes(ਨਤੀਜਾ);
        uint64_t ਬਫਰ_ਸ਼ੁਰੂ = (uint64_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ਨਤੀਜਾ->ਬਫਰ);
        uint64_t ਬਫਰ_ਮਾਪ = (uint64_t) ggml_backend_ਬਫਰ_get_ਮਾਪ(ਨਤੀਜਾ->ਬਫਰ);
        ਜੇ (tensor->ਅੰਕੜੇ + tensor_ਮਾਪ < tensor->ਅੰਕੜੇ ||
            tensor->ਅੰਕੜੇ < ਬਫਰ_ਸ਼ੁਰੂ || tensor->ਅੰਕੜੇ + tensor_ਮਾਪ > ਬਫਰ_ਸ਼ੁਰੂ + ਬਫਰ_ਮਾਪ) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(ਨਤੀਜਾ->type),
                      ਨਤੀਜਾ->ne[0], ਨਤੀਜਾ->ne[1], ਨਤੀਜਾ->ne[2], ਨਤੀਜਾ->ne[3],
                      tensor->ਅੰਕੜੇ, tensor->ਅੰਕੜੇ + tensor_ਮਾਪ, ਬਫਰ_ਸ਼ੁਰੂ, ਬਫਰ_ਸ਼ੁਰੂ + ਬਫਰ_ਮਾਪ);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ਨਤੀਜਾ->op = (ggml_op) tensor->op;
    ਲਈ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t); i++) {
        ਨਤੀਜਾ->op_params[i] = tensor->op_params[i];
    }
    ਨਤੀਜਾ->flags = tensor->flags;
    ਨਤੀਜਾ->ਅੰਕੜੇ = reinterpret_cast<ਖਾਲੀ *>(tensor->ਅੰਕੜੇ);
    ggml_ਸੈੱਟ_name(ਨਤੀਜਾ, tensor->name);
    ਵਾਪਸ ਨਤੀਜਾ;
}


ਤਾਰਕਿਕ rpc_server::ਸੈੱਟ_tensor(ਅਚਲ std::vector<uint8_t> & ਨਿਵੇਸ਼) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t ਵਿੱਥ;
    ਅਚਲ size_t ਸਿਰਲੇਖ_ਮਾਪ = ਆਕਾਰ(rpc_tensor) + ਆਕਾਰ(cache_flag) + ਆਕਾਰ(ਵਿੱਥ);
    ਜੇ (ਨਿਵੇਸ਼.size() < ਸਿਰਲੇਖ_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ rpc_tensor * ਵਿੱਚ_tensor = (ਅਚਲ rpc_tensor *)ਨਿਵੇਸ਼.ਅੰਕੜੇ();
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&cache_flag, ਨਿਵੇਸ਼.ਅੰਕੜੇ() + ਆਕਾਰ(rpc_tensor), ਆਕਾਰ(cache_flag));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਵਿੱਥ,     ਨਿਵੇਸ਼.ਅੰਕੜੇ() + ਆਕਾਰ(rpc_tensor) + ਆਕਾਰ(cache_flag), ਆਕਾਰ(ਵਿੱਥ));
    ਅਚਲ size_t size = ਨਿਵੇਸ਼.size() - ਸਿਰਲੇਖ_ਮਾਪ;

    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, ਵਿੱਚ_tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ, ਵਿੱਥ,
            rpc_log_format_ਮਾਪ(size).c_str());

    // sanitize tensor->data
    {
        ਅਚਲ size_t p0 = (size_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ);
        ਅਚਲ size_t p1 = p0 + ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);

        ਜੇ (ਵਿੱਚ_tensor->ਅੰਕੜੇ + ਵਿੱਥ < p0 || ਵਿੱਚ_tensor->ਅੰਕੜੇ + ਵਿੱਥ >= p1 || size > (p1 - ਵਿੱਚ_tensor->ਅੰਕੜੇ - ਵਿੱਥ)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, ਵਿੱਚ_tensor->ਅੰਕੜੇ, ਵਿੱਥ, size, p0, p1);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ = ਨਿਵੇਸ਼.ਅੰਕੜੇ() + ਸਿਰਲੇਖ_ਮਾਪ;
    ਜੇ (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ਅਚਲ uint8_t*)ਅੰਕੜੇ, size);
        ਅੱਖਰ hash_str[17];
        snprintf(hash_str, ਆਕਾਰ(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.ਲਿਖ((ਅਚਲ ਅੱਖਰ *)ਅੰਕੜੇ, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_ਸੈੱਟ(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::ਸੈੱਟ_tensor_2d(ਅਚਲ std::vector<uint8_t> & ਨਿਵੇਸ਼) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    ਜੇ (ਨਿਵੇਸ਼.size() < ਆਕਾਰ(rpc_tensor) + 4*ਆਕਾਰ(uint64_t)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ rpc_tensor * ਵਿੱਚ_tensor = (ਅਚਲ rpc_tensor *)ਨਿਵੇਸ਼.ਅੰਕੜੇ();
    uint64_t ਸਿਰਲੇਖ[4];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਸਿਰਲੇਖ, ਨਿਵੇਸ਼.ਅੰਕੜੇ() + ਆਕਾਰ(rpc_tensor), ਆਕਾਰ(ਸਿਰਲੇਖ));
    ਅਚਲ uint64_t ਵਿੱਥ   = ਸਿਰਲੇਖ[0];
    ਅਚਲ uint64_t size     = ਸਿਰਲੇਖ[1];
    ਅਚਲ uint64_t n_copies = ਸਿਰਲੇਖ[2];
    ਅਚਲ uint64_t stride   = ਸਿਰਲੇਖ[3];

    ਅਚਲ uint64_t ਅੰਕੜੇ_ਮਾਪ = ਨਿਵੇਸ਼.size() - ਆਕਾਰ(rpc_tensor) - 4*ਆਕਾਰ(uint64_t);
    ਜੇ (n_copies == 0 || size == 0 || size > ਅੰਕੜੇ_ਮਾਪ / n_copies || size * n_copies != ਅੰਕੜੇ_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, ਵਿੱਚ_tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ, ਵਿੱਥ,
            rpc_log_format_ਮਾਪ(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        ਜੇ (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ uint64_t ਵਿਸਤਾਰ = (n_copies - 1)*stride + size;
        ਅਚਲ uint64_t p0 = (uint64_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ);
        ਅਚਲ uint64_t p1 = p0 + ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);

        ਜੇ (ਵਿੱਚ_tensor->ਅੰਕੜੇ < p0 || ਵਿੱਚ_tensor->ਅੰਕੜੇ > p1 || ਵਿੱਥ > p1 - ਵਿੱਚ_tensor->ਅੰਕੜੇ || ਵਿਸਤਾਰ > p1 - ਵਿੱਚ_tensor->ਅੰਕੜੇ - ਵਿੱਥ) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, ਵਿੱਚ_tensor->ਅੰਕੜੇ, ਵਿੱਥ, ਵਿਸਤਾਰ, p0, p1);
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ਵਿੱਥ > ggml_nbytes(tensor) || ਵਿਸਤਾਰ > ggml_nbytes(tensor) - ਵਿੱਥ) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, ਵਿੱਥ, ਵਿਸਤਾਰ, ggml_nbytes(tensor));
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ = ਨਿਵੇਸ਼.ਅੰਕੜੇ() + ਆਕਾਰ(rpc_tensor) + 4*ਆਕਾਰ(uint64_t);
    ggml_backend_tensor_ਸੈੱਟ_2d(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride, size);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & ਅੰਕੜੇ) {
    ਜੇ (!cache_dir) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅੱਖਰ hash_str[17];
    snprintf(hash_str, ਆਕਾਰ(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    ਜੇ (!fs::exists(cache_file, ec)) {
        ਵਾਪਸ ਝੂਠ;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    ਅੰਕੜੇ.resize(size);
    ifs.ਪਠ((ਅੱਖਰ *)ਅੰਕੜੇ.ਅੰਕੜੇ(), size);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::ਸੈੱਟ_tensor_hash(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_req & request, rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_rsp & response)
{
    sync_all_backends();
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }

    std::vector<uint8_t> cached_file;
    ਜੇ (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.ਵਿੱਥ, request.hash);
        response.ਨਤੀਜਾ = 0;
        ਵਾਪਸ ਸੱਚ;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ, request.ਵਿੱਥ,
            rpc_log_format_ਮਾਪ(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ਅਚਲ size_t p0 = (size_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ);
        ਅਚਲ size_t p1 = p0 + ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);

        ਜੇ (request.tensor.ਅੰਕੜੇ + request.ਵਿੱਥ < p0
         || request.tensor.ਅੰਕੜੇ + request.ਵਿੱਥ >= p1
         || size > (p1 - request.tensor.ਅੰਕੜੇ - request.ਵਿੱਥ)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.ਅੰਕੜੇ, request.ਵਿੱਥ, size, request.hash, p0, p1);
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ggml_backend_tensor_ਸੈੱਟ(tensor, cached_file.ਅੰਕੜੇ(), request.ਵਿੱਥ, size);
    response.ਨਤੀਜਾ = 1;
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::init_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_init_tensor_req & request) {
    sync_all_backends();
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ਵਾਪਸ ਝੂਠ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ);
    // Call the backend's buffer_init_tensor function
    ggml_backend_ਬਫਰ_t ਬਫਰ = tensor->ਬਫਰ;
    ਜੇ (ਬਫਰ && ਬਫਰ->iface.init_tensor) {
        ਬਫਰ->iface.init_tensor(ਬਫਰ, tensor);
    } ਨਹੀਂਤਾਂ {
        ਜੇ (!ਬਫਰ) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    ਜੇ (tensor->extra != ਖਾਲੀ_ਸੂਚਕ) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::get_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ, request.ਵਿੱਥ,
             rpc_log_format_ਮਾਪ(request.size).c_str());

    // sanitize tensor->data
    {
        ਅਚਲ size_t p0 = (size_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ);
        ਅਚਲ size_t p1 = p0 + ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);

        ਜੇ (request.tensor.ਅੰਕੜੇ + request.ਵਿੱਥ < p0 ||
            request.tensor.ਅੰਕੜੇ + request.ਵਿੱਥ >= p1 ||
            request.size > (p1 - request.tensor.ਅੰਕੜੇ - request.ਵਿੱਥ)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.ਅੰਕੜੇ, request.ਵਿੱਥ, request.size, p0, p1);
                ਵਾਪਸ ਝੂਠ;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.ਅੰਕੜੇ(), request.ਵਿੱਥ, request.size);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::get_tensor_2d(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (tensor == ਖਾਲੀ_ਸੂਚਕ || tensor->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ਖਾਲੀ*)tensor->ਬਫਰ, tensor->ਅੰਕੜੇ, request.ਵਿੱਥ,
            rpc_log_format_ਮਾਪ(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        ਜੇ (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ uint64_t ਵਿਸਤਾਰ = (request.n_copies - 1)*request.stride + request.size;
        ਅਚਲ uint64_t p0 = (uint64_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(tensor->ਬਫਰ);
        ਅਚਲ uint64_t p1 = p0 + ggml_backend_ਬਫਰ_get_ਮਾਪ(tensor->ਬਫਰ);

        ਜੇ (request.tensor.ਅੰਕੜੇ < p0 || request.tensor.ਅੰਕੜੇ > p1 || request.ਵਿੱਥ > p1 - request.tensor.ਅੰਕੜੇ ||
                ਵਿਸਤਾਰ > p1 - request.tensor.ਅੰਕੜੇ - request.ਵਿੱਥ) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.ਅੰਕੜੇ, request.ਵਿੱਥ, ਵਿਸਤਾਰ, p0, p1);
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (request.ਵਿੱਥ > ggml_nbytes(tensor) || ਵਿਸਤਾਰ > ggml_nbytes(tensor) - request.ਵਿੱਥ) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.ਵਿੱਥ, ਵਿਸਤਾਰ, ggml_nbytes(tensor));
            ਵਾਪਸ ਝੂਠ;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.ਅੰਕੜੇ(), request.ਵਿੱਥ, request.size, request.n_copies, request.stride, request.size);
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::copy_tensor(ਅਚਲ rpc_ਸੁਨੇਹਾ_copy_tensor_req & request, rpc_ਸੁਨੇਹਾ_copy_tensor_rsp & response) {
    sync_all_backends();
    ਬਣਤਰ ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    ਜੇ (src == ਖਾਲੀ_ਸੂਚਕ || dst == ਖਾਲੀ_ਸੂਚਕ || src->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ || dst->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }

    uint64_t src_ਮਾਪ   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_ਅੰਕੜੇ   = (uint64_t) dst->ਅੰਕੜੇ;
    uint64_t dst_ਅਧਾਰ   = (uint64_t) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(dst->ਬਫਰ);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_ਬਫਰ_get_ਮਾਪ(dst->ਬਫਰ);

    ਜੇ (dst_ਅੰਕੜੇ + src_ਮਾਪ > dst_ਅਧਾਰ + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_ਅੰਕੜੇ,
                         dst_ਅੰਕੜੇ + src_ਮਾਪ,
                         dst_ਅਧਾਰ,
                         dst_ਅਧਾਰ + dst_buf_sz);
        ਵਾਪਸ ਝੂਠ;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.ਨਤੀਜਾ = ggml_backend_ਬਫਰ_copy_tensor(src, dst);
    ਵਾਪਸ ਸੱਚ;
}

ggml_tensor * rpc_server::create_ਗੰਢ(uint64_t id,
                                      ਬਣਤਰ ggml_context * ctx,
                                      ਅਚਲ std::unordered_map<uint64_t, ਅਚਲ rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ਬਣਤਰ ggml_tensor*> & tensor_map) {
    ਜੇ (tensor_map.find(id) != tensor_map.end()) {
        ਵਾਪਸ tensor_map[id];
    }
    // Safely find the tensor pointer
    ਸਵੈਚਾਲਿਤ it_ptr = tensor_ptrs.find(id);
    ਜੇ (it_ptr == tensor_ptrs.end()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਅਚਲ rpc_tensor * tensor = it_ptr->second;

    ਬਣਤਰ ggml_tensor * ਨਤੀਜਾ = deserialize_tensor(ctx, tensor);
    ਜੇ (ਨਤੀਜਾ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ਨਤੀਜਾ->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ && ਨਤੀਜਾ->ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    tensor_map[id] = ਨਤੀਜਾ;
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        ਜੇ (tensor->src[i] == 0) {
            ਨਤੀਜਾ->src[i] = ਖਾਲੀ_ਸੂਚਕ;
        } ਨਹੀਂਤਾਂ {
            ਨਤੀਜਾ->src[i] = create_ਗੰਢ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            ਜੇ (ਨਤੀਜਾ->src[i] == ਖਾਲੀ_ਸੂਚਕ) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }
    }

    // Handle view_src similarly
    ਜੇ (tensor->view_src == 0) {
        ਨਤੀਜਾ->view_src = ਖਾਲੀ_ਸੂਚਕ;
    } ਨਹੀਂਤਾਂ {
        ਨਤੀਜਾ->view_src = create_ਗੰਢ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        ਜੇ (ਨਤੀਜਾ->view_src == ਖਾਲੀ_ਸੂਚਕ) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }
    ਨਤੀਜਾ->view_offs = tensor->view_offs;
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ rpc_server::ਗ੍ਰਾਫ_compute(ਅਚਲ std::vector<uint8_t> & ਨਿਵੇਸ਼) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    ਜੇ (ਨਿਵੇਸ਼.size() < 2*ਆਕਾਰ(uint32_t) + ਆਕਾਰ(uint64_t)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ uint8_t * src = ਨਿਵੇਸ਼.ਅੰਕੜੇ();
    uint32_t device;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&device, src, ਆਕਾਰ(device));
    src += ਆਕਾਰ(device);
    ਜੇ (device >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    uint64_t uid;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&uid, src, ਆਕਾਰ(uid));
    src += ਆਕਾਰ(uid);
    uint32_t n_nodes;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&n_nodes, src, ਆਕਾਰ(n_nodes));
    src += ਆਕਾਰ(n_nodes);
    ਜੇ (ਨਿਵੇਸ਼.size() < 2*ਆਕਾਰ(uint32_t) + ਆਕਾਰ(uint64_t) + n_nodes*ਆਕਾਰ(uint64_t) + ਆਕਾਰ(uint32_t)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ uint64_t * nodes = (ਅਚਲ uint64_t *)src;
    src += n_nodes*ਆਕਾਰ(uint64_t);
    uint32_t n_tensors;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&n_tensors, src, ਆਕਾਰ(n_tensors));
    src += ਆਕਾਰ(n_tensors);
    ਜੇ (ਨਿਵੇਸ਼.size() < 2*ਆਕਾਰ(uint32_t) + ਆਕਾਰ(uint64_t) + n_nodes*ਆਕਾਰ(uint64_t) + ਆਕਾਰ(uint32_t) + n_tensors*ਆਕਾਰ(rpc_tensor)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ rpc_tensor * tensors = (ਅਚਲ rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    ਜੇ (uid != 0 && stored_graphs[device].size() >= ਗ੍ਰਾਫ_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_ਗ੍ਰਾਫ sg_tmp;
    stored_ਗ੍ਰਾਫ & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_ਮਾਪ = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_ਗ੍ਰਾਫ_overhead_custom(n_nodes, ਝੂਠ);
    ਜੇ (sg.ਬਫਰ.size() < buf_ਮਾਪ) {
        sg.ਬਫਰ.resize(buf_ਮਾਪ);
    }
    ਬਣਤਰ ggml_init_params params = {
        /*.mem_size   =*/ buf_ਮਾਪ,
        /*.mem_buffer =*/ sg.ਬਫਰ.ਅੰਕੜੇ(),
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ਬਣਤਰ ggml_cgraph * graph = ggml_new_ਗ੍ਰਾਫ_custom(ctx, n_nodes, ਝੂਠ);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ਅਚਲ rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    ਲਈ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ਅਚਲ ਸਵੈਚਾਲਿਤ t_deserialize = std::chrono::steady_clock::now();
    ਲਈ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&id, &nodes[i], ਆਕਾਰ(id));
        graph->nodes[i] = create_ਗੰਢ(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        ਜੇ (graph->nodes[i] == ਖਾਲੀ_ਸੂਚਕ && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (graph->nodes[i] != ਖਾਲੀ_ਸੂਚਕ) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ਅਚਲ size_t hash_pos = ggml_hash_ਪਾਓ(&graph->ਦੇਖਿਆ_hash_ਸੈੱਟ, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_ਗਿਣਤੀ;
        }
    }
    ਅਚਲ std::chrono::duration<ਦੁੱਗਣਾ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(backends[device], graph);
    GGML_ASSERT(ਸਥਿਤੀ == GGML_ਸਥਿਤੀ_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::ਗ੍ਰਾਫ_recompute(ਅਚਲ rpc_ਸੁਨੇਹਾ_ਗ੍ਰਾਫ_recompute_req & request) {
    uint32_t device = request.device;
    ਜੇ (device >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਸਵੈਚਾਲਿਤ it = stored_graphs[device].find(request.uid);
    ਜੇ (it == stored_graphs[device].end() || it->second.graph == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(backends[device], graph);
    GGML_ASSERT(ਸਥਿਤੀ == GGML_ਸਥਿਤੀ_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ਵਾਪਸ ਸੱਚ;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
ਖਾਲੀ rpc_server::sync_all_backends() {
    ਲਈ (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
ਤਾਰਕਿਕ rpc_server::comm_init(ਅਚਲ rpc_ਸੁਨੇਹਾ_comm_init_req & request, rpc_ਸੁਨੇਹਾ_comm_init_rsp & response) {
    response.ok = 0;
    ਜੇ (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ਵਾਪਸ ਸੱਚ;
    }
    comm_state & state = comm_ਹਾਲਤਾਂ[request.device];
    ਜੇ (state.peer != ਖਾਲੀ_ਸੂਚਕ) {
        response.ok = 1;
        ਵਾਪਸ ਸੱਚ;
    }
    uint8_t local_caps[RPC_CONN_CAPS_ਮਾਪ] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_ਮਾਪ] = {};
    ਜੇ (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.ਪੋਰਟ);
        ਜੇ (srv == ਖਾਲੀ_ਸੂਚਕ) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.ਪੋਰਟ);
            ਵਾਪਸ ਸੱਚ;
        }
        state.peer = srv->accept();
        ਜੇ (state.peer == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (!state.peer->recv_ਅੰਕੜੇ(remote_caps, ਆਕਾਰ(remote_caps))) {
            state.peer = ਖਾਲੀ_ਸੂਚਕ;
            ਵਾਪਸ ਸੱਚ;
        }
        state.peer->get_caps(local_caps);
        ਜੇ (!state.peer->send_ਅੰਕੜੇ(local_caps, ਆਕਾਰ(local_caps))) {
            state.peer = ਖਾਲੀ_ਸੂਚਕ;
            ਵਾਪਸ ਸੱਚ;
        }
        state.peer->update_caps(remote_caps);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ std::string host(request.host, strnlen(request.host, ਆਕਾਰ(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        ਲਈ (ਪੂਰਨ i = 0; i < 100 && state.peer == ਖਾਲੀ_ਸੂਚਕ; i++) {
            state.peer = socket_t::connect(host.c_str(), request.ਪੋਰਟ);
            ਜੇ (state.peer == ਖਾਲੀ_ਸੂਚਕ) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        ਜੇ (state.peer == ਖਾਲੀ_ਸੂਚਕ) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.ਪੋਰਟ);
            ਵਾਪਸ ਸੱਚ;
        }
        state.peer->get_caps(local_caps);
        ਜੇ (!state.peer->send_ਅੰਕੜੇ(local_caps, ਆਕਾਰ(local_caps)) ||
            !state.peer->recv_ਅੰਕੜੇ(remote_caps, ਆਕਾਰ(remote_caps))) {
            state.peer = ਖਾਲੀ_ਸੂਚਕ;
            ਵਾਪਸ ਸੱਚ;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::comm_allreduce(ਅਚਲ rpc_ਸੁਨੇਹਾ_comm_allreduce_req & request) {
    ਜੇ (request.device >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    comm_state & state = comm_ਹਾਲਤਾਂ[request.device];
    ਜੇ (state.peer == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_ਮਾਪ = 16*ggml_tensor_overhead() + 2*ggml_ਗ੍ਰਾਫ_overhead_custom(8, ਝੂਠ);
    ਬਣਤਰ ggml_init_params params = {
        /*.mem_size   =*/ ctx_ਮਾਪ,
        /*.mem_buffer =*/ ਨਿਰਾ,
        /*.no_alloc   =*/ ਸੱਚ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ਖਾਲੀ_ਸੂਚਕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    ਜੇ (t_dst == ਖਾਲੀ_ਸੂਚਕ || t_dst->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ size_t  nbytes = ggml_nbytes(t_dst);
    ਅਚਲ int64_t ne     = ggml_nelements(t_dst);
    ਜੇ (nbytes == 0) {
        ਵਾਪਸ ਸੱਚ;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ਅਚਲ ਤਾਰਕਿਕ   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ਅਚਲ size_t wire_ਬਾਈਟਾਂ = wire_bf16 ? (size_t) ne*2 : nbytes;
    ਅਚਲ size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    ਜੇ (state.scratch_ਮਾਪ < need) {
        state.scratch.reset(ggml_backend_alloc_ਬਫਰ(backend, need));
        state.scratch_ਮਾਪ = need;
    }
    ਅੱਖਰ * scratch_ਅਧਾਰ = (ਅੱਖਰ *) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(state.scratch.get());
    state.send_buf.resize(wire_ਬਾਈਟਾਂ);
    state.recv_buf.resize(wire_ਬਾਈਟਾਂ);

    ਸਵੈਚਾਲਿਤ new_scratch_tensor = [&](ggml_type type, size_t ਵਿੱਥ) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->ਬਫਰ = state.scratch.get();
        t->ਅੰਕੜੇ   = scratch_ਅਧਾਰ + ਵਿੱਥ;
        ਵਾਪਸ t;
    };
    ਸਵੈਚਾਲਿਤ new_cpy_ਗੰਢ = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->ਬਫਰ = dst->ਬਫਰ;
        t->ਅੰਕੜੇ   = dst->ਅੰਕੜੇ;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ਵਾਪਸ t;
    };
    ਸਵੈਚਾਲਿਤ compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_ਗ੍ਰਾਫ_custom(ctx, 2, ਝੂਠ);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != ਖਾਲੀ_ਸੂਚਕ ? 2 : 1;
        ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_ਗ੍ਰਾਫ_compute_async(backend, graph);
        GGML_ASSERT(ਸਥਿਤੀ == GGML_ਸਥਿਤੀ_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = ਖਾਲੀ_ਸੂਚਕ;
    ggml_tensor * t_wire_recv = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ਗੰਢ(t_dst, t_wire_send), ਖਾਲੀ_ਸੂਚਕ);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.ਅੰਕੜੇ(), 0, wire_ਬਾਈਟਾਂ);
    } ਨਹੀਂਤਾਂ {
        ggml_backend_tensor_get(t_dst, state.send_buf.ਅੰਕੜੇ(), 0, wire_ਬਾਈਟਾਂ);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    ਜੇ (state.rank == 0) {
        ਜੇ (!state.peer->send_ਅੰਕੜੇ(state.send_buf.ਅੰਕੜੇ(), wire_ਬਾਈਟਾਂ) || !state.peer->flush() ||
            !state.peer->recv_ਅੰਕੜੇ(state.recv_buf.ਅੰਕੜੇ(), wire_ਬਾਈਟਾਂ)) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (!state.peer->recv_ਅੰਕੜੇ(state.recv_buf.ਅੰਕੜੇ(), wire_ਬਾਈਟਾਂ) ||
            !state.peer->send_ਅੰਕੜੇ(state.send_buf.ਅੰਕੜੇ(), wire_ਬਾਈਟਾਂ) || !state.peer->flush()) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (wire_bf16) {
        ggml_backend_tensor_ਸੈੱਟ(t_wire_recv, state.recv_buf.ਅੰਕੜੇ(), 0, wire_ਬਾਈਟਾਂ);
        t_cast = new_cpy_ਗੰਢ(t_wire_recv, t_peer);
    } ਨਹੀਂਤਾਂ {
        ggml_backend_tensor_ਸੈੱਟ(t_peer, state.recv_buf.ਅੰਕੜੇ(), 0, wire_ਬਾਈਟਾਂ);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ਜੋੜੋ;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->ਬਫਰ = t_dst->ਬਫਰ;
    t_red->ਅੰਕੜੇ   = t_dst->ਅੰਕੜੇ;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    ਜੇ (t_cast != ਖਾਲੀ_ਸੂਚਕ) {
        compute_nodes(t_cast, t_red);
    } ਨਹੀਂਤਾਂ {
        compute_nodes(t_red, ਖਾਲੀ_ਸੂਚਕ);
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::comm_free(ਅਚਲ rpc_ਸੁਨੇਹਾ_comm_free_req & request) {
    ਜੇ (request.device >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    comm_ਹਾਲਤਾਂ[request.device] = comm_state();
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ rpc_server::get_device_ਸਮ੍ਰਿਤੀ(ਅਚਲ rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_req & request, rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_rsp & response) {
    uint32_t dev_id = request.device;
    ਜੇ (dev_id >= backends.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    size_t ਮੁਕਤ, ਕੁੱਲ;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_ਸਮ੍ਰਿਤੀ(dev, &ਮੁਕਤ, &ਕੁੱਲ);
    response.free_mem = ਮੁਕਤ;
    response.ਕੁੱਲ_mem = ਕੁੱਲ;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_ਮਾਪ(response.free_mem).c_str(), rpc_log_format_ਮਾਪ(response.ਕੁੱਲ_mem).c_str());
    ਵਾਪਸ ਸੱਚ;
}

rpc_server::~rpc_server() {
    ਲਈ (ਸਵੈਚਾਲਿਤ ਬਫਰ : buffers) {
        ggml_backend_ਬਫਰ_free(ਬਫਰ);
    }
}

ਸਥਿਰ ਖਾਲੀ rpc_serve_client(ਅਚਲ std::vector<ggml_backend_t> & backends, ਅਚਲ ਅੱਖਰ * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    ਜੇ (!sock->recv_ਅੰਕੜੇ(&cmd, 1)) {
        ਵਾਪਸ;
    }
    ਜੇ (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ਵਾਪਸ;
    }

    // Read input_size and validate protocol version
    uint64_t hello_ਨਿਵੇਸ਼_ਮਾਪ;
    ਜੇ (!sock->recv_ਅੰਕੜੇ(&hello_ਨਿਵੇਸ਼_ਮਾਪ, ਆਕਾਰ(hello_ਨਿਵੇਸ਼_ਮਾਪ))) {
        ਵਾਪਸ;
    }

    ਜੇ (hello_ਨਿਵੇਸ਼_ਮਾਪ != ਆਕਾਰ(rpc_ਸੁਨੇਹਾ_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_ਨਿਵੇਸ਼_ਮਾਪ, ਆਕਾਰ(rpc_ਸੁਨੇਹਾ_hello_req), RPC_PROTO_MAJOR_ਸੰਸਕਰਣ);
        ਵਾਪਸ;
    }

    rpc_ਸੁਨੇਹਾ_hello_req req = {};
    ਜੇ (!sock->recv_ਅੰਕੜੇ(&req, ਆਕਾਰ(req))) {
        ਵਾਪਸ;
    }

    rpc_ਸੁਨੇਹਾ_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    ਜੇ (!send_ਸੁਨੇਹਾ(sock, &rsp, ਆਕਾਰ(rsp))) {
        ਵਾਪਸ;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    ਜਦੋਂ (ਸੱਚ) {
        ਜੇ (!sock->recv_ਅੰਕੜੇ(&cmd, 1)) {
            ਰੋਕੋ;
        }
        ਜੇ (cmd >= RPC_CMD_ਗਿਣਤੀ) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            ਰੋਕੋ;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((ਗਿਣਤੀ rpc_cmd)cmd));
        ਚੋਣ (cmd) {
            ਹਾਲਤ RPC_CMD_HELLO: {
                // HELLO command is handled above
                ਵਾਪਸ;
            }
            ਹਾਲਤ RPC_CMD_DEVICE_ਗਿਣਤੀ: {
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, ਖਾਲੀ_ਸੂਚਕ, 0)) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_device_ਗਿਣਤੀ_rsp response;
                response.device_ਗਿਣਤੀ = backends.size();
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ALLOC_ਬਫਰ: {
                rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_alloc_ਬਫਰ_rsp response;
                ਜੇ (!server.alloc_ਬਫਰ(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_GET_ALLOC_ਮਾਪ: {
                rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_get_alloc_ਮਾਪ_rsp response;
                ਜੇ (!server.get_alloc_ਮਾਪ(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_GET_ALIGNMENT: {
                rpc_ਸੁਨੇਹਾ_get_alignment_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_get_alignment_rsp response;
                ਜੇ (!server.get_alignment(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_GET_MAX_ਮਾਪ: {
                rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_get_max_ਮਾਪ_rsp response;
                ਜੇ (!server.get_max_ਮਾਪ(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਬਫਰ_GET_ਅਧਾਰ: {
                rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_ਬਫਰ_get_ਅਧਾਰ_rsp response;
                ਜੇ (!server.ਬਫਰ_get_ਅਧਾਰ(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_FREE_ਬਫਰ: {
                rpc_ਸੁਨੇਹਾ_free_ਬਫਰ_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.free_ਬਫਰ(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਬਫਰ_CLEAR: {
                rpc_ਸੁਨੇਹਾ_ਬਫਰ_clear_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.ਬਫਰ_clear(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_MEMSET_TENSOR: {
                rpc_ਸੁਨੇਹਾ_memset_tensor_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.memset_tensor(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਸੈੱਟ_TENSOR: {
                std::vector<uint8_t> ਨਿਵੇਸ਼;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, ਨਿਵੇਸ਼)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.ਸੈੱਟ_tensor(ਨਿਵੇਸ਼)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਸੈੱਟ_TENSOR_2_D: {
                std::vector<uint8_t> ਨਿਵੇਸ਼;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, ਨਿਵੇਸ਼)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.ਸੈੱਟ_tensor_2d(ਨਿਵੇਸ਼)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_GET_TENSOR_2D: {
                rpc_ਸੁਨੇਹਾ_get_tensor_2d_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                std::vector<uint8_t> response;
                ਜੇ (!server.get_tensor_2d(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, response.ਅੰਕੜੇ(), response.size())) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਸੈੱਟ_TENSOR_HASH: {
                rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_ਸੈੱਟ_tensor_hash_rsp response;
                ਜੇ (!server.ਸੈੱਟ_tensor_hash(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_INIT_TENSOR: {
                rpc_ਸੁਨੇਹਾ_init_tensor_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request,ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.init_tensor(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_GET_TENSOR: {
                rpc_ਸੁਨੇਹਾ_get_tensor_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                std::vector<uint8_t> response;
                ਜੇ (!server.get_tensor(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, response.ਅੰਕੜੇ(), response.size())) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_COPY_TENSOR: {
                rpc_ਸੁਨੇਹਾ_copy_tensor_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_copy_tensor_rsp response;
                ਜੇ (!server.copy_tensor(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਗ੍ਰਾਫ_COMPUTE: {
                std::vector<uint8_t> ਨਿਵੇਸ਼;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, ਨਿਵੇਸ਼)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.ਗ੍ਰਾਫ_compute(ਨਿਵੇਸ਼)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_ਗ੍ਰਾਫ_RECOMPUTE: {
                rpc_ਸੁਨੇਹਾ_ਗ੍ਰਾਫ_recompute_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.ਗ੍ਰਾਫ_recompute(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_COMM_INIT: {
                rpc_ਸੁਨੇਹਾ_comm_init_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_comm_init_rsp response;
                ਜੇ (!server.comm_init(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_COMM_ALLREDUCE: {
                rpc_ਸੁਨੇਹਾ_comm_allreduce_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.comm_allreduce(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_COMM_FREE: {
                rpc_ਸੁਨੇਹਾ_comm_free_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                ਜੇ (!server.comm_free(request)) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਹਾਲਤ RPC_CMD_GET_DEVICE_ਸਮ੍ਰਿਤੀ: {
                rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_req request;
                ਜੇ (!recv_ਸੁਨੇਹਾ(sock, &request, ਆਕਾਰ(request))) {
                    ਵਾਪਸ;
                }
                rpc_ਸੁਨੇਹਾ_get_device_ਸਮ੍ਰਿਤੀ_rsp response;
                ਜੇ (!server.get_device_ਸਮ੍ਰਿਤੀ(request, response)) {
                    ਵਾਪਸ;
                }
                ਜੇ (!send_ਸੁਨੇਹਾ(sock, &response, ਆਕਾਰ(response))) {
                    ਵਾਪਸ;
                }
                ਰੋਕੋ;
            }
            ਮੂਲ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((ਗਿਣਤੀ rpc_cmd)cmd));
                ਵਾਪਸ;
            }
        }
    }
}

ਖਾਲੀ ggml_backend_rpc_ਸ਼ੁਰੂ_server(ਅਚਲ ਅੱਖਰ * endpoint, ਅਚਲ ਅੱਖਰ * cache_dir,
                                   size_t n_ਧਾਗੇ, size_t n_devices, ggml_backend_dev_t * devices) {
    ਜੇ (n_devices == 0 || devices == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ਵਾਪਸ;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_ਸੰਸਕਰਣ,
        RPC_PROTO_MINOR_ਸੰਸਕਰਣ,
        RPC_PROTO_PATCH_ਸੰਸਕਰਣ);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    ਲਈ (size_t i = 0; i < n_devices; i++) {
        ਸਵੈਚਾਲਿਤ dev = devices[i];
        size_t ਮੁਕਤ, ਕੁੱਲ;
        ggml_backend_dev_ਸਮ੍ਰਿਤੀ(dev, &ਮੁਕਤ, &ਕੁੱਲ);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 ਕੁੱਲ / 1024 / 1024, ਮੁਕਤ / 1024 / 1024);
        ਸਵੈਚਾਲਿਤ backend = ggml_backend_dev_init(dev, ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ਵਾਪਸ;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (reg) {
            ਸਵੈਚਾਲਿਤ ggml_backend_ਸੈੱਟ_n_ਧਾਗੇ_fn = (ggml_backend_ਸੈੱਟ_n_ਧਾਗੇ_t) ggml_backend_reg_get_proc_ਪਤਾ(reg, "ggml_backend_set_n_threads");
            ਜੇ (ggml_backend_ਸੈੱਟ_n_ਧਾਗੇ_fn) {
                ggml_backend_ਸੈੱਟ_n_ਧਾਗੇ_fn(backend, n_ਧਾਗੇ);
            }
        }
    }

    std::string host;
    ਪੂਰਨ ਪੋਰਟ;
    ਜੇ (!parse_endpoint(endpoint, host, ਪੋਰਟ)) {
        ਵਾਪਸ;
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#ਨਹੀਂਤਾਂ
    LOG_INFO("  transport      : TCP\n");
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA
    ਜੇ (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ਵਾਪਸ;
    }
    ਸਵੈਚਾਲਿਤ server_socket = socket_t::create_server(host.c_str(), ਪੋਰਟ);
    ਜੇ (server_socket == ਖਾਲੀ_ਸੂਚਕ) {
        LOG_ERROR("Failed to create server socket\n");
        ਵਾਪਸ;
    }
    ਜਦੋਂ (ਸੱਚ) {
        ਸਵੈਚਾਲਿਤ client_socket = server_socket->accept();
        ਜੇ (client_socket == ਖਾਲੀ_ਸੂਚਕ) {
            LOG_ERROR("Failed to accept client connection\n");
            ਵਾਪਸ;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    ਲਈ (ਸਵੈਚਾਲਿਤ backend : backends) {
        ggml_backend_free(backend);
    }
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ਵਾਪਸ ctx->description.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_ਸਮ੍ਰਿਤੀ(ctx->endpoint.c_str(), ctx->device, ਮੁਕਤ, ਕੁੱਲ);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                 = */ ਸੱਚ,
        /* .host_buffer           = */ ਝੂਠ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ ਸੱਚ,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ਵਾਪਸ ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_rpc_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ਵਾਪਸ ggml_backend_rpc_ਬਫਰ_type(ctx->endpoint.c_str(), ctx->device);

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(op);
    //TODO: call the remote backend and cache the results
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਜੇ (!buft || buft->iface.get_name != ggml_backend_rpc_ਬਫਰ_type_name) {
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_rpc_ਬਫਰ_type_context * buft_ctx = (ggml_backend_rpc_ਬਫਰ_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ਵਾਪਸ buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ਸਥਿਰ ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(ctx->endpoint);
    ਵਾਪਸ dispatcher->event_new(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ਨਿਰਾ,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ਨਿਰਾ,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ਬਣਤਰ ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ਵਾਪਸ ctx ? ctx->name.c_str() : "RPC";
}

ਸਥਿਰ size_t ggml_backend_rpc_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ਵਾਪਸ ctx ? ctx->devices.size() : 0;
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ਜੇ (ctx == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(ਸੂਚਕਾਂਕ < ctx->devices.size());
        ਵਾਪਸ ctx->devices[ਸੂਚਕਾਂਕ];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ਬਣਤਰ ggml_backend_rpc_comm_context {
    ਬਣਤਰ rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ਸਥਿਰ ਖਾਲੀ ggml_backend_rpc_comm_free(ਖਾਲੀ * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ਜੇ (comm_ctx == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & rank : comm_ctx->ranks) {
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, ਆਕਾਰ(*request));
        rank.dispatcher->busy_spin_release();
    }
    ਮਿਟਾਓ comm_ctx;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ਜੇ (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਜੇ (!ggml_backend_is_rpc(backends[i])) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & rank : ranks) {
            ਜੇ (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    ਪੂਰਨ port0;
    ਜੇ (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਅਚਲ uint32_t comm_ਪੋਰਟ = (uint32_t) port0 + 1000;
    ਜੇ (host0.size() >= 64) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_ਸੁਨੇਹਾ_comm_init_rsp> responses(n_backends);
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->ਪੋਰਟ   = comm_ਪੋਰਟ;
        ਜੇ (i > 0) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, ਆਕਾਰ(*request), &responses[i], ਆਕਾਰ(responses[i]));
    }
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    ਤਾਰਕਿਕ ok = ਸੱਚ;
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਜੇ (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = ਝੂਠ;
        }
    }
    ਜੇ (!ok) {
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ਵਾਪਸ ਨਵਾਂ ggml_backend_rpc_comm_context{std::move(ranks)};
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_rpc_comm_allreduce_tensor(ਖਾਲੀ * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ਜੇ (comm_ctx == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ size_t n_ranks = comm_ctx->ranks.size();
    ਅਚਲ int64_t ne = ggml_nelements(tensors[0]);
    ਜੇ (ne == 0) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਲਈ (size_t i = 0; i < n_ranks; i++) {
        ਜੇ (tensors[i] == ਖਾਲੀ_ਸੂਚਕ || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ || !ggml_backend_ਬਫਰ_is_rpc(tensors[i]->ਬਫਰ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        ਜੇ ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਲਈ (size_t i = 0; i < n_ranks; i++) {
        ਸਵੈਚਾਲਿਤ request = std::make_shared<rpc_ਸੁਨੇਹਾ_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, ਆਕਾਰ(*request));
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_rpc_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    ਜੇ (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_rpc_ਜੋੜੋ_server;
    }
    ਜੇ (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_rpc_ਸ਼ੁਰੂ_server;
    }
    ਜੇ (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_rpc_comm_init;
    }
    ਜੇ (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_rpc_comm_free;
    }
    ਜੇ (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ਵਾਪਸ ਨਿਰਾ;

    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_rpc_reg(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_rpc_reg;
}

ਸਥਿਰ uint32_t ggml_backend_rpc_get_device_ਗਿਣਤੀ(ਅਚਲ ਅੱਖਰ * endpoint) {
    ਸਵੈਚਾਲਿਤ dispatcher = get_dispatcher(endpoint);
    rpc_ਸੁਨੇਹਾ_device_ਗਿਣਤੀ_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_ਗਿਣਤੀ, ਖਾਲੀ_ਸੂਚਕ, 0, &response, ਆਕਾਰ(response));
    ਵਾਪਸ response.device_ਗਿਣਤੀ;
}

ਸਥਿਰ ਅਚਲ ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_ਗਿਣਤੀ,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_rpc_ਜੋੜੋ_server(ਅਚਲ ਅੱਖਰ * endpoint) {
    ਸਥਿਰ std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ਸਥਿਰ std::mutex mutex;
    ਸਥਿਰ uint32_t dev_id = 0;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
    ਜੇ (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ਵਾਪਸ reg_map[endpoint];
    }
    uint32_t dev_ਗਿਣਤੀ = ggml_backend_rpc_get_device_ਗਿਣਤੀ(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_ਗਿਣਤੀ);
    ਜੇ (dev_ਗਿਣਤੀ == 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ggml_backend_rpc_reg_context * ctx = ਨਵਾਂ ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    ਲਈ (uint32_t ind = 0; ind < dev_ਗਿਣਤੀ; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = ਨਵਾਂ ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = ਨਵਾਂ ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = ਨਵਾਂ ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ਵਾਪਸ reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
