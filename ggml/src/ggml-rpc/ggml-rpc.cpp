#ಅನ್ತರ್ಭಾವ "ggml-rpc.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpp.h"
#ಅನ್ತರ್ಭಾವ "transport.h"
#ಅನ್ತರ್ಭಾವ "log.h"

#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <chrono>
#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <thread>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <queue>
#ಅನ್ತರ್ಭಾವ <condition_variable>
#ಅನ್ತರ್ಭಾವ <future>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <filesystem>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <atomic>

ನಾಮಾಕಾಶ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) ಯದಿ (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(ತಳ್ಳು, 1)
// ggml_tensor is serialized into rpc_tensor
ಸಂರಚನಾ rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    ವರ್ಣ name[GGML_MAX_NAME];

    int32_t use_ಎಣಿಕೆ;
};

static_assert(ಪರಿಮಾಣ(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
ಗಣನಾ rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_ಎಣಿಕೆ,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_ಎಣಿಕೆ,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * rpc_cmd_name(ಗಣನಾ rpc_cmd cmd) {
    ವಿಕಲ್ಪ (cmd) {
        ಪಕ್ಷ RPC_CMD_ALLOC_BUFFER:      ಪ್ರತ್ಯಾಗಮ "ALLOC_BUFFER";
        ಪಕ್ಷ RPC_CMD_GET_ALIGNMENT:     ಪ್ರತ್ಯಾಗಮ "GET_ALIGNMENT";
        ಪಕ್ಷ RPC_CMD_GET_MAX_SIZE:      ಪ್ರತ್ಯಾಗಮ "GET_MAX_SIZE";
        ಪಕ್ಷ RPC_CMD_BUFFER_GET_BASE:   ಪ್ರತ್ಯಾಗಮ "BUFFER_GET_BASE";
        ಪಕ್ಷ RPC_CMD_FREE_BUFFER:       ಪ್ರತ್ಯಾಗಮ "FREE_BUFFER";
        ಪಕ್ಷ RPC_CMD_BUFFER_CLEAR:      ಪ್ರತ್ಯಾಗಮ "BUFFER_CLEAR";
        ಪಕ್ಷ RPC_CMD_SET_TENSOR:        ಪ್ರತ್ಯಾಗಮ "SET_TENSOR";
        ಪಕ್ಷ RPC_CMD_SET_TENSOR_HASH:   ಪ್ರತ್ಯಾಗಮ "SET_TENSOR_HASH";
        ಪಕ್ಷ RPC_CMD_GET_TENSOR:        ಪ್ರತ್ಯಾಗಮ "GET_TENSOR";
        ಪಕ್ಷ RPC_CMD_COPY_TENSOR:       ಪ್ರತ್ಯಾಗಮ "COPY_TENSOR";
        ಪಕ್ಷ RPC_CMD_GRAPH_COMPUTE:     ಪ್ರತ್ಯಾಗಮ "GRAPH_COMPUTE";
        ಪಕ್ಷ RPC_CMD_GET_DEVICE_MEMORY: ಪ್ರತ್ಯಾಗಮ "GET_DEVICE_MEMORY";
        ಪಕ್ಷ RPC_CMD_INIT_TENSOR:       ಪ್ರತ್ಯಾಗಮ "INIT_TENSOR";
        ಪಕ್ಷ RPC_CMD_GET_ALLOC_SIZE:    ಪ್ರತ್ಯಾಗಮ "GET_ALLOC_SIZE";
        ಪಕ್ಷ RPC_CMD_HELLO:             ಪ್ರತ್ಯಾಗಮ "HELLO";
        ಪಕ್ಷ RPC_CMD_DEVICE_ಎಣಿಕೆ:      ಪ್ರತ್ಯಾಗಮ "DEVICE_COUNT";
        ಪಕ್ಷ RPC_CMD_GRAPH_RECOMPUTE:   ಪ್ರತ್ಯಾಗಮ "GRAPH_RECOMPUTE";
        ಪಕ್ಷ RPC_CMD_MEMSET_TENSOR:     ಪ್ರತ್ಯಾಗಮ "MEMSET_TENSOR";
        ಪಕ್ಷ RPC_CMD_SET_TENSOR_2D:     ಪ್ರತ್ಯಾಗಮ "SET_TENSOR_2D";
        ಪಕ್ಷ RPC_CMD_GET_TENSOR_2D:     ಪ್ರತ್ಯಾಗಮ "GET_TENSOR_2D";
        ಪಕ್ಷ RPC_CMD_COMM_INIT:         ಪ್ರತ್ಯಾಗಮ "COMM_INIT";
        ಪಕ್ಷ RPC_CMD_COMM_ALLREDUCE:    ಪ್ರತ್ಯಾಗಮ "COMM_ALLREDUCE";
        ಪಕ್ಷ RPC_CMD_COMM_FREE:         ಪ್ರತ್ಯಾಗಮ "COMM_FREE";
        ಪಕ್ಷ RPC_CMD_NONE:              ಪ್ರತ್ಯಾಗಮ "NONE";
        ಸ್ವಭಾವ:                        ಪ್ರತ್ಯಾಗಮ "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ಧ್ರುವ size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ಧ್ರುವ size_t GRAPH_CACHE_MAX = 1024;

ಸಂರಚನಾ rpc_ಸಂದೇಶ_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_device_ಎಣಿಕೆ_rsp {
    uint32_t device_ಎಣಿಕೆ;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_alloc_size_rsp {
    uint64_t alloc_size;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_init_tensor_req {
    rpc_tensor tensor;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_alignment_req {
    uint32_t device;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_alignment_rsp {
    uint64_t alignment;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_max_size_req {
    uint32_t device;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_max_size_rsp {
    uint64_t max_size;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_buffer_get_base_req {
    uint64_t remote_ptr;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_buffer_get_base_rsp {
    uint64_t base_ptr;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_free_buffer_req {
    uint64_t remote_ptr;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_set_tensor_hash_rsp {
    uint8_t result;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_copy_tensor_rsp {
    uint8_t result;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_device_memory_req {
    uint32_t device;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    ವರ್ಣ     host[64];  // rank > 0: rank 0's host
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_comm_init_rsp {
    uint8_t ok;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ಸಂರಚನಾ rpc_ಸಂದೇಶ_comm_free_req {
    uint32_t device;
};

#pragma pack(ತೆಗೆ)

// RPC data structures

ಸ್ಥಿರ ggml_guid_t ggml_backend_rpc_guid() {
    ಸ್ಥಿರ ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ಸಂರಚನಾ ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ಸಂರಚನಾ ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

ವರ್ಗ rpc_dispatcher;
ಸಂರಚನಾ ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ಸಂರಚನಾ ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    ಶೂನ್ಯ                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ಸ್ಥಿರ uint64_t fnv_hash(ಧ್ರುವ uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ಧ್ರುವ uint64_t fnv_ಅವಿಭಾಜ್ಯ = 0x100000001b3ULL;

    ಪ್ರತಿ (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_ಅವಿಭಾಜ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ hash;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ send_ಸಂದೇಶ(socket_ptr sock, ಧ್ರುವ ಶೂನ್ಯ * ಸಂದೇಶ, size_t ಸಂದೇಶ_size) {
    ಯದಿ (!sock->send_data(&ಸಂದೇಶ_size, ಪರಿಮಾಣ(ಸಂದೇಶ_size))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!sock->send_data(ಸಂದೇಶ, ಸಂದೇಶ_size)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ sock->flush();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ recv_ಸಂದೇಶ(socket_ptr sock, ಶೂನ್ಯ * ಸಂದೇಶ, size_t ಸಂದೇಶ_size) {
    uint64_t size;
    ಯದಿ (!sock->recv_data(&size, ಪರಿಮಾಣ(size))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (size != ಸಂದೇಶ_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ sock->recv_data(ಸಂದೇಶ, ಸಂದೇಶ_size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ recv_ಸಂದೇಶ(socket_ptr sock, std::vector<uint8_t> & ಒಳಹರಿವು) {
    uint64_t size;
    ಯದಿ (!sock->recv_data(&size, ಪರಿಮಾಣ(size))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯತ್ನ {
        ಒಳಹರಿವು.resize(size);
    } ಗೃಹಾಣ (ಧ್ರುವ std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ sock->recv_data(ಒಳಹರಿವು.data(), size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ parse_endpoint(ಧ್ರುವ std::string & endpoint, std::string & host, ಪೂರ್ಣಾಙ್ಕ & port) {
    size_t pos = endpoint.find(':');
    ಯದಿ (pos == std::string::npos) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    host = endpoint.substr(0, pos);
    ಯತ್ನ {
        port = std::stoi(endpoint.substr(pos + 1));
    } ಗೃಹಾಣ (...) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ಸ್ಥಿರ ತಾರ್ಕಿಕ send_rpc_cmd(socket_ptr sock, ಗಣನಾ rpc_cmd cmd, ಧ್ರುವ ಶೂನ್ಯ * ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size) {
    uint8_t cmd_byte = cmd;
    ಯದಿ (!sock->send_data(&cmd_byte, ಪರಿಮಾಣ(cmd_byte))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!sock->send_data(&ಒಳಹರಿವು_size, ಪರಿಮಾಣ(ಒಳಹರಿವು_size))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!sock->send_data(ಒಳಹರಿವು, ಒಳಹರಿವು_size)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ಸ್ಥಿರ ತಾರ್ಕಿಕ send_rpc_cmd(socket_ptr sock, ಗಣನಾ rpc_cmd cmd, ಧ್ರುವ ಶೂನ್ಯ * ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size, ಶೂನ್ಯ * output, size_t output_size) {
    ಯದಿ (!send_rpc_cmd(sock, cmd, ಒಳಹರಿವು, ಒಳಹರಿವು_size)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    uint64_t out_size;
    ಯದಿ (!sock->recv_data(&out_size, ಪರಿಮಾಣ(out_size))) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (out_size != output_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!sock->recv_data(output, output_size)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ಸ್ಥಿರ constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ rpc_cpu_relax() {
#ಯದಿ defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ ಚಞ್ಚಲ("yield" ::: "memory");
#ಅನ್ಯಥಾ
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ಸ್ಥಿರ ತಾರ್ಕಿಕ negotiate_hello(ಧ್ರುವ std::shared_ptr<socket_t> & sock) {
    rpc_ಸಂದೇಶ_hello_req request = {};
    rpc_ಸಂದೇಶ_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    ತಾರ್ಕಿಕ status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, ಪರಿಮಾಣ(request), &response, ಪರಿಮಾಣ(response));
    RPC_STATUS_ASSERT(status);

    ಯದಿ (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

template <typename T>
ವರ್ಗ message_queue {
ಸಾರ್ವಜನಿಕ:
    message_queue() {}

    ತಾರ್ಕಿಕ ತಳ್ಳು(ಧ್ರುವ T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        ಯದಿ (interrupted) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        queue.ತಳ್ಳು(value);
        cvar.notify_all();
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ ತೆಗೆ(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [ಏತತ್] { ಪ್ರತ್ಯಾಗಮ !queue.empty() || interrupted; });
        ಯದಿ (interrupted) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        *out = queue.front();
        queue.ತೆಗೆ();
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_ತೆಗೆ(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        ಯದಿ (interrupted || queue.empty()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        *out = queue.front();
        queue.ತೆಗೆ();
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = ಸತ್ಯ;
        lock.unlock();
        cvar.notify_all();
    }

ಗುಪ್ತ:
    ತಾರ್ಕಿಕ interrupted = ಅಸತ್ಯ;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

ವರ್ಗ rpc_dispatcher {
ಸಾರ್ವಜನಿಕ:
    rpc_dispatcher() {
    }

    ಶೂನ್ಯ send(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size);
    ಶೂನ್ಯ send(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size, ಶೂನ್ಯ * output, size_t output_size);
    ಶೂನ್ಯ send_async(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size);
    ಶೂನ್ಯ send_async(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size, ಶೂನ್ಯ * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    ಶೂನ್ಯ event_free(ggml_backend_event_t event);
    ಶೂನ್ಯ event_synchronize(ggml_backend_event_t event);
    ಶೂನ್ಯ event_record(ggml_backend_event_t event);
    ಶೂನ್ಯ synchronize();
    ಶೂನ್ಯ busy_spin_acquire();
    ಶೂನ್ಯ busy_spin_release();
    ಶೂನ್ಯ graph_compute(uint32_t device, ಧ್ರುವ ggml_cgraph * cgraph);

    ಶೂನ್ಯ start(ಧ್ರುವ std::string & endpoint);
    ಶೂನ್ಯ work();

    ~rpc_dispatcher();

ಗುಪ್ತ:
    ಸಂರಚನಾ rpc_ಸಂದೇಶ {
        rpc_cmd                       cmd;
        std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ>   ಒಳಹರಿವು;
        size_t                        ಒಳಹರಿವು_size;
        ಶೂನ್ಯ                        * output;
        size_t                        output_size;
        std::promise<ಶೂನ್ಯ>            completion;
    };
    using rpc_ಸಂದೇಶ_ptr   = std::shared_ptr<rpc_ಸಂದೇಶ>;
    using rpc_ಸಂದೇಶ_queue = message_queue<rpc_ಸಂದೇಶ_ptr>;
    ಸಂರಚನಾ rpc_event {
        rpc_ಸಂದೇಶ_ptr              ಸಂದೇಶ;
        std::shared_future<ಶೂನ್ಯ> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_ಸಂದೇಶ_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

ಶೂನ್ಯ rpc_dispatcher::send(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size) {
    auto ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ->cmd = cmd;
    ಸಂದೇಶ->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ->ಒಳಹರಿವು_size = ಒಳಹರಿವು_size;
    ಸಂದೇಶ->output = nullptr;
    ಸಂದೇಶ->output_size = 0;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ));
    auto future = ಸಂದೇಶ->completion.get_future();
    future.wait();
}

ಶೂನ್ಯ rpc_dispatcher::send_async(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size) {
    auto ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ->cmd = cmd;
    ಸಂದೇಶ->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ->ಒಳಹರಿವು_size = ಒಳಹರಿವು_size;
    ಸಂದೇಶ->output = nullptr;
    ಸಂದೇಶ->output_size = 0;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ));
}

ಶೂನ್ಯ rpc_dispatcher::send(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size, ಶೂನ್ಯ * output, size_t output_size) {
    auto ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ->cmd = cmd;
    ಸಂದೇಶ->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ->ಒಳಹರಿವು_size = ಒಳಹರಿವು_size;
    ಸಂದೇಶ->output = output;
    ಸಂದೇಶ->output_size = output_size;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ));
    auto future = ಸಂದೇಶ->completion.get_future();
    future.wait();
}

ಶೂನ್ಯ rpc_dispatcher::send_async(ಗಣನಾ rpc_cmd cmd, std::shared_ptr<ಧ್ರುವ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_size, ಶೂನ್ಯ * output, size_t output_size) {
    auto ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ->cmd = cmd;
    ಸಂದೇಶ->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ->ಒಳಹರಿವು_size = ಒಳಹರಿವು_size;
    ಸಂದೇಶ->output = output;
    ಸಂದೇಶ->output_size = output_size;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = ನೂತನ rpc_event;
    ev->ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ev->ಸಂದೇಶ->cmd = RPC_CMD_NONE;
    ev->sf = ev->ಸಂದೇಶ->completion.get_future().share();
    GGML_ASSERT(queue.ತಳ್ಳು(ev->ಸಂದೇಶ));
    ಪ್ರತ್ಯಾಗಮ ನೂತನ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

ಶೂನ್ಯ rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

ಶೂನ್ಯ rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

ಶೂನ್ಯ rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ev->ಸಂದೇಶ->cmd = RPC_CMD_NONE;
    ev->sf = ev->ಸಂದೇಶ->completion.get_future().share();
    GGML_ASSERT(queue.ತಳ್ಳು(ev->ಸಂದೇಶ));
}

ಶೂನ್ಯ rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto ಸಂದೇಶ = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ));
    ಸಂದೇಶ->completion.get_future().wait();
}

ಶೂನ್ಯ rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

ಶೂನ್ಯ rpc_dispatcher::busy_spin_release() {
    ಧ್ರುವ unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

ಶೂನ್ಯ rpc_dispatcher::start(ಧ್ರುವ std::string & endpoint) {
    std::string host;
    ಪೂರ್ಣಾಙ್ಕ port;
    ಯದಿ (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    ಯದಿ (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    ಯದಿ (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    ಯದಿ (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = ಸತ್ಯ;
    thread = std::thread(rpc_dispatcher_trampoline, ಏತತ್);
}

ಶೂನ್ಯ rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    ಯಾವತ್ (running) {
        rpc_ಸಂದೇಶ_ptr ಸಂದೇಶ_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ಧ್ರುವ ತಾರ್ಕಿಕ spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        ಯದಿ (spin) {
            ಯದಿ (!queue.try_ತೆಗೆ(&ಸಂದೇಶ_ptr)) {
                rpc_cpu_relax();
                ಅನುವರ್ತ;
            }
        } ಅನ್ಯಥಾ ಯದಿ (!queue.ತೆಗೆ(&ಸಂದೇಶ_ptr)) {
            ಭಙ್ಗ;
        }
        ಯದಿ (ಸಂದೇಶ_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(ಸಂದೇಶ_ptr->cmd),
                     rpc_log_format_size(ಸಂದೇಶ_ptr->ಒಳಹರಿವು_size).c_str(), rpc_log_format_size(ಸಂದೇಶ_ptr->output_size).c_str());
            ಧ್ರುವ ತಾರ್ಕಿಕ trace = rpc_log_debug_level() >= 3;
            ಧ್ರುವ auto t0 = std::chrono::steady_clock::now();
            ಯದಿ (ಸಂದೇಶ_ptr->output) {
                ತಾರ್ಕಿಕ status = send_rpc_cmd(sock, ಸಂದೇಶ_ptr->cmd, ಸಂದೇಶ_ptr->ಒಳಹರಿವು.get(), ಸಂದೇಶ_ptr->ಒಳಹರಿವು_size, ಸಂದೇಶ_ptr->output, ಸಂದೇಶ_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } ಅನ್ಯಥಾ {
                ತಾರ್ಕಿಕ status = send_rpc_cmd(sock, ಸಂದೇಶ_ptr->cmd, ಸಂದೇಶ_ptr->ಒಳಹರಿವು.get(), ಸಂದೇಶ_ptr->ಒಳಹರಿವು_size);
                RPC_STATUS_ASSERT(status);
            }
            ಯದಿ (trace) {
                ಧ್ರುವ std::chrono::duration<ದ್ವಿಗುಣ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(ಸಂದೇಶ_ptr->cmd), elapsed.count());
            }
        } ಅನ್ಯಥಾ {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        ಸಂದೇಶ_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = ಅಸತ್ಯ;
    queue.interrupt();
    sock = nullptr;
    ಯದಿ (thread.joinable()) {
        thread.join();
    }
}

ಸ್ಥಿರ std::shared_ptr<rpc_dispatcher> get_dispatcher(ಧ್ರುವ std::string & endpoint) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    ಸ್ಥಿರ std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    ಯದಿ (it != dispatchers.end()) {
        ಯದಿ (auto dispatcher = it->second.lock()) {
            ಪ್ರತ್ಯಾಗಮ dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ಪ್ರತ್ಯಾಗಮ dispatcher;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_ಸಂದೇಶ_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, ಪರಿಮಾಣ(*request));
    delete ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ಯದಿ (ctx->base_ptr != nullptr) {
        ಪ್ರತ್ಯಾಗಮ ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_ಸಂದೇಶ_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_ಸಂದೇಶ_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
    ctx->base_ptr = reinterpret_cast<ಶೂನ್ಯ *>(response.base_ptr);
    ಪ್ರತ್ಯಾಗಮ ctx->base_ptr;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

ಸ್ಥಿರ rpc_tensor serialize_tensor(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    ಯದಿ (!tensor) {
        memset(&result, 0, ಪರಿಮಾಣ(result));
        ಪ್ರತ್ಯಾಗಮ result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ಯದಿ (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        ಯದಿ (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } ಅನ್ಯಥಾ {
            result.buffer = 0;
            result.data = 0;
        }
    } ಅನ್ಯಥಾ {
        result.buffer = 0;
        result.data   = 0;
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
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, ಪರಿಮಾಣ(result.name));
    result.use_ಎಣಿಕೆ = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    ಯದಿ (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_ಸಂದೇಶ_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, ಪರಿಮಾಣ(*request));
    }
    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_ಸಂದೇಶ_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, ಪರಿಮಾಣ(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ಸ್ಥಿರ std::shared_ptr<uint8_t> serialize_set_tensor(ಧ್ರುವ rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, ಧ್ರುವ ಶೂನ್ಯ * data, size_t size, size_t & ಒಳಹರಿವು_size) {
    ಒಳಹರಿವು_size = ಪರಿಮಾಣ(rpc_tensor) + ಪರಿಮಾಣ(cache_flag) + ಪರಿಮಾಣ(offset) + size;
    uint8_t * ಒಳಹರಿವು = ನೂತನ uint8_t[ಒಳಹರಿವು_size]();
    uint8_t * p = ಒಳಹರಿವು;
    memcpy(p, &rpc_tensor, ಪರಿಮಾಣ(rpc_tensor)); p += ಪರಿಮಾಣ(rpc_tensor);
    memcpy(p, &cache_flag, ಪರಿಮಾಣ(cache_flag)); p += ಪರಿಮಾಣ(cache_flag);
    memcpy(p, &offset,     ಪರಿಮಾಣ(offset));     p += ಪರಿಮಾಣ(offset);
    memcpy(p, data, size);
    ಪ್ರತ್ಯಾಗಮ std::shared_ptr<uint8_t>(ಒಳಹರಿವು, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ಸ್ಥಿರ ತಾರ್ಕಿಕ rpc_use_hash_cache(ಧ್ರುವ ggml_tensor * tensor, size_t size) {
    ಪ್ರತ್ಯಾಗಮ size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    ಯದಿ (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_ಸಂದೇಶ_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ಧ್ರುವ uint8_t*)data, size);
        rpc_ಸಂದೇಶ_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        ಯದಿ (response.result) {
            // the server has the same data, no need to send it
            ಪ್ರತ್ಯಾಗಮ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ಒಳಹರಿವು_size;
    auto ಒಳಹರಿವು = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, ಒಳಹರಿವು_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, ಒಳಹರಿವು, ಒಳಹರಿವು_size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t ಒಳಹರಿವು_size = ಪರಿಮಾಣ(rpc_tensor) + 4*ಪರಿಮಾಣ(uint64_t) + size*n_copies;
    uint8_t * ಒಳಹರಿವು = ನೂತನ uint8_t[ಒಳಹರಿವು_size]();
    uint8_t * dest = ಒಳಹರಿವು;
    memcpy(dest, &rpc_tensor, ಪರಿಮಾಣ(rpc_tensor));
    dest += ಪರಿಮಾಣ(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, ಪರಿಮಾಣ(header));
    dest += ಪರಿಮಾಣ(header);
    ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (ಧ್ರುವ ವರ್ಣ *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> ಒಳಹರಿವು_ptr(ಒಳಹರಿವು, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, ಒಳಹರಿವು_ptr, ಒಳಹರಿವು_size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_ಸಂದೇಶ_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    ಯದಿ (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ಪರಿಮಾಣ(*request), data, size*n_copies);
    } ಅನ್ಯಥಾ {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ಪರಿಮಾಣ(*request), packed.data(), packed.size());
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            memcpy((ವರ್ಣ *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_ಸಂದೇಶ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, ಪರಿಮಾಣ(*request), data, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಯದಿ (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        ಯದಿ (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_ಸಂದೇಶ_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_ಸಂದೇಶ_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
        ಪ್ರತ್ಯಾಗಮ response.result;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_ಸಂದೇಶ_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, ಪರಿಮಾಣ(*request));
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ಪ್ರತ್ಯಾಗಮ buft_ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_ಸಂದೇಶ_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_ಸಂದೇಶ_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
    ಯದಿ (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            ನೂತನ ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        ಪ್ರತ್ಯಾಗಮ buffer;
    } ಅನ್ಯಥಾ {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

ಸ್ಥಿರ size_t get_alignment(ಧ್ರುವ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_ಸಂದೇಶ_get_alignment_req>();
    request->device = device;
    rpc_ಸಂದೇಶ_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    ಪ್ರತ್ಯಾಗಮ response.alignment;
}

ಸ್ಥಿರ size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ಪ್ರತ್ಯಾಗಮ buft_ctx->alignment;
}

ಸ್ಥಿರ size_t get_max_size(ಧ್ರುವ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_ಸಂದೇಶ_get_max_size_req>();
    request->device = device;
    rpc_ಸಂದೇಶ_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    ಪ್ರತ್ಯಾಗಮ response.max_size;
}

ಸ್ಥಿರ size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ಪ್ರತ್ಯಾಗಮ buft_ctx->max_size;
}

ಸ್ಥಿರ size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    ತಾರ್ಕಿಕ rpc_get = ಅಸತ್ಯ;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    ಯದಿ (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ಧ್ರುವ size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ಸಂರಚನಾ alloc_size_cache_ಕೀಲಿ {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.device = buft_ctx->device;
        ಕೀಲಿ.type = tensor->type;
        ಕೀಲಿ.op = tensor->op;
        memcpy(ಕೀಲಿ.op_params, tensor->op_params, ಪರಿಮಾಣ(ಕೀಲಿ.op_params));
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
            ಕೀಲಿ.ne[i] = (uint32_t)tensor->ne[i];
            ಕೀಲಿ.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ಧ್ರುವ uint8_t *)&ಕೀಲಿ, ಪರಿಮಾಣ(ಕೀಲಿ));
        cache_hash = fnv_hash((ಧ್ರುವ uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ಸ್ಥಿರ std::mutex cache_mutex;
        ಸ್ಥಿರ std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            ಯದಿ (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                ಪ್ರತ್ಯಾಗಮ std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_ಸಂದೇಶ_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_ಸಂದೇಶ_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        ಪ್ರತ್ಯಾಗಮ std::max<size_t>(response.alloc_size, min_size);
    }

    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);
}

ಸ್ಥಿರ ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
    /* .is_host             = */ ರಿಕ್ತ,
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ಪ್ರತ್ಯಾಗಮ rpc_ctx->name.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    ಯದಿ (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_ಸಂದೇಶ_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ಧ್ರುವ uint8_t*)data, size);
        rpc_ಸಂದೇಶ_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        ಯದಿ (response.result) {
            // the server has the same data, no need to send it
            ಪ್ರತ್ಯಾಗಮ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ಒಳಹರಿವು_size;
    auto ಒಳಹರಿವು = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, ಒಳಹರಿವು_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, ಒಳಹರಿವು, ಒಳಹರಿವು_size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_ಸಂದೇಶ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, ಪರಿಮಾಣ(*request), data, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ಸ್ಥಿರ ಶೂನ್ಯ add_tensor(ggml_tensor * tensor, ಧ್ರುವ ggml_cgraph * cgraph, ಧ್ರುವ rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    ಯದಿ (tensor == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (visited.find(tensor) != visited.end()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    visited.insert(tensor);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    ಧ್ರುವ size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    ಯದಿ (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_ಎಣಿಕೆ = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

ಸ್ಥಿರ uint8_t * serialize_graph(uint32_t device, ಧ್ರುವ ggml_cgraph * cgraph, ಧ್ರುವ rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    ಪ್ರತಿ (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*ಪರಿಮಾಣ(uint32_t) + ಪರಿಮಾಣ(uint64_t) + n_nodes * ಪರಿಮಾಣ(uint64_t) + ಪರಿಮಾಣ(uint32_t) + n_tensors * ಪರಿಮಾಣ(rpc_tensor);
    uint8_t * output = ನೂತನ uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, ಪರಿಮಾಣ(device));
    dest += ಪರಿಮಾಣ(device);
    memcpy(dest, &cgraph->uid, ಪರಿಮಾಣ(cgraph->uid));
    dest += ಪರಿಮಾಣ(cgraph->uid);
    memcpy(dest, &n_nodes, ಪರಿಮಾಣ(n_nodes));
    dest += ಪರಿಮಾಣ(n_nodes);
    ಪ್ರತಿ (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * ಪರಿಮಾಣ(uint64_t), &cgraph->nodes[i], ಪರಿಮಾಣ(uint64_t));
    }
    dest += n_nodes * ಪರಿಮಾಣ(uint64_t);
    memcpy(dest, &n_tensors, ಪರಿಮಾಣ(n_tensors));
    dest += ಪರಿಮಾಣ(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * ಪರಿಮಾಣ(rpc_tensor));
    ಪ್ರತ್ಯಾಗಮ output;
}

ಶೂನ್ಯ rpc_dispatcher::graph_compute(uint32_t device, ಧ್ರುವ ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    ತಾರ್ಕಿಕ reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    ಯದಿ (reuse) {
        auto request = std::make_shared<rpc_ಸಂದೇಶ_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, ಪರಿಮಾಣ(*request));
    } ಅನ್ಯಥಾ {
        ಯದಿ (cgraph->uid != 0) {
            ಯದಿ (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t ಒಳಹರಿವು_size = 0;
        uint8_t * ಒಳಹರಿವು = serialize_graph(device, cgraph, ಏತತ್, &ಒಳಹರಿವು_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(ಒಳಹರಿವು_size).c_str());
        std::shared_ptr<uint8_t> ಒಳಹರಿವು_ptr(ಒಳಹರಿವು, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, ಒಳಹರಿವು_ptr, ಒಳಹರಿವು_size);
    }
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ಧ್ರುವ ggml_tensor * tensor, ಶೂನ್ಯ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ರಿಕ್ತ,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ರಿಕ್ತ,
    /* .graph_plan_free         = */ ರಿಕ್ತ,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ರಿಕ್ತ,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ರಿಕ್ತ,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(ಧ್ರುವ ವರ್ಣ * endpoint, uint32_t device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ಸ್ಥಿರ std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    ಯದಿ (it != buft_map.end()) {
        ಪ್ರತ್ಯಾಗಮ it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = ನೂತನ ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = ನೂತನ ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    ಪ್ರತ್ಯಾಗಮ buft;
}

ggml_backend_t ggml_backend_rpc_init(ಧ್ರುವ ವರ್ಣ * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = ನೂತನ ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = ನೂತನ ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ಪ್ರತ್ಯಾಗಮ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_rpc(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend != ರಿಕ್ತ && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

ಶೂನ್ಯ ggml_backend_rpc_get_device_memory(ಧ್ರುವ ವರ್ಣ * endpoint, uint32_t device, size_t * ಮುಕ್ತ, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_ಸಂದೇಶ_get_device_memory_req>();
    request->device = device;
    rpc_ಸಂದೇಶ_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, ಪರಿಮಾಣ(*request), &response, ಪರಿಮಾಣ(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *ಮುಕ್ತ = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

ವರ್ಗ rpc_server {
ಸಾರ್ವಜನಿಕ:
    rpc_server(std::vector<ggml_backend_t> all_backends, ಧ್ರುವ ವರ್ಣ * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    ಶೂನ್ಯ hello(rpc_ಸಂದೇಶ_hello_rsp & response);
    ತಾರ್ಕಿಕ alloc_buffer(ಧ್ರುವ rpc_ಸಂದೇಶ_alloc_buffer_req & request, rpc_ಸಂದೇಶ_alloc_buffer_rsp & response);
    ತಾರ್ಕಿಕ get_alignment(ಧ್ರುವ rpc_ಸಂದೇಶ_get_alignment_req & request, rpc_ಸಂದೇಶ_get_alignment_rsp & response);
    ತಾರ್ಕಿಕ get_max_size(ಧ್ರುವ rpc_ಸಂದೇಶ_get_max_size_req & request, rpc_ಸಂದೇಶ_get_max_size_rsp & response);
    ತಾರ್ಕಿಕ buffer_get_base(ಧ್ರುವ rpc_ಸಂದೇಶ_buffer_get_base_req & request, rpc_ಸಂದೇಶ_buffer_get_base_rsp & response);
    ತಾರ್ಕಿಕ free_buffer(ಧ್ರುವ rpc_ಸಂದೇಶ_free_buffer_req & request);
    ತಾರ್ಕಿಕ buffer_clear(ಧ್ರುವ rpc_ಸಂದೇಶ_buffer_clear_req & request);
    ತಾರ್ಕಿಕ memset_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_memset_tensor_req & request);
    ತಾರ್ಕಿಕ set_tensor(ಧ್ರುವ std::vector<uint8_t> & ಒಳಹರಿವು);
    ತಾರ್ಕಿಕ set_tensor_2d(ಧ್ರುವ std::vector<uint8_t> & ಒಳಹರಿವು);
    ತಾರ್ಕಿಕ set_tensor_hash(ಧ್ರುವ rpc_ಸಂದೇಶ_set_tensor_hash_req & request, rpc_ಸಂದೇಶ_set_tensor_hash_rsp & response);
    ತಾರ್ಕಿಕ get_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_get_tensor_req & request, std::vector<uint8_t> & response);
    ತಾರ್ಕಿಕ get_tensor_2d(ಧ್ರುವ rpc_ಸಂದೇಶ_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    ತಾರ್ಕಿಕ copy_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_copy_tensor_req & request, rpc_ಸಂದೇಶ_copy_tensor_rsp & response);
    ತಾರ್ಕಿಕ graph_compute(ಧ್ರುವ std::vector<uint8_t> & ಒಳಹರಿವು);
    ತಾರ್ಕಿಕ graph_recompute(ಧ್ರುವ rpc_ಸಂದೇಶ_graph_recompute_req & request);
    ತಾರ್ಕಿಕ comm_init(ಧ್ರುವ rpc_ಸಂದೇಶ_comm_init_req & request, rpc_ಸಂದೇಶ_comm_init_rsp & response);
    ತಾರ್ಕಿಕ comm_allreduce(ಧ್ರುವ rpc_ಸಂದೇಶ_comm_allreduce_req & request);
    ತಾರ್ಕಿಕ comm_free(ಧ್ರುವ rpc_ಸಂದೇಶ_comm_free_req & request);
    ತಾರ್ಕಿಕ init_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_init_tensor_req & request);
    ತಾರ್ಕಿಕ get_alloc_size(ಧ್ರುವ rpc_ಸಂದೇಶ_get_alloc_size_req & request, rpc_ಸಂದೇಶ_get_alloc_size_rsp & response);
    ತಾರ್ಕಿಕ get_device_memory(ಧ್ರುವ rpc_ಸಂದೇಶ_get_device_memory_req & request, rpc_ಸಂದೇಶ_get_device_memory_rsp & response);

    ಸಂರಚನಾ stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

ಗುಪ್ತ:
    ಶೂನ್ಯ sync_all_backends();
    ತಾರ್ಕಿಕ get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(ಸಂರಚನಾ ggml_context * ctx, ಧ್ರುವ rpc_tensor * tensor);
    ggml_tensor * create_ಗಂಟು(uint64_t id,
                              ಸಂರಚನಾ ggml_context * ctx,
                              ಧ್ರುವ std::unordered_map<uint64_t, ಧ್ರುವ rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ಸಂರಚನಾ ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ಸಂರಚನಾ comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ಧ್ರುವ ವರ್ಣ * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

ಶೂನ್ಯ rpc_server::hello(rpc_ಸಂದೇಶ_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

ತಾರ್ಕಿಕ rpc_server::get_alloc_size(ಧ್ರುವ rpc_ಸಂದೇಶ_get_alloc_size_req & request, rpc_ಸಂದೇಶ_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    ಯದಿ (dev_id >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_buffer_type_t buft;
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (ಶೂನ್ಯ*)tensor->buffer, tensor->data);
    ಯದಿ (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } ಅನ್ಯಥಾ {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::alloc_buffer(ಧ್ರುವ rpc_ಸಂದೇಶ_alloc_buffer_req & request, rpc_ಸಂದೇಶ_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    ಯದಿ (dev_id >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    ಯದಿ (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } ಅನ್ಯಥಾ {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::get_alignment(ಧ್ರುವ rpc_ಸಂದೇಶ_get_alignment_req & request, rpc_ಸಂದೇಶ_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    ಯದಿ (dev_id >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::get_max_size(ಧ್ರುವ rpc_ಸಂದೇಶ_get_max_size_req & request, rpc_ಸಂದೇಶ_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    ಯದಿ (dev_id >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::buffer_get_base(ಧ್ರುವ rpc_ಸಂದೇಶ_buffer_get_base_req & request, rpc_ಸಂದೇಶ_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    ಯದಿ (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಶೂನ್ಯ * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::free_buffer(ಧ್ರುವ rpc_ಸಂದೇಶ_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    ಯದಿ (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    ಪ್ರತಿ (auto & sgs : stored_graphs) {
        ಪ್ರತಿ (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::buffer_clear(ಧ್ರುವ rpc_ಸಂದೇಶ_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    ಯದಿ (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::memset_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_memset_tensor_req & request) {
    sync_all_backends();
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint64_t tensor_size = ggml_nbytes(tensor);
    ಯದಿ (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    ಧ್ರುವ uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    ಯದಿ (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ uint64_t data_offset = request.tensor.data - buffer_start;
    ಯದಿ (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ggml_tensor * rpc_server::deserialize_tensor(ಸಂರಚನಾ ggml_context * ctx, ಧ್ರುವ rpc_tensor * tensor) {
    // Validate tensor type before using it
    ಯದಿ (tensor->type >= GGML_TYPE_ಎಣಿಕೆ) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    ಯದಿ (ggml_blck_size((ಗಣನಾ ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    ಯದಿ (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತಿ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    ಯದಿ (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    ಯದಿ (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        ಯದಿ (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    ಪ್ರತಿ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<ಶೂನ್ಯ *>(tensor->data);
    ggml_set_name(result, tensor->name);
    ಪ್ರತ್ಯಾಗಮ result;
}


ತಾರ್ಕಿಕ rpc_server::set_tensor(ಧ್ರುವ std::vector<uint8_t> & ಒಳಹರಿವು) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    ಧ್ರುವ size_t header_size = ಪರಿಮಾಣ(rpc_tensor) + ಪರಿಮಾಣ(cache_flag) + ಪರಿಮಾಣ(offset);
    ಯದಿ (ಒಳಹರಿವು.size() < header_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ rpc_tensor * in_tensor = (ಧ್ರುವ rpc_tensor *)ಒಳಹರಿವು.data();
    memcpy(&cache_flag, ಒಳಹರಿವು.data() + ಪರಿಮಾಣ(rpc_tensor), ಪರಿಮಾಣ(cache_flag));
    memcpy(&offset,     ಒಳಹರಿವು.data() + ಪರಿಮಾಣ(rpc_tensor) + ಪರಿಮಾಣ(cache_flag), ಪರಿಮಾಣ(offset));
    ಧ್ರುವ size_t size = ಒಳಹರಿವು.size() - header_size;

    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    ಯದಿ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        ಧ್ರುವ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ಧ್ರುವ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ಯದಿ (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ಶೂನ್ಯ * data = ಒಳಹರಿವು.data() + header_size;
    ಯದಿ (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ಧ್ರುವ uint8_t*)data, size);
        ವರ್ಣ hash_str[17];
        snprintf(hash_str, ಪರಿಮಾಣ(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.ಲಿಖ((ಧ್ರುವ ವರ್ಣ *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::set_tensor_2d(ಧ್ರುವ std::vector<uint8_t> & ಒಳಹರಿವು) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    ಯದಿ (ಒಳಹರಿವು.size() < ಪರಿಮಾಣ(rpc_tensor) + 4*ಪರಿಮಾಣ(uint64_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ rpc_tensor * in_tensor = (ಧ್ರುವ rpc_tensor *)ಒಳಹರಿವು.data();
    uint64_t header[4];
    memcpy(header, ಒಳಹರಿವು.data() + ಪರಿಮಾಣ(rpc_tensor), ಪರಿಮಾಣ(header));
    ಧ್ರುವ uint64_t offset   = header[0];
    ಧ್ರುವ uint64_t size     = header[1];
    ಧ್ರುವ uint64_t n_copies = header[2];
    ಧ್ರುವ uint64_t stride   = header[3];

    ಧ್ರುವ uint64_t data_size = ಒಳಹರಿವು.size() - ಪರಿಮಾಣ(rpc_tensor) - 4*ಪರಿಮಾಣ(uint64_t);
    ಯದಿ (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    ಯದಿ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        ಯದಿ (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ uint64_t span = (n_copies - 1)*stride + size;
        ಧ್ರುವ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ಧ್ರುವ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ಯದಿ (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ಶೂನ್ಯ * data = ಒಳಹರಿವು.data() + ಪರಿಮಾಣ(rpc_tensor) + 4*ಪರಿಮಾಣ(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    ಯದಿ (!cache_dir) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ವರ್ಣ hash_str[17];
    snprintf(hash_str, ಪರಿಮಾಣ(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    ಯದಿ (!fs::exists(cache_file, ec)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.ಪಠ((ವರ್ಣ *)data.data(), size);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::set_tensor_hash(ಧ್ರುವ rpc_ಸಂದೇಶ_set_tensor_hash_req & request, rpc_ಸಂದೇಶ_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    std::vector<uint8_t> cached_file;
    ಯದಿ (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ಧ್ರುವ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ಧ್ರುವ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ಯದಿ (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::init_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_init_tensor_req & request) {
    sync_all_backends();
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (ಶೂನ್ಯ*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    ಯದಿ (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } ಅನ್ಯಥಾ {
        ಯದಿ (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    ಯದಿ (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::get_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        ಧ್ರುವ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ಧ್ರುವ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ಯದಿ (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::get_tensor_2d(ಧ್ರುವ rpc_ಸಂದೇಶ_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        ಯದಿ (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        ಧ್ರುವ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ಧ್ರುವ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ಯದಿ (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::copy_tensor(ಧ್ರುವ rpc_ಸಂದೇಶ_copy_tensor_req & request, rpc_ಸಂದೇಶ_copy_tensor_rsp & response) {
    sync_all_backends();
    ಸಂರಚನಾ ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    ಯದಿ (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    ಯದಿ (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ggml_tensor * rpc_server::create_ಗಂಟು(uint64_t id,
                                      ಸಂರಚನಾ ggml_context * ctx,
                                      ಧ್ರುವ std::unordered_map<uint64_t, ಧ್ರುವ rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ಸಂರಚನಾ ggml_tensor*> & tensor_map) {
    ಯದಿ (tensor_map.find(id) != tensor_map.end()) {
        ಪ್ರತ್ಯಾಗಮ tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    ಯದಿ (it_ptr == tensor_ptrs.end()) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಧ್ರುವ rpc_tensor * tensor = it_ptr->second;

    ಸಂರಚನಾ ggml_tensor * result = deserialize_tensor(ctx, tensor);
    ಯದಿ (result == nullptr) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    tensor_map[id] = result;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        ಯದಿ (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } ಅನ್ಯಥಾ {
            result->src[i] = create_ಗಂಟು(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            ಯದಿ (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }
    }

    // Handle view_src similarly
    ಯದಿ (tensor->view_src == 0) {
        result->view_src = nullptr;
    } ಅನ್ಯಥಾ {
        result->view_src = create_ಗಂಟು(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        ಯದಿ (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    ಪ್ರತ್ಯಾಗಮ result;
}

ತಾರ್ಕಿಕ rpc_server::graph_compute(ಧ್ರುವ std::vector<uint8_t> & ಒಳಹರಿವು) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    ಯದಿ (ಒಳಹರಿವು.size() < 2*ಪರಿಮಾಣ(uint32_t) + ಪರಿಮಾಣ(uint64_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ uint8_t * src = ಒಳಹರಿವು.data();
    uint32_t device;
    memcpy(&device, src, ಪರಿಮಾಣ(device));
    src += ಪರಿಮಾಣ(device);
    ಯದಿ (device >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    uint64_t uid;
    memcpy(&uid, src, ಪರಿಮಾಣ(uid));
    src += ಪರಿಮಾಣ(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, ಪರಿಮಾಣ(n_nodes));
    src += ಪರಿಮಾಣ(n_nodes);
    ಯದಿ (ಒಳಹರಿವು.size() < 2*ಪರಿಮಾಣ(uint32_t) + ಪರಿಮಾಣ(uint64_t) + n_nodes*ಪರಿಮಾಣ(uint64_t) + ಪರಿಮಾಣ(uint32_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ uint64_t * nodes = (ಧ್ರುವ uint64_t *)src;
    src += n_nodes*ಪರಿಮಾಣ(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, ಪರಿಮಾಣ(n_tensors));
    src += ಪರಿಮಾಣ(n_tensors);
    ಯದಿ (ಒಳಹರಿವು.size() < 2*ಪರಿಮಾಣ(uint32_t) + ಪರಿಮಾಣ(uint64_t) + n_nodes*ಪರಿಮಾಣ(uint64_t) + ಪರಿಮಾಣ(uint32_t) + n_tensors*ಪರಿಮಾಣ(rpc_tensor)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ rpc_tensor * tensors = (ಧ್ರುವ rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    ಯದಿ (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, ಅಸತ್ಯ);
    ಯದಿ (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    ಸಂರಚನಾ ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ಸಂರಚನಾ ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, ಅಸತ್ಯ);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ಧ್ರುವ rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    ಪ್ರತಿ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ಧ್ರುವ auto t_deserialize = std::chrono::steady_clock::now();
    ಪ್ರತಿ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], ಪರಿಮಾಣ(id));
        graph->nodes[i] = create_ಗಂಟು(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        ಯದಿ (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ಧ್ರುವ size_t hash_pos = ggml_hash_ಸೇರಿಸು(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_ಎಣಿಕೆ;
        }
    }
    ಧ್ರುವ std::chrono::duration<ದ್ವಿಗುಣ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::graph_recompute(ಧ್ರುವ rpc_ಸಂದೇಶ_graph_recompute_req & request) {
    uint32_t device = request.device;
    ಯದಿ (device >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    auto it = stored_graphs[device].find(request.uid);
    ಯದಿ (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
ಶೂನ್ಯ rpc_server::sync_all_backends() {
    ಪ್ರತಿ (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
ತಾರ್ಕಿಕ rpc_server::comm_init(ಧ್ರುವ rpc_ಸಂದೇಶ_comm_init_req & request, rpc_ಸಂದೇಶ_comm_init_rsp & response) {
    response.ok = 0;
    ಯದಿ (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    comm_state & state = comm_states[request.device];
    ಯದಿ (state.peer != nullptr) {
        response.ok = 1;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    ಯದಿ (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        ಯದಿ (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        state.peer = srv->accept();
        ಯದಿ (state.peer == nullptr) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಯದಿ (!state.peer->recv_data(remote_caps, ಪರಿಮಾಣ(remote_caps))) {
            state.peer = nullptr;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        state.peer->get_caps(local_caps);
        ಯದಿ (!state.peer->send_data(local_caps, ಪರಿಮಾಣ(local_caps))) {
            state.peer = nullptr;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        state.peer->update_caps(remote_caps);
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ std::string host(request.host, strnlen(request.host, ಪರಿಮಾಣ(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            ಯದಿ (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        ಯದಿ (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        state.peer->get_caps(local_caps);
        ಯದಿ (!state.peer->send_data(local_caps, ಪರಿಮಾಣ(local_caps)) ||
            !state.peer->recv_data(remote_caps, ಪರಿಮಾಣ(remote_caps))) {
            state.peer = nullptr;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::comm_allreduce(ಧ್ರುವ rpc_ಸಂದೇಶ_comm_allreduce_req & request) {
    ಯದಿ (request.device >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    comm_state & state = comm_states[request.device];
    ಯದಿ (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, ಅಸತ್ಯ);
    ಸಂರಚನಾ ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ರಿಕ್ತ,
        /*.no_alloc   =*/ ಸತ್ಯ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    ಯದಿ (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ size_t  nbytes = ggml_nbytes(t_dst);
    ಧ್ರುವ int64_t ne     = ggml_nelements(t_dst);
    ಯದಿ (nbytes == 0) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ಧ್ರುವ ತಾರ್ಕಿಕ   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ಧ್ರುವ size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    ಧ್ರುವ size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    ಯದಿ (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    ವರ್ಣ * scratch_base = (ವರ್ಣ *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        ಪ್ರತ್ಯಾಗಮ t;
    };
    auto new_cpy_ಗಂಟು = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ಪ್ರತ್ಯಾಗಮ t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, ಅಸತ್ಯ);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    ಯದಿ (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ಗಂಟು(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } ಅನ್ಯಥಾ {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    ಯದಿ (state.rank == 0) {
        ಯದಿ (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    ಯದಿ (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_ಗಂಟು(t_wire_recv, t_peer);
    } ಅನ್ಯಥಾ {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    ಯದಿ (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } ಅನ್ಯಥಾ {
        compute_nodes(t_red, nullptr);
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::comm_free(ಧ್ರುವ rpc_ಸಂದೇಶ_comm_free_req & request) {
    ಯದಿ (request.device >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    comm_states[request.device] = comm_state();
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ rpc_server::get_device_memory(ಧ್ರುವ rpc_ಸಂದೇಶ_get_device_memory_req & request, rpc_ಸಂದೇಶ_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    ಯದಿ (dev_id >= backends.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    size_t ಮುಕ್ತ, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &ಮುಕ್ತ, &total);
    response.free_mem = ಮುಕ್ತ;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

rpc_server::~rpc_server() {
    ಪ್ರತಿ (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rpc_serve_client(ಧ್ರುವ std::vector<ggml_backend_t> & backends, ಧ್ರುವ ವರ್ಣ * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    ಯದಿ (!sock->recv_data(&cmd, 1)) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ಪ್ರತ್ಯಾಗಮ;
    }

    // Read input_size and validate protocol version
    uint64_t hello_ಒಳಹರಿವು_size;
    ಯದಿ (!sock->recv_data(&hello_ಒಳಹರಿವು_size, ಪರಿಮಾಣ(hello_ಒಳಹರಿವು_size))) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (hello_ಒಳಹರಿವು_size != ಪರಿಮಾಣ(rpc_ಸಂದೇಶ_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_ಒಳಹರಿವು_size, ಪರಿಮಾಣ(rpc_ಸಂದೇಶ_hello_req), RPC_PROTO_MAJOR_VERSION);
        ಪ್ರತ್ಯಾಗಮ;
    }

    rpc_ಸಂದೇಶ_hello_req req = {};
    ಯದಿ (!sock->recv_data(&req, ಪರಿಮಾಣ(req))) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    rpc_ಸಂದೇಶ_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    ಯದಿ (!send_ಸಂದೇಶ(sock, &rsp, ಪರಿಮಾಣ(rsp))) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    ಯಾವತ್ (ಸತ್ಯ) {
        ಯದಿ (!sock->recv_data(&cmd, 1)) {
            ಭಙ್ಗ;
        }
        ಯದಿ (cmd >= RPC_CMD_ಎಣಿಕೆ) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            ಭಙ್ಗ;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((ಗಣನಾ rpc_cmd)cmd));
        ವಿಕಲ್ಪ (cmd) {
            ಪಕ್ಷ RPC_CMD_HELLO: {
                // HELLO command is handled above
                ಪ್ರತ್ಯಾಗಮ;
            }
            ಪಕ್ಷ RPC_CMD_DEVICE_ಎಣಿಕೆ: {
                ಯದಿ (!recv_ಸಂದೇಶ(sock, nullptr, 0)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_device_ಎಣಿಕೆ_rsp response;
                response.device_ಎಣಿಕೆ = backends.size();
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_ALLOC_BUFFER: {
                rpc_ಸಂದೇಶ_alloc_buffer_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_alloc_buffer_rsp response;
                ಯದಿ (!server.alloc_buffer(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GET_ALLOC_SIZE: {
                rpc_ಸಂದೇಶ_get_alloc_size_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_get_alloc_size_rsp response;
                ಯದಿ (!server.get_alloc_size(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GET_ALIGNMENT: {
                rpc_ಸಂದೇಶ_get_alignment_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_get_alignment_rsp response;
                ಯದಿ (!server.get_alignment(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GET_MAX_SIZE: {
                rpc_ಸಂದೇಶ_get_max_size_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_get_max_size_rsp response;
                ಯದಿ (!server.get_max_size(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_BUFFER_GET_BASE: {
                rpc_ಸಂದೇಶ_buffer_get_base_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_buffer_get_base_rsp response;
                ಯದಿ (!server.buffer_get_base(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_FREE_BUFFER: {
                rpc_ಸಂದೇಶ_free_buffer_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.free_buffer(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_BUFFER_CLEAR: {
                rpc_ಸಂದೇಶ_buffer_clear_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.buffer_clear(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_MEMSET_TENSOR: {
                rpc_ಸಂದೇಶ_memset_tensor_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.memset_tensor(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> ಒಳಹರಿವು;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, ಒಳಹರಿವು)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.set_tensor(ಒಳಹರಿವು)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> ಒಳಹರಿವು;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, ಒಳಹರಿವು)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.set_tensor_2d(ಒಳಹರಿವು)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GET_TENSOR_2D: {
                rpc_ಸಂದೇಶ_get_tensor_2d_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                std::vector<uint8_t> response;
                ಯದಿ (!server.get_tensor_2d(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, response.data(), response.size())) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_SET_TENSOR_HASH: {
                rpc_ಸಂದೇಶ_set_tensor_hash_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_set_tensor_hash_rsp response;
                ಯದಿ (!server.set_tensor_hash(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_INIT_TENSOR: {
                rpc_ಸಂದೇಶ_init_tensor_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request,ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.init_tensor(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GET_TENSOR: {
                rpc_ಸಂದೇಶ_get_tensor_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                std::vector<uint8_t> response;
                ಯದಿ (!server.get_tensor(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, response.data(), response.size())) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_COPY_TENSOR: {
                rpc_ಸಂದೇಶ_copy_tensor_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_copy_tensor_rsp response;
                ಯದಿ (!server.copy_tensor(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> ಒಳಹರಿವು;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, ಒಳಹರಿವು)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.graph_compute(ಒಳಹರಿವು)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_ಸಂದೇಶ_graph_recompute_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.graph_recompute(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_COMM_INIT: {
                rpc_ಸಂದೇಶ_comm_init_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_comm_init_rsp response;
                ಯದಿ (!server.comm_init(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_COMM_ALLREDUCE: {
                rpc_ಸಂದೇಶ_comm_allreduce_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.comm_allreduce(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_COMM_FREE: {
                rpc_ಸಂದೇಶ_comm_free_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!server.comm_free(request)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_ಸಂದೇಶ_get_device_memory_req request;
                ಯದಿ (!recv_ಸಂದೇಶ(sock, &request, ಪರಿಮಾಣ(request))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                rpc_ಸಂದೇಶ_get_device_memory_rsp response;
                ಯದಿ (!server.get_device_memory(request, response)) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಯದಿ (!send_ಸಂದೇಶ(sock, &response, ಪರಿಮಾಣ(response))) {
                    ಪ್ರತ್ಯಾಗಮ;
                }
                ಭಙ್ಗ;
            }
            ಸ್ವಭಾವ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((ಗಣನಾ rpc_cmd)cmd));
                ಪ್ರತ್ಯಾಗಮ;
            }
        }
    }
}

ಶೂನ್ಯ ggml_backend_rpc_start_server(ಧ್ರುವ ವರ್ಣ * endpoint, ಧ್ರುವ ವರ್ಣ * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    ಯದಿ (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ಪ್ರತ್ಯಾಗಮ;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    ಪ್ರತಿ (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t ಮುಕ್ತ, total;
        ggml_backend_dev_memory(dev, &ಮುಕ್ತ, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, ಮುಕ್ತ / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        ಯದಿ (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ಪ್ರತ್ಯಾಗಮ;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        ಯದಿ (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            ಯದಿ (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    ಪೂರ್ಣಾಙ್ಕ port;
    ಯದಿ (!parse_endpoint(endpoint, host, port)) {
        ಪ್ರತ್ಯಾಗಮ;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#ಅನ್ಯಥಾ
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    ಯದಿ (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ಪ್ರತ್ಯಾಗಮ;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    ಯದಿ (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯಾವತ್ (ಸತ್ಯ) {
        auto client_socket = server_socket->accept();
        ಯದಿ (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            ಪ್ರತ್ಯಾಗಮ;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    ಪ್ರತಿ (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ctx->name.c_str();
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, ಮುಕ್ತ, total);
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ಸಂರಚನಾ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ಸತ್ಯ,
        /* .host_buffer           = */ ಅಸತ್ಯ,
        /* .buffer_from_host_ptr  = */ ಅಸತ್ಯ,
        /* .events                = */ ಸತ್ಯ,
        /* .mmap_support          = */ ಸತ್ಯ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಪ್ರತ್ಯಾಗಮ ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ಯದಿ (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ಪ್ರತ್ಯಾಗಮ buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    ಪ್ರತ್ಯಾಗಮ dispatcher->event_new(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ ರಿಕ್ತ,
    /* .buffer_from_host_ptr = */ ರಿಕ್ತ,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ರಿಕ್ತ,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ಸಂರಚನಾ ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ಪ್ರತ್ಯಾಗಮ ctx ? ctx->name.c_str() : "RPC";
}

ಸ್ಥಿರ size_t ggml_backend_rpc_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ಪ್ರತ್ಯಾಗಮ ctx ? ctx->devices.size() : 0;
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ಯದಿ (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(index < ctx->devices.size());
        ಪ್ರತ್ಯಾಗಮ ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ಸಂರಚನಾ ggml_backend_rpc_comm_context {
    ಸಂರಚನಾ rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_comm_free(ಶೂನ್ಯ * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ಯದಿ (comm_ctx == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಪ್ರತಿ (ಧ್ರುವ auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_ಸಂದೇಶ_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, ಪರಿಮಾಣ(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ಯದಿ (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        ಯದಿ (!ggml_backend_is_rpc(backends[i])) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        ಪ್ರತಿ (ಧ್ರುವ auto & rank : ranks) {
            ಯದಿ (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    ಪೂರ್ಣಾಙ್ಕ port0;
    ಯದಿ (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಧ್ರುವ uint32_t comm_port = (uint32_t) port0 + 1000;
    ಯದಿ (host0.size() >= 64) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತಿ (ಧ್ರುವ auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_ಸಂದೇಶ_comm_init_rsp> responses(n_backends);
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_ಸಂದೇಶ_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        ಯದಿ (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, ಪರಿಮಾಣ(*request), &responses[i], ಪರಿಮಾಣ(responses[i]));
    }
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    ತಾರ್ಕಿಕ ok = ಸತ್ಯ;
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        ಯದಿ (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = ಅಸತ್ಯ;
        }
    }
    ಯದಿ (!ok) {
        ಪ್ರತಿ (ಧ್ರುವ auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ಪ್ರತ್ಯಾಗಮ ನೂತನ ggml_backend_rpc_comm_context{std::move(ranks)};
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_comm_allreduce_tensor(ಶೂನ್ಯ * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ಯದಿ (comm_ctx == nullptr) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ size_t n_ranks = comm_ctx->ranks.size();
    ಧ್ರುವ int64_t ne = ggml_nelements(tensors[0]);
    ಯದಿ (ne == 0) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತಿ (size_t i = 0; i < n_ranks; i++) {
        ಯದಿ (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        ಯದಿ ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಪ್ರತಿ (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_ಸಂದೇಶ_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, ಪರಿಮಾಣ(*request));
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    ಯದಿ (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_rpc_add_server;
    }
    ಯದಿ (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_rpc_start_server;
    }
    ಯದಿ (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_rpc_comm_init;
    }
    ಯದಿ (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_rpc_comm_free;
    }
    ಯದಿ (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;

    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ರಿಕ್ತ,
    };

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_rpc_reg;
}

ಸ್ಥಿರ uint32_t ggml_backend_rpc_get_device_ಎಣಿಕೆ(ಧ್ರುವ ವರ್ಣ * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_ಸಂದೇಶ_device_ಎಣಿಕೆ_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_ಎಣಿಕೆ, nullptr, 0, &response, ಪರಿಮಾಣ(response));
    ಪ್ರತ್ಯಾಗಮ response.device_ಎಣಿಕೆ;
}

ಸ್ಥಿರ ಧ್ರುವ ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_ಎಣಿಕೆ,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(ಧ್ರುವ ವರ್ಣ * endpoint) {
    ಸ್ಥಿರ std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ಸ್ಥಿರ std::mutex mutex;
    ಸ್ಥಿರ uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    ಯದಿ (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ಪ್ರತ್ಯಾಗಮ reg_map[endpoint];
    }
    uint32_t dev_ಎಣಿಕೆ = ggml_backend_rpc_get_device_ಎಣಿಕೆ(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_ಎಣಿಕೆ);
    ಯದಿ (dev_ಎಣಿಕೆ == 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = ನೂತನ ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    ಪ್ರತಿ (uint32_t ind = 0; ind < dev_ಎಣಿಕೆ; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = ನೂತನ ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = ನೂತನ ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = ನೂತನ ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ಪ್ರತ್ಯಾಗಮ reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
