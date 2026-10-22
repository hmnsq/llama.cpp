#அந்தர்பாவ "ggml-rpc.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-cpp.h"
#அந்தர்பாவ "transport.h"
#அந்தர்பாவ "log.h"

#அந்தர்பாவ <array>
#அந்தர்பாவ <chrono>
#அந்தர்பாவ <cinttypes>
#அந்தர்பாவ <optional>
#அந்தர்பாவ <string>
#அந்தர்பாவ <thread>
#அந்தர்பாவ <vector>
#அந்தர்பாவ <queue>
#அந்தர்பாவ <condition_variable>
#அந்தர்பாவ <future>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <filesystem>
#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <atomic>

நாமாகாஶ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) எனில் (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(தள்ளு, 1)
// ggml_tensor is serialized into rpc_tensor
கட்டமைப்பு rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / அளவு(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    எழுத்து name[GGML_MAX_NAME];

    int32_t use_மொத்தம்;
};

static_assert(அளவு(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
எண்ணிக்கை rpc_cmd {
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
    RPC_CMD_DEVICE_மொத்தம்,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_மொத்தம்,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

நிலையான மாறிலி எழுத்து * rpc_cmd_name(எண்ணிக்கை rpc_cmd cmd) {
    தேர்வு (cmd) {
        நிலை RPC_CMD_ALLOC_BUFFER:      திரும்பு "ALLOC_BUFFER";
        நிலை RPC_CMD_GET_ALIGNMENT:     திரும்பு "GET_ALIGNMENT";
        நிலை RPC_CMD_GET_MAX_SIZE:      திரும்பு "GET_MAX_SIZE";
        நிலை RPC_CMD_BUFFER_GET_BASE:   திரும்பு "BUFFER_GET_BASE";
        நிலை RPC_CMD_FREE_BUFFER:       திரும்பு "FREE_BUFFER";
        நிலை RPC_CMD_BUFFER_CLEAR:      திரும்பு "BUFFER_CLEAR";
        நிலை RPC_CMD_SET_TENSOR:        திரும்பு "SET_TENSOR";
        நிலை RPC_CMD_SET_TENSOR_HASH:   திரும்பு "SET_TENSOR_HASH";
        நிலை RPC_CMD_GET_TENSOR:        திரும்பு "GET_TENSOR";
        நிலை RPC_CMD_COPY_TENSOR:       திரும்பு "COPY_TENSOR";
        நிலை RPC_CMD_GRAPH_COMPUTE:     திரும்பு "GRAPH_COMPUTE";
        நிலை RPC_CMD_GET_DEVICE_MEMORY: திரும்பு "GET_DEVICE_MEMORY";
        நிலை RPC_CMD_INIT_TENSOR:       திரும்பு "INIT_TENSOR";
        நிலை RPC_CMD_GET_ALLOC_SIZE:    திரும்பு "GET_ALLOC_SIZE";
        நிலை RPC_CMD_HELLO:             திரும்பு "HELLO";
        நிலை RPC_CMD_DEVICE_மொத்தம்:      திரும்பு "DEVICE_COUNT";
        நிலை RPC_CMD_GRAPH_RECOMPUTE:   திரும்பு "GRAPH_RECOMPUTE";
        நிலை RPC_CMD_MEMSET_TENSOR:     திரும்பு "MEMSET_TENSOR";
        நிலை RPC_CMD_SET_TENSOR_2D:     திரும்பு "SET_TENSOR_2D";
        நிலை RPC_CMD_GET_TENSOR_2D:     திரும்பு "GET_TENSOR_2D";
        நிலை RPC_CMD_COMM_INIT:         திரும்பு "COMM_INIT";
        நிலை RPC_CMD_COMM_ALLREDUCE:    திரும்பு "COMM_ALLREDUCE";
        நிலை RPC_CMD_COMM_FREE:         திரும்பு "COMM_FREE";
        நிலை RPC_CMD_NONE:              திரும்பு "NONE";
        இயல்பு:                        திரும்பு "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
மாறிலி size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
மாறிலி size_t GRAPH_CACHE_MAX = 1024;

கட்டமைப்பு rpc_செய்தி_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

கட்டமைப்பு rpc_செய்தி_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

கட்டமைப்பு rpc_செய்தி_device_மொத்தம்_rsp {
    uint32_t device_மொத்தம்;
};

கட்டமைப்பு rpc_செய்தி_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

கட்டமைப்பு rpc_செய்தி_get_alloc_size_rsp {
    uint64_t alloc_size;
};

கட்டமைப்பு rpc_செய்தி_init_tensor_req {
    rpc_tensor tensor;
};

கட்டமைப்பு rpc_செய்தி_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

கட்டமைப்பு rpc_செய்தி_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

கட்டமைப்பு rpc_செய்தி_get_alignment_req {
    uint32_t device;
};

கட்டமைப்பு rpc_செய்தி_get_alignment_rsp {
    uint64_t alignment;
};

கட்டமைப்பு rpc_செய்தி_get_max_size_req {
    uint32_t device;
};

கட்டமைப்பு rpc_செய்தி_get_max_size_rsp {
    uint64_t max_size;
};

கட்டமைப்பு rpc_செய்தி_buffer_get_base_req {
    uint64_t remote_ptr;
};

கட்டமைப்பு rpc_செய்தி_buffer_get_base_rsp {
    uint64_t base_ptr;
};

கட்டமைப்பு rpc_செய்தி_free_buffer_req {
    uint64_t remote_ptr;
};

கட்டமைப்பு rpc_செய்தி_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

கட்டமைப்பு rpc_செய்தி_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

கட்டமைப்பு rpc_செய்தி_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

கட்டமைப்பு rpc_செய்தி_set_tensor_hash_rsp {
    uint8_t result;
};

கட்டமைப்பு rpc_செய்தி_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

கட்டமைப்பு rpc_செய்தி_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

கட்டமைப்பு rpc_செய்தி_copy_tensor_rsp {
    uint8_t result;
};

கட்டமைப்பு rpc_செய்தி_get_device_memory_req {
    uint32_t device;
};

கட்டமைப்பு rpc_செய்தி_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

கட்டமைப்பு rpc_செய்தி_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

கட்டமைப்பு rpc_செய்தி_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

கட்டமைப்பு rpc_செய்தி_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    எழுத்து     host[64];  // rank > 0: rank 0's host
};

கட்டமைப்பு rpc_செய்தி_comm_init_rsp {
    uint8_t ok;
};

கட்டமைப்பு rpc_செய்தி_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

கட்டமைப்பு rpc_செய்தி_comm_free_req {
    uint32_t device;
};

#pragma pack(எடு)

// RPC data structures

நிலையான ggml_guid_t ggml_backend_rpc_guid() {
    நிலையான ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    திரும்பு &guid;
}

கட்டமைப்பு ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

கட்டமைப்பு ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

வகுப்பு rpc_dispatcher;
கட்டமைப்பு ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

கட்டமைப்பு ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    வெறுமை                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
நிலையான uint64_t fnv_hash(மாறிலி uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    மாறிலி uint64_t fnv_பகாஎண் = 0x100000001b3ULL;

    க்கு (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_பகாஎண்;
    }
    திரும்பு hash;
}

நிலையான தருக்க send_செய்தி(socket_ptr sock, மாறிலி வெறுமை * செய்தி, size_t செய்தி_size) {
    எனில் (!sock->send_data(&செய்தி_size, அளவு(செய்தி_size))) {
        திரும்பு பொய்;
    }
    எனில் (!sock->send_data(செய்தி, செய்தி_size)) {
        திரும்பு பொய்;
    }
    திரும்பு sock->flush();
}

நிலையான தருக்க recv_செய்தி(socket_ptr sock, வெறுமை * செய்தி, size_t செய்தி_size) {
    uint64_t size;
    எனில் (!sock->recv_data(&size, அளவு(size))) {
        திரும்பு பொய்;
    }
    எனில் (size != செய்தி_size) {
        திரும்பு பொய்;
    }
    திரும்பு sock->recv_data(செய்தி, செய்தி_size);
}

நிலையான தருக்க recv_செய்தி(socket_ptr sock, std::vector<uint8_t> & உள்ளீடு) {
    uint64_t size;
    எனில் (!sock->recv_data(&size, அளவு(size))) {
        திரும்பு பொய்;
    }
    முயல் {
        உள்ளீடு.resize(size);
    } பிடி (மாறிலி std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        திரும்பு பொய்;
    }
    திரும்பு sock->recv_data(உள்ளீடு.data(), size);
}

நிலையான தருக்க parse_endpoint(மாறிலி std::string & endpoint, std::string & host, முழு & port) {
    size_t pos = endpoint.find(':');
    எனில் (pos == std::string::npos) {
        திரும்பு பொய்;
    }
    host = endpoint.substr(0, pos);
    முயல் {
        port = std::stoi(endpoint.substr(pos + 1));
    } பிடி (...) {
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
நிலையான தருக்க send_rpc_cmd(socket_ptr sock, எண்ணிக்கை rpc_cmd cmd, மாறிலி வெறுமை * உள்ளீடு, size_t உள்ளீடு_size) {
    uint8_t cmd_byte = cmd;
    எனில் (!sock->send_data(&cmd_byte, அளவு(cmd_byte))) {
        திரும்பு பொய்;
    }
    எனில் (!sock->send_data(&உள்ளீடு_size, அளவு(உள்ளீடு_size))) {
        திரும்பு பொய்;
    }
    எனில் (!sock->send_data(உள்ளீடு, உள்ளீடு_size)) {
        திரும்பு பொய்;
    }
    திரும்பு sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
நிலையான தருக்க send_rpc_cmd(socket_ptr sock, எண்ணிக்கை rpc_cmd cmd, மாறிலி வெறுமை * உள்ளீடு, size_t உள்ளீடு_size, வெறுமை * output, size_t output_size) {
    எனில் (!send_rpc_cmd(sock, cmd, உள்ளீடு, உள்ளீடு_size)) {
        திரும்பு பொய்;
    }
    uint64_t out_size;
    எனில் (!sock->recv_data(&out_size, அளவு(out_size))) {
        திரும்பு பொய்;
    }
    எனில் (out_size != output_size) {
        திரும்பு பொய்;
    }
    எனில் (!sock->recv_data(output, output_size)) {
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
நிலையான constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

நிலையான உள்வரி வெறுமை rpc_cpu_relax() {
#எனில் defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ நிலையற்ற("yield" ::: "memory");
#இல்லையெனில்
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
நிலையான தருக்க negotiate_hello(மாறிலி std::shared_ptr<socket_t> & sock) {
    rpc_செய்தி_hello_req request = {};
    rpc_செய்தி_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    தருக்க status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, அளவு(request), &response, அளவு(response));
    RPC_STATUS_ASSERT(status);

    எனில் (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        திரும்பு பொய்;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    திரும்பு உண்மை;
}

template <typename T>
வகுப்பு message_queue {
பொது:
    message_queue() {}

    தருக்க தள்ளு(மாறிலி T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        எனில் (interrupted) {
            திரும்பு பொய்;
        }
        queue.தள்ளு(value);
        cvar.notify_all();
        திரும்பு உண்மை;
    }

    தருக்க எடு(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [இது] { திரும்பு !queue.empty() || interrupted; });
        எனில் (interrupted) {
            திரும்பு பொய்;
        }
        *out = queue.front();
        queue.எடு();
        திரும்பு உண்மை;
    }

    தருக்க try_எடு(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        எனில் (interrupted || queue.empty()) {
            திரும்பு பொய்;
        }
        *out = queue.front();
        queue.எடு();
        திரும்பு உண்மை;
    }

    வெறுமை interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = உண்மை;
        lock.unlock();
        cvar.notify_all();
    }

தனிப்பட்ட:
    தருக்க interrupted = பொய்;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

வகுப்பு rpc_dispatcher {
பொது:
    rpc_dispatcher() {
    }

    வெறுமை send(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size);
    வெறுமை send(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size, வெறுமை * output, size_t output_size);
    வெறுமை send_async(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size);
    வெறுமை send_async(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size, வெறுமை * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    வெறுமை event_free(ggml_backend_event_t event);
    வெறுமை event_synchronize(ggml_backend_event_t event);
    வெறுமை event_record(ggml_backend_event_t event);
    வெறுமை synchronize();
    வெறுமை busy_spin_acquire();
    வெறுமை busy_spin_release();
    வெறுமை graph_compute(uint32_t device, மாறிலி ggml_cgraph * cgraph);

    வெறுமை start(மாறிலி std::string & endpoint);
    வெறுமை work();

    ~rpc_dispatcher();

தனிப்பட்ட:
    கட்டமைப்பு rpc_செய்தி {
        rpc_cmd                       cmd;
        std::shared_ptr<மாறிலி வெறுமை>   உள்ளீடு;
        size_t                        உள்ளீடு_size;
        வெறுமை                        * output;
        size_t                        output_size;
        std::promise<வெறுமை>            completion;
    };
    using rpc_செய்தி_ptr   = std::shared_ptr<rpc_செய்தி>;
    using rpc_செய்தி_queue = message_queue<rpc_செய்தி_ptr>;
    கட்டமைப்பு rpc_event {
        rpc_செய்தி_ptr              செய்தி;
        std::shared_future<வெறுமை> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_செய்தி_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

நிலையான வெறுமை rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

வெறுமை rpc_dispatcher::send(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size) {
    auto செய்தி = std::make_shared<rpc_செய்தி>();
    செய்தி->cmd = cmd;
    செய்தி->உள்ளீடு = உள்ளீடு;
    செய்தி->உள்ளீடு_size = உள்ளீடு_size;
    செய்தி->output = nullptr;
    செய்தி->output_size = 0;
    GGML_ASSERT(queue.தள்ளு(செய்தி));
    auto future = செய்தி->completion.get_future();
    future.wait();
}

வெறுமை rpc_dispatcher::send_async(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size) {
    auto செய்தி = std::make_shared<rpc_செய்தி>();
    செய்தி->cmd = cmd;
    செய்தி->உள்ளீடு = உள்ளீடு;
    செய்தி->உள்ளீடு_size = உள்ளீடு_size;
    செய்தி->output = nullptr;
    செய்தி->output_size = 0;
    GGML_ASSERT(queue.தள்ளு(செய்தி));
}

வெறுமை rpc_dispatcher::send(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size, வெறுமை * output, size_t output_size) {
    auto செய்தி = std::make_shared<rpc_செய்தி>();
    செய்தி->cmd = cmd;
    செய்தி->உள்ளீடு = உள்ளீடு;
    செய்தி->உள்ளீடு_size = உள்ளீடு_size;
    செய்தி->output = output;
    செய்தி->output_size = output_size;
    GGML_ASSERT(queue.தள்ளு(செய்தி));
    auto future = செய்தி->completion.get_future();
    future.wait();
}

வெறுமை rpc_dispatcher::send_async(எண்ணிக்கை rpc_cmd cmd, std::shared_ptr<மாறிலி வெறுமை> உள்ளீடு, size_t உள்ளீடு_size, வெறுமை * output, size_t output_size) {
    auto செய்தி = std::make_shared<rpc_செய்தி>();
    செய்தி->cmd = cmd;
    செய்தி->உள்ளீடு = உள்ளீடு;
    செய்தி->உள்ளீடு_size = உள்ளீடு_size;
    செய்தி->output = output;
    செய்தி->output_size = output_size;
    GGML_ASSERT(queue.தள்ளு(செய்தி));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = புதிய rpc_event;
    ev->செய்தி = std::make_shared<rpc_செய்தி>();
    ev->செய்தி->cmd = RPC_CMD_NONE;
    ev->sf = ev->செய்தி->completion.get_future().share();
    GGML_ASSERT(queue.தள்ளு(ev->செய்தி));
    திரும்பு புதிய ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

வெறுமை rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

வெறுமை rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

வெறுமை rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->செய்தி = std::make_shared<rpc_செய்தி>();
    ev->செய்தி->cmd = RPC_CMD_NONE;
    ev->sf = ev->செய்தி->completion.get_future().share();
    GGML_ASSERT(queue.தள்ளு(ev->செய்தி));
}

வெறுமை rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto செய்தி = std::make_shared<rpc_செய்தி>();
    செய்தி->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.தள்ளு(செய்தி));
    செய்தி->completion.get_future().wait();
}

வெறுமை rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

வெறுமை rpc_dispatcher::busy_spin_release() {
    மாறிலி unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

வெறுமை rpc_dispatcher::start(மாறிலி std::string & endpoint) {
    std::string host;
    முழு port;
    எனில் (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    எனில் (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    எனில் (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    எனில் (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = உண்மை;
    thread = std::thread(rpc_dispatcher_trampoline, இது);
}

வெறுமை rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    போது (running) {
        rpc_செய்தி_ptr செய்தி_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        மாறிலி தருக்க spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        எனில் (spin) {
            எனில் (!queue.try_எடு(&செய்தி_ptr)) {
                rpc_cpu_relax();
                தொடர்;
            }
        } இல்லையெனில் எனில் (!queue.எடு(&செய்தி_ptr)) {
            நிறுத்து;
        }
        எனில் (செய்தி_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(செய்தி_ptr->cmd),
                     rpc_log_format_size(செய்தி_ptr->உள்ளீடு_size).c_str(), rpc_log_format_size(செய்தி_ptr->output_size).c_str());
            மாறிலி தருக்க trace = rpc_log_debug_level() >= 3;
            மாறிலி auto t0 = std::chrono::steady_clock::now();
            எனில் (செய்தி_ptr->output) {
                தருக்க status = send_rpc_cmd(sock, செய்தி_ptr->cmd, செய்தி_ptr->உள்ளீடு.get(), செய்தி_ptr->உள்ளீடு_size, செய்தி_ptr->output, செய்தி_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } இல்லையெனில் {
                தருக்க status = send_rpc_cmd(sock, செய்தி_ptr->cmd, செய்தி_ptr->உள்ளீடு.get(), செய்தி_ptr->உள்ளீடு_size);
                RPC_STATUS_ASSERT(status);
            }
            எனில் (trace) {
                மாறிலி std::chrono::duration<இரட்டை, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(செய்தி_ptr->cmd), elapsed.count());
            }
        } இல்லையெனில் {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        செய்தி_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = பொய்;
    queue.interrupt();
    sock = nullptr;
    எனில் (thread.joinable()) {
        thread.join();
    }
}

நிலையான std::shared_ptr<rpc_dispatcher> get_dispatcher(மாறிலி std::string & endpoint) {
    நிலையான std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    நிலையான std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    எனில் (it != dispatchers.end()) {
        எனில் (auto dispatcher = it->second.lock()) {
            திரும்பு dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    திரும்பு dispatcher;
}

நிலையான வெறுமை ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_செய்தி_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, அளவு(*request));
    delete ctx;
}

நிலையான வெறுமை * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    எனில் (ctx->base_ptr != nullptr) {
        திரும்பு ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_செய்தி_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_செய்தி_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, அளவு(*request), &response, அளவு(response));
    ctx->base_ptr = reinterpret_cast<வெறுமை *>(response.base_ptr);
    திரும்பு ctx->base_ptr;
}

நிலையான தருக்க ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    திரும்பு buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

நிலையான rpc_tensor serialize_tensor(மாறிலி ggml_tensor * tensor, மாறிலி rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    எனில் (!tensor) {
        memset(&result, 0, அளவு(result));
        திரும்பு result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    எனில் (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        எனில் (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } இல்லையெனில் {
            result.buffer = 0;
            result.data = 0;
        }
    } இல்லையெனில் {
        result.buffer = 0;
        result.data   = 0;
    }
    க்கு (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    க்கு (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / அளவு(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    க்கு (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, அளவு(result.name));
    result.use_மொத்தம் = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    திரும்பு result;
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    எனில் (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_செய்தி_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, அளவு(*request));
    }
    திரும்பு GGML_STATUS_SUCCESS;
}

நிலையான வெறுமை ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_செய்தி_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, அளவு(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
நிலையான std::shared_ptr<uint8_t> serialize_set_tensor(மாறிலி rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, மாறிலி வெறுமை * data, size_t size, size_t & உள்ளீடு_size) {
    உள்ளீடு_size = அளவு(rpc_tensor) + அளவு(cache_flag) + அளவு(offset) + size;
    uint8_t * உள்ளீடு = புதிய uint8_t[உள்ளீடு_size]();
    uint8_t * p = உள்ளீடு;
    memcpy(p, &rpc_tensor, அளவு(rpc_tensor)); p += அளவு(rpc_tensor);
    memcpy(p, &cache_flag, அளவு(cache_flag)); p += அளவு(cache_flag);
    memcpy(p, &offset,     அளவு(offset));     p += அளவு(offset);
    memcpy(p, data, size);
    திரும்பு std::shared_ptr<uint8_t>(உள்ளீடு, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
நிலையான தருக்க rpc_use_hash_cache(மாறிலி ggml_tensor * tensor, size_t size) {
    திரும்பு size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

நிலையான வெறுமை ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    எனில் (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_செய்தி_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((மாறிலி uint8_t*)data, size);
        rpc_செய்தி_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, அளவு(*request), &response, அளவு(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        எனில் (response.result) {
            // the server has the same data, no need to send it
            திரும்பு;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t உள்ளீடு_size;
    auto உள்ளீடு = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, உள்ளீடு_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, உள்ளீடு, உள்ளீடு_size);
}

நிலையான வெறுமை ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, மாறிலி வெறுமை * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t உள்ளீடு_size = அளவு(rpc_tensor) + 4*அளவு(uint64_t) + size*n_copies;
    uint8_t * உள்ளீடு = புதிய uint8_t[உள்ளீடு_size]();
    uint8_t * dest = உள்ளீடு;
    memcpy(dest, &rpc_tensor, அளவு(rpc_tensor));
    dest += அளவு(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, அளவு(header));
    dest += அளவு(header);
    க்கு (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (மாறிலி எழுத்து *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> உள்ளீடு_ptr(உள்ளீடு, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, உள்ளீடு_ptr, உள்ளீடு_size);
}

நிலையான வெறுமை ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * tensor, வெறுமை * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_செய்தி_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    எனில் (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, அளவு(*request), data, size*n_copies);
    } இல்லையெனில் {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, அளவு(*request), packed.data(), packed.size());
        க்கு (size_t i = 0; i < n_copies; i++) {
            memcpy((எழுத்து *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

நிலையான வெறுமை ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_செய்தி_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, அளவு(*request), data, size);
}

நிலையான தருக்க ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    எனில் (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        எனில் (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            திரும்பு பொய்;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_செய்தி_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_செய்தி_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, அளவு(*request), &response, அளவு(response));
        திரும்பு response.result;
    }
    திரும்பு பொய்;
}

நிலையான வெறுமை ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_செய்தி_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, அளவு(*request));
}

நிலையான ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ வெற்று,
};

நிலையான மாறிலி எழுத்து * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    திரும்பு buft_ctx->name.c_str();
}

நிலையான ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_செய்தி_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_செய்தி_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, அளவு(*request), &response, அளவு(response));
    எனில் (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            புதிய ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        திரும்பு buffer;
    } இல்லையெனில் {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        திரும்பு nullptr;
    }
}

நிலையான size_t get_alignment(மாறிலி std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_செய்தி_get_alignment_req>();
    request->device = device;
    rpc_செய்தி_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, அளவு(*request), &response, அளவு(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    திரும்பு response.alignment;
}

நிலையான size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    திரும்பு buft_ctx->alignment;
}

நிலையான size_t get_max_size(மாறிலி std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_செய்தி_get_max_size_req>();
    request->device = device;
    rpc_செய்தி_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, அளவு(*request), &response, அளவு(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    திரும்பு response.max_size;
}

நிலையான size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    திரும்பு buft_ctx->max_size;
}

நிலையான size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    தருக்க rpc_get = பொய்;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    எனில் (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        மாறிலி size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        கட்டமைப்பு alloc_size_cache_சாவி {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / அளவு(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_சாவி சாவி = {};
        சாவி.device = buft_ctx->device;
        சாவி.type = tensor->type;
        சாவி.op = tensor->op;
        memcpy(சாவி.op_params, tensor->op_params, அளவு(சாவி.op_params));
        க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
            சாவி.ne[i] = (uint32_t)tensor->ne[i];
            சாவி.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((மாறிலி uint8_t *)&சாவி, அளவு(சாவி));
        cache_hash = fnv_hash((மாறிலி uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        நிலையான std::mutex cache_mutex;
        நிலையான std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            எனில் (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                திரும்பு std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_செய்தி_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_செய்தி_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, அளவு(*request), &response, அளவு(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        திரும்பு std::max<size_t>(response.alloc_size, min_size);
    }

    திரும்பு ggml_nbytes(tensor);
}

நிலையான ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ வெற்று,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ வெற்று,
    /* .is_host             = */ வெற்று,
};

நிலையான மாறிலி எழுத்து * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    திரும்பு rpc_ctx->name.c_str();
}

நிலையான வெறுமை ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

நிலையான வெறுமை ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    எனில் (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_செய்தி_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((மாறிலி uint8_t*)data, size);
        rpc_செய்தி_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, அளவு(*request), &response, அளவு(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        எனில் (response.result) {
            // the server has the same data, no need to send it
            திரும்பு;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t உள்ளீடு_size;
    auto உள்ளீடு = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, உள்ளீடு_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, உள்ளீடு, உள்ளீடு_size);
}

நிலையான வெறுமை ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, மாறிலி ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_செய்தி_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, அளவு(*request), data, size);
}

நிலையான வெறுமை ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

நிலையான வெறுமை add_tensor(ggml_tensor * tensor, மாறிலி ggml_cgraph * cgraph, மாறிலி rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    எனில் (tensor == nullptr) {
        திரும்பு;
    }
    எனில் (visited.find(tensor) != visited.end()) {
        திரும்பு;
    }
    visited.insert(tensor);
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    மாறிலி size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    எனில் (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_மொத்தம் = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

நிலையான uint8_t * serialize_graph(uint32_t device, மாறிலி ggml_cgraph * cgraph, மாறிலி rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    க்கு (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*அளவு(uint32_t) + அளவு(uint64_t) + n_nodes * அளவு(uint64_t) + அளவு(uint32_t) + n_tensors * அளவு(rpc_tensor);
    uint8_t * output = புதிய uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, அளவு(device));
    dest += அளவு(device);
    memcpy(dest, &cgraph->uid, அளவு(cgraph->uid));
    dest += அளவு(cgraph->uid);
    memcpy(dest, &n_nodes, அளவு(n_nodes));
    dest += அளவு(n_nodes);
    க்கு (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * அளவு(uint64_t), &cgraph->nodes[i], அளவு(uint64_t));
    }
    dest += n_nodes * அளவு(uint64_t);
    memcpy(dest, &n_tensors, அளவு(n_tensors));
    dest += அளவு(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * அளவு(rpc_tensor));
    திரும்பு output;
}

வெறுமை rpc_dispatcher::graph_compute(uint32_t device, மாறிலி ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    தருக்க reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    எனில் (reuse) {
        auto request = std::make_shared<rpc_செய்தி_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, அளவு(*request));
    } இல்லையெனில் {
        எனில் (cgraph->uid != 0) {
            எனில் (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t உள்ளீடு_size = 0;
        uint8_t * உள்ளீடு = serialize_graph(device, cgraph, இது, &உள்ளீடு_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(உள்ளீடு_size).c_str());
        std::shared_ptr<uint8_t> உள்ளீடு_ptr(உள்ளீடு, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, உள்ளீடு_ptr, உள்ளீடு_size);
    }
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    திரும்பு GGML_STATUS_SUCCESS;
}

நிலையான வெறுமை ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

நிலையான வெறுமை ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

நிலையான வெறுமை ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, மாறிலி வெறுமை * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

நிலையான வெறுமை ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, மாறிலி ggml_tensor * tensor, வெறுமை * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

நிலையான ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ வெற்று,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ வெற்று,
    /* .graph_plan_free         = */ வெற்று,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ வெற்று,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ வெற்று,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(மாறிலி எழுத்து * endpoint, uint32_t device) {
    நிலையான std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    நிலையான std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    எனில் (it != buft_map.end()) {
        திரும்பு it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = புதிய ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = புதிய ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    திரும்பு buft;
}

ggml_backend_t ggml_backend_rpc_init(மாறிலி எழுத்து * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = புதிய ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = புதிய ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    திரும்பு backend;
}

தருக்க ggml_backend_is_rpc(ggml_backend_t backend) {
    திரும்பு backend != வெற்று && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

வெறுமை ggml_backend_rpc_get_device_memory(மாறிலி எழுத்து * endpoint, uint32_t device, size_t * விடுவி, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_செய்தி_get_device_memory_req>();
    request->device = device;
    rpc_செய்தி_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, அளவு(*request), &response, அளவு(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *விடுவி = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

வகுப்பு rpc_server {
பொது:
    rpc_server(std::vector<ggml_backend_t> all_backends, மாறிலி எழுத்து * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    வெறுமை hello(rpc_செய்தி_hello_rsp & response);
    தருக்க alloc_buffer(மாறிலி rpc_செய்தி_alloc_buffer_req & request, rpc_செய்தி_alloc_buffer_rsp & response);
    தருக்க get_alignment(மாறிலி rpc_செய்தி_get_alignment_req & request, rpc_செய்தி_get_alignment_rsp & response);
    தருக்க get_max_size(மாறிலி rpc_செய்தி_get_max_size_req & request, rpc_செய்தி_get_max_size_rsp & response);
    தருக்க buffer_get_base(மாறிலி rpc_செய்தி_buffer_get_base_req & request, rpc_செய்தி_buffer_get_base_rsp & response);
    தருக்க free_buffer(மாறிலி rpc_செய்தி_free_buffer_req & request);
    தருக்க buffer_clear(மாறிலி rpc_செய்தி_buffer_clear_req & request);
    தருக்க memset_tensor(மாறிலி rpc_செய்தி_memset_tensor_req & request);
    தருக்க set_tensor(மாறிலி std::vector<uint8_t> & உள்ளீடு);
    தருக்க set_tensor_2d(மாறிலி std::vector<uint8_t> & உள்ளீடு);
    தருக்க set_tensor_hash(மாறிலி rpc_செய்தி_set_tensor_hash_req & request, rpc_செய்தி_set_tensor_hash_rsp & response);
    தருக்க get_tensor(மாறிலி rpc_செய்தி_get_tensor_req & request, std::vector<uint8_t> & response);
    தருக்க get_tensor_2d(மாறிலி rpc_செய்தி_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    தருக்க copy_tensor(மாறிலி rpc_செய்தி_copy_tensor_req & request, rpc_செய்தி_copy_tensor_rsp & response);
    தருக்க graph_compute(மாறிலி std::vector<uint8_t> & உள்ளீடு);
    தருக்க graph_recompute(மாறிலி rpc_செய்தி_graph_recompute_req & request);
    தருக்க comm_init(மாறிலி rpc_செய்தி_comm_init_req & request, rpc_செய்தி_comm_init_rsp & response);
    தருக்க comm_allreduce(மாறிலி rpc_செய்தி_comm_allreduce_req & request);
    தருக்க comm_free(மாறிலி rpc_செய்தி_comm_free_req & request);
    தருக்க init_tensor(மாறிலி rpc_செய்தி_init_tensor_req & request);
    தருக்க get_alloc_size(மாறிலி rpc_செய்தி_get_alloc_size_req & request, rpc_செய்தி_get_alloc_size_rsp & response);
    தருக்க get_device_memory(மாறிலி rpc_செய்தி_get_device_memory_req & request, rpc_செய்தி_get_device_memory_rsp & response);

    கட்டமைப்பு stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

தனிப்பட்ட:
    வெறுமை sync_all_backends();
    தருக்க get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(கட்டமைப்பு ggml_context * ctx, மாறிலி rpc_tensor * tensor);
    ggml_tensor * create_முனை(uint64_t id,
                              கட்டமைப்பு ggml_context * ctx,
                              மாறிலி std::unordered_map<uint64_t, மாறிலி rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, கட்டமைப்பு ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    கட்டமைப்பு comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    மாறிலி எழுத்து * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

வெறுமை rpc_server::hello(rpc_செய்தி_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

தருக்க rpc_server::get_alloc_size(மாறிலி rpc_செய்தி_get_alloc_size_req & request, rpc_செய்தி_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    எனில் (dev_id >= backends.size()) {
        திரும்பு பொய்;
    }
    ggml_backend_buffer_type_t buft;
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    எனில் (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        திரும்பு பொய்;
    }
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        எனில் (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (வெறுமை*)tensor->buffer, tensor->data);
    எனில் (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } இல்லையெனில் {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    திரும்பு உண்மை;
}

தருக்க rpc_server::alloc_buffer(மாறிலி rpc_செய்தி_alloc_buffer_req & request, rpc_செய்தி_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    எனில் (dev_id >= backends.size()) {
        திரும்பு பொய்;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    எனில் (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } இல்லையெனில் {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    திரும்பு உண்மை;
}

தருக்க rpc_server::get_alignment(மாறிலி rpc_செய்தி_get_alignment_req & request, rpc_செய்தி_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    எனில் (dev_id >= backends.size()) {
        திரும்பு பொய்;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    திரும்பு உண்மை;
}

தருக்க rpc_server::get_max_size(மாறிலி rpc_செய்தி_get_max_size_req & request, rpc_செய்தி_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    எனில் (dev_id >= backends.size()) {
        திரும்பு பொய்;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    திரும்பு உண்மை;
}

தருக்க rpc_server::buffer_get_base(மாறிலி rpc_செய்தி_buffer_get_base_req & request, rpc_செய்தி_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    எனில் (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        திரும்பு பொய்;
    }
    வெறுமை * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    திரும்பு உண்மை;
}

தருக்க rpc_server::free_buffer(மாறிலி rpc_செய்தி_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    எனில் (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        திரும்பு பொய்;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    க்கு (auto & sgs : stored_graphs) {
        க்கு (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    திரும்பு உண்மை;
}

தருக்க rpc_server::buffer_clear(மாறிலி rpc_செய்தி_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    எனில் (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        திரும்பு பொய்;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    திரும்பு உண்மை;
}

தருக்க rpc_server::memset_tensor(மாறிலி rpc_செய்தி_memset_tensor_req & request) {
    sync_all_backends();
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    எனில் (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }

    மாறிலி uint64_t tensor_size = ggml_nbytes(tensor);
    எனில் (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        திரும்பு பொய்;
    }

    மாறிலி uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    மாறிலி uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    எனில் (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        திரும்பு பொய்;
    }
    மாறிலி uint64_t data_offset = request.tensor.data - buffer_start;
    எனில் (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        திரும்பு பொய்;
    }
    எனில் (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        திரும்பு பொய்;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (வெறுமை *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    திரும்பு உண்மை;
}

ggml_tensor * rpc_server::deserialize_tensor(கட்டமைப்பு ggml_context * ctx, மாறிலி rpc_tensor * tensor) {
    // Validate tensor type before using it
    எனில் (tensor->type >= GGML_TYPE_மொத்தம்) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        திரும்பு nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    எனில் (ggml_blck_size((எண்ணிக்கை ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        திரும்பு nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    எனில் (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        திரும்பு nullptr;
    }

    க்கு (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    எனில் (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    எனில் (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        எனில் (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            திரும்பு nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    க்கு (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / அளவு(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<வெறுமை *>(tensor->data);
    ggml_set_name(result, tensor->name);
    திரும்பு result;
}


தருக்க rpc_server::set_tensor(மாறிலி std::vector<uint8_t> & உள்ளீடு) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    மாறிலி size_t header_size = அளவு(rpc_tensor) + அளவு(cache_flag) + அளவு(offset);
    எனில் (உள்ளீடு.size() < header_size) {
        திரும்பு பொய்;
    }
    மாறிலி rpc_tensor * in_tensor = (மாறிலி rpc_tensor *)உள்ளீடு.data();
    memcpy(&cache_flag, உள்ளீடு.data() + அளவு(rpc_tensor), அளவு(cache_flag));
    memcpy(&offset,     உள்ளீடு.data() + அளவு(rpc_tensor) + அளவு(cache_flag), அளவு(offset));
    மாறிலி size_t size = உள்ளீடு.size() - header_size;

    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    எனில் (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (வெறுமை*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        மாறிலி size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        மாறிலி size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        எனில் (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            திரும்பு பொய்;
        }
    }

    மாறிலி வெறுமை * data = உள்ளீடு.data() + header_size;
    எனில் (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((மாறிலி uint8_t*)data, size);
        எழுத்து hash_str[17];
        snprintf(hash_str, அளவு(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.லிக((மாறிலி எழுத்து *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    திரும்பு உண்மை;
}

தருக்க rpc_server::set_tensor_2d(மாறிலி std::vector<uint8_t> & உள்ளீடு) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    எனில் (உள்ளீடு.size() < அளவு(rpc_tensor) + 4*அளவு(uint64_t)) {
        திரும்பு பொய்;
    }
    மாறிலி rpc_tensor * in_tensor = (மாறிலி rpc_tensor *)உள்ளீடு.data();
    uint64_t header[4];
    memcpy(header, உள்ளீடு.data() + அளவு(rpc_tensor), அளவு(header));
    மாறிலி uint64_t offset   = header[0];
    மாறிலி uint64_t size     = header[1];
    மாறிலி uint64_t n_copies = header[2];
    மாறிலி uint64_t stride   = header[3];

    மாறிலி uint64_t data_size = உள்ளீடு.size() - அளவு(rpc_tensor) - 4*அளவு(uint64_t);
    எனில் (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        திரும்பு பொய்;
    }

    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    எனில் (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (வெறுமை*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        எனில் (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            திரும்பு பொய்;
        }
        மாறிலி uint64_t span = (n_copies - 1)*stride + size;
        மாறிலி uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        மாறிலி uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        எனில் (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            திரும்பு பொய்;
        }
        எனில் (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            திரும்பு பொய்;
        }
    }

    மாறிலி வெறுமை * data = உள்ளீடு.data() + அளவு(rpc_tensor) + 4*அளவு(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    திரும்பு உண்மை;
}

தருக்க rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    எனில் (!cache_dir) {
        திரும்பு பொய்;
    }
    எழுத்து hash_str[17];
    snprintf(hash_str, அளவு(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    எனில் (!fs::exists(cache_file, ec)) {
        திரும்பு பொய்;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.பட((எழுத்து *)data.data(), size);
    திரும்பு உண்மை;
}

தருக்க rpc_server::set_tensor_hash(மாறிலி rpc_செய்தி_set_tensor_hash_req & request, rpc_செய்தி_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    எனில் (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }

    std::vector<uint8_t> cached_file;
    எனில் (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        திரும்பு உண்மை;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (வெறுமை*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        மாறிலி size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        மாறிலி size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        எனில் (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            திரும்பு பொய்;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    திரும்பு உண்மை;
}

தருக்க rpc_server::init_tensor(மாறிலி rpc_செய்தி_init_tensor_req & request) {
    sync_all_backends();
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    எனில் (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        திரும்பு பொய்;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (வெறுமை*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    எனில் (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } இல்லையெனில் {
        எனில் (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    எனில் (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

தருக்க rpc_server::get_tensor(மாறிலி rpc_செய்தி_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    எனில் (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (வெறுமை*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        மாறிலி size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        மாறிலி size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        எனில் (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                திரும்பு பொய்;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    திரும்பு உண்மை;
}

தருக்க rpc_server::get_tensor_2d(மாறிலி rpc_செய்தி_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    எனில் (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (வெறுமை*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        எனில் (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            திரும்பு பொய்;
        }
        எனில் (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            திரும்பு பொய்;
        }
        மாறிலி uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        மாறிலி uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        மாறிலி uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        எனில் (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            திரும்பு பொய்;
        }
        எனில் (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            திரும்பு பொய்;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    திரும்பு உண்மை;
}

தருக்க rpc_server::copy_tensor(மாறிலி rpc_செய்தி_copy_tensor_req & request, rpc_செய்தி_copy_tensor_rsp & response) {
    sync_all_backends();
    கட்டமைப்பு ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    எனில் (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        திரும்பு பொய்;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    எனில் (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        திரும்பு பொய்;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    திரும்பு உண்மை;
}

ggml_tensor * rpc_server::create_முனை(uint64_t id,
                                      கட்டமைப்பு ggml_context * ctx,
                                      மாறிலி std::unordered_map<uint64_t, மாறிலி rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, கட்டமைப்பு ggml_tensor*> & tensor_map) {
    எனில் (tensor_map.find(id) != tensor_map.end()) {
        திரும்பு tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    எனில் (it_ptr == tensor_ptrs.end()) {
        திரும்பு nullptr;
    }
    மாறிலி rpc_tensor * tensor = it_ptr->second;

    கட்டமைப்பு ggml_tensor * result = deserialize_tensor(ctx, tensor);
    எனில் (result == nullptr) {
        திரும்பு nullptr;
    }
    எனில் (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        திரும்பு nullptr;
    }
    tensor_map[id] = result;
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        எனில் (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } இல்லையெனில் {
            result->src[i] = create_முனை(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            எனில் (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                திரும்பு nullptr;
            }
        }
    }

    // Handle view_src similarly
    எனில் (tensor->view_src == 0) {
        result->view_src = nullptr;
    } இல்லையெனில் {
        result->view_src = create_முனை(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        எனில் (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            திரும்பு nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    திரும்பு result;
}

தருக்க rpc_server::graph_compute(மாறிலி std::vector<uint8_t> & உள்ளீடு) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    எனில் (உள்ளீடு.size() < 2*அளவு(uint32_t) + அளவு(uint64_t)) {
        திரும்பு பொய்;
    }
    மாறிலி uint8_t * src = உள்ளீடு.data();
    uint32_t device;
    memcpy(&device, src, அளவு(device));
    src += அளவு(device);
    எனில் (device >= backends.size()) {
        திரும்பு பொய்;
    }
    uint64_t uid;
    memcpy(&uid, src, அளவு(uid));
    src += அளவு(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, அளவு(n_nodes));
    src += அளவு(n_nodes);
    எனில் (உள்ளீடு.size() < 2*அளவு(uint32_t) + அளவு(uint64_t) + n_nodes*அளவு(uint64_t) + அளவு(uint32_t)) {
        திரும்பு பொய்;
    }
    மாறிலி uint64_t * nodes = (மாறிலி uint64_t *)src;
    src += n_nodes*அளவு(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, அளவு(n_tensors));
    src += அளவு(n_tensors);
    எனில் (உள்ளீடு.size() < 2*அளவு(uint32_t) + அளவு(uint64_t) + n_nodes*அளவு(uint64_t) + அளவு(uint32_t) + n_tensors*அளவு(rpc_tensor)) {
        திரும்பு பொய்;
    }
    மாறிலி rpc_tensor * tensors = (மாறிலி rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    எனில் (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, பொய்);
    எனில் (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    கட்டமைப்பு ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    கட்டமைப்பு ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, பொய்);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, மாறிலி rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    க்கு (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    மாறிலி auto t_deserialize = std::chrono::steady_clock::now();
    க்கு (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], அளவு(id));
        graph->nodes[i] = create_முனை(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        எனில் (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            திரும்பு பொய்;
        }
        எனில் (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            மாறிலி size_t hash_pos = ggml_hash_செருகு(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_மொத்தம்;
        }
    }
    மாறிலி std::chrono::duration<இரட்டை, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    திரும்பு உண்மை;
}

தருக்க rpc_server::graph_recompute(மாறிலி rpc_செய்தி_graph_recompute_req & request) {
    uint32_t device = request.device;
    எனில் (device >= backends.size()) {
        திரும்பு பொய்;
    }
    auto it = stored_graphs[device].find(request.uid);
    எனில் (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        திரும்பு பொய்;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    திரும்பு உண்மை;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
வெறுமை rpc_server::sync_all_backends() {
    க்கு (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
தருக்க rpc_server::comm_init(மாறிலி rpc_செய்தி_comm_init_req & request, rpc_செய்தி_comm_init_rsp & response) {
    response.ok = 0;
    எனில் (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        திரும்பு உண்மை;
    }
    comm_state & state = comm_states[request.device];
    எனில் (state.peer != nullptr) {
        response.ok = 1;
        திரும்பு உண்மை;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    எனில் (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        எனில் (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            திரும்பு உண்மை;
        }
        state.peer = srv->accept();
        எனில் (state.peer == nullptr) {
            திரும்பு உண்மை;
        }
        எனில் (!state.peer->recv_data(remote_caps, அளவு(remote_caps))) {
            state.peer = nullptr;
            திரும்பு உண்மை;
        }
        state.peer->get_caps(local_caps);
        எனில் (!state.peer->send_data(local_caps, அளவு(local_caps))) {
            state.peer = nullptr;
            திரும்பு உண்மை;
        }
        state.peer->update_caps(remote_caps);
    } இல்லையெனில் {
        மாறிலி std::string host(request.host, strnlen(request.host, அளவு(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        க்கு (முழு i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            எனில் (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        எனில் (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            திரும்பு உண்மை;
        }
        state.peer->get_caps(local_caps);
        எனில் (!state.peer->send_data(local_caps, அளவு(local_caps)) ||
            !state.peer->recv_data(remote_caps, அளவு(remote_caps))) {
            state.peer = nullptr;
            திரும்பு உண்மை;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    திரும்பு உண்மை;
}

தருக்க rpc_server::comm_allreduce(மாறிலி rpc_செய்தி_comm_allreduce_req & request) {
    எனில் (request.device >= backends.size()) {
        திரும்பு பொய்;
    }
    comm_state & state = comm_states[request.device];
    எனில் (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        திரும்பு பொய்;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, பொய்);
    கட்டமைப்பு ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ வெற்று,
        /*.no_alloc   =*/ உண்மை,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    எனில் (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        திரும்பு பொய்;
    }
    மாறிலி size_t  nbytes = ggml_nbytes(t_dst);
    மாறிலி int64_t ne     = ggml_nelements(t_dst);
    எனில் (nbytes == 0) {
        திரும்பு உண்மை;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    மாறிலி தருக்க   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    மாறிலி size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    மாறிலி size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    எனில் (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    எழுத்து * scratch_base = (எழுத்து *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        திரும்பு t;
    };
    auto new_cpy_முனை = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        திரும்பு t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, பொய்);
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
    எனில் (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_முனை(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } இல்லையெனில் {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    எனில் (state.rank == 0) {
        எனில் (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            திரும்பு பொய்;
        }
    } இல்லையெனில் {
        எனில் (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            திரும்பு பொய்;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    எனில் (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_முனை(t_wire_recv, t_peer);
    } இல்லையெனில் {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    எனில் (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } இல்லையெனில் {
        compute_nodes(t_red, nullptr);
    }
    திரும்பு உண்மை;
}

தருக்க rpc_server::comm_free(மாறிலி rpc_செய்தி_comm_free_req & request) {
    எனில் (request.device >= backends.size()) {
        திரும்பு பொய்;
    }
    comm_states[request.device] = comm_state();
    திரும்பு உண்மை;
}

தருக்க rpc_server::get_device_memory(மாறிலி rpc_செய்தி_get_device_memory_req & request, rpc_செய்தி_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    எனில் (dev_id >= backends.size()) {
        திரும்பு பொய்;
    }
    size_t விடுவி, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &விடுவி, &total);
    response.free_mem = விடுவி;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    திரும்பு உண்மை;
}

rpc_server::~rpc_server() {
    க்கு (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

நிலையான வெறுமை rpc_serve_client(மாறிலி std::vector<ggml_backend_t> & backends, மாறிலி எழுத்து * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    எனில் (!sock->recv_data(&cmd, 1)) {
        திரும்பு;
    }
    எனில் (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        திரும்பு;
    }

    // Read input_size and validate protocol version
    uint64_t hello_உள்ளீடு_size;
    எனில் (!sock->recv_data(&hello_உள்ளீடு_size, அளவு(hello_உள்ளீடு_size))) {
        திரும்பு;
    }

    எனில் (hello_உள்ளீடு_size != அளவு(rpc_செய்தி_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_உள்ளீடு_size, அளவு(rpc_செய்தி_hello_req), RPC_PROTO_MAJOR_VERSION);
        திரும்பு;
    }

    rpc_செய்தி_hello_req req = {};
    எனில் (!sock->recv_data(&req, அளவு(req))) {
        திரும்பு;
    }

    rpc_செய்தி_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    எனில் (!send_செய்தி(sock, &rsp, அளவு(rsp))) {
        திரும்பு;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    போது (உண்மை) {
        எனில் (!sock->recv_data(&cmd, 1)) {
            நிறுத்து;
        }
        எனில் (cmd >= RPC_CMD_மொத்தம்) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            நிறுத்து;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((எண்ணிக்கை rpc_cmd)cmd));
        தேர்வு (cmd) {
            நிலை RPC_CMD_HELLO: {
                // HELLO command is handled above
                திரும்பு;
            }
            நிலை RPC_CMD_DEVICE_மொத்தம்: {
                எனில் (!recv_செய்தி(sock, nullptr, 0)) {
                    திரும்பு;
                }
                rpc_செய்தி_device_மொத்தம்_rsp response;
                response.device_மொத்தம் = backends.size();
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_ALLOC_BUFFER: {
                rpc_செய்தி_alloc_buffer_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_alloc_buffer_rsp response;
                எனில் (!server.alloc_buffer(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GET_ALLOC_SIZE: {
                rpc_செய்தி_get_alloc_size_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_get_alloc_size_rsp response;
                எனில் (!server.get_alloc_size(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GET_ALIGNMENT: {
                rpc_செய்தி_get_alignment_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_get_alignment_rsp response;
                எனில் (!server.get_alignment(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GET_MAX_SIZE: {
                rpc_செய்தி_get_max_size_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_get_max_size_rsp response;
                எனில் (!server.get_max_size(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_BUFFER_GET_BASE: {
                rpc_செய்தி_buffer_get_base_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_buffer_get_base_rsp response;
                எனில் (!server.buffer_get_base(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_FREE_BUFFER: {
                rpc_செய்தி_free_buffer_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.free_buffer(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_BUFFER_CLEAR: {
                rpc_செய்தி_buffer_clear_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.buffer_clear(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_MEMSET_TENSOR: {
                rpc_செய்தி_memset_tensor_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.memset_tensor(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> உள்ளீடு;
                எனில் (!recv_செய்தி(sock, உள்ளீடு)) {
                    திரும்பு;
                }
                எனில் (!server.set_tensor(உள்ளீடு)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> உள்ளீடு;
                எனில் (!recv_செய்தி(sock, உள்ளீடு)) {
                    திரும்பு;
                }
                எனில் (!server.set_tensor_2d(உள்ளீடு)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GET_TENSOR_2D: {
                rpc_செய்தி_get_tensor_2d_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                std::vector<uint8_t> response;
                எனில் (!server.get_tensor_2d(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, response.data(), response.size())) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_SET_TENSOR_HASH: {
                rpc_செய்தி_set_tensor_hash_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_set_tensor_hash_rsp response;
                எனில் (!server.set_tensor_hash(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_INIT_TENSOR: {
                rpc_செய்தி_init_tensor_req request;
                எனில் (!recv_செய்தி(sock, &request,அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.init_tensor(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GET_TENSOR: {
                rpc_செய்தி_get_tensor_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                std::vector<uint8_t> response;
                எனில் (!server.get_tensor(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, response.data(), response.size())) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_COPY_TENSOR: {
                rpc_செய்தி_copy_tensor_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_copy_tensor_rsp response;
                எனில் (!server.copy_tensor(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> உள்ளீடு;
                எனில் (!recv_செய்தி(sock, உள்ளீடு)) {
                    திரும்பு;
                }
                எனில் (!server.graph_compute(உள்ளீடு)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_செய்தி_graph_recompute_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.graph_recompute(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_COMM_INIT: {
                rpc_செய்தி_comm_init_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_comm_init_rsp response;
                எனில் (!server.comm_init(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_COMM_ALLREDUCE: {
                rpc_செய்தி_comm_allreduce_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.comm_allreduce(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_COMM_FREE: {
                rpc_செய்தி_comm_free_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                எனில் (!server.comm_free(request)) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            நிலை RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_செய்தி_get_device_memory_req request;
                எனில் (!recv_செய்தி(sock, &request, அளவு(request))) {
                    திரும்பு;
                }
                rpc_செய்தி_get_device_memory_rsp response;
                எனில் (!server.get_device_memory(request, response)) {
                    திரும்பு;
                }
                எனில் (!send_செய்தி(sock, &response, அளவு(response))) {
                    திரும்பு;
                }
                நிறுத்து;
            }
            இயல்பு: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((எண்ணிக்கை rpc_cmd)cmd));
                திரும்பு;
            }
        }
    }
}

வெறுமை ggml_backend_rpc_start_server(மாறிலி எழுத்து * endpoint, மாறிலி எழுத்து * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    எனில் (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        திரும்பு;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    க்கு (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t விடுவி, total;
        ggml_backend_dev_memory(dev, &விடுவி, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, விடுவி / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        எனில் (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            திரும்பு;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        எனில் (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            எனில் (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    முழு port;
    எனில் (!parse_endpoint(endpoint, host, port)) {
        திரும்பு;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#இல்லையெனில்
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    எனில் (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        திரும்பு;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    எனில் (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        திரும்பு;
    }
    போது (உண்மை) {
        auto client_socket = server_socket->accept();
        எனில் (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            திரும்பு;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    க்கு (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

நிலையான மாறிலி எழுத்து * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    திரும்பு ctx->name.c_str();
}

நிலையான மாறிலி எழுத்து * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    திரும்பு ctx->description.c_str();
}

நிலையான வெறுமை ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, விடுவி, total);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    திரும்பு GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, கட்டமைப்பு ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ உண்மை,
        /* .host_buffer           = */ பொய்,
        /* .buffer_from_host_ptr  = */ பொய்,
        /* .events                = */ உண்மை,
        /* .mmap_support          = */ உண்மை,
    };
}

நிலையான ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    திரும்பு ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

நிலையான ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    திரும்பு ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

நிலையான தருக்க ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    எனில் (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        திரும்பு பொய்;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    திரும்பு buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

நிலையான ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    திரும்பு dispatcher->event_new(dev);
}

நிலையான வெறுமை ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

நிலையான வெறுமை ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ வெற்று,
    /* .buffer_from_host_ptr = */ வெற்று,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ வெற்று,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

கட்டமைப்பு ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

நிலையான மாறிலி எழுத்து * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    திரும்பு ctx ? ctx->name.c_str() : "RPC";
}

நிலையான size_t ggml_backend_rpc_reg_get_device_மொத்தம்(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    திரும்பு ctx ? ctx->devices.size() : 0;
}

நிலையான ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    எனில் (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } இல்லையெனில் {
        GGML_ASSERT(index < ctx->devices.size());
        திரும்பு ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
கட்டமைப்பு ggml_backend_rpc_comm_context {
    கட்டமைப்பு rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

நிலையான வெறுமை ggml_backend_rpc_comm_free(வெறுமை * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    எனில் (comm_ctx == nullptr) {
        திரும்பு;
    }
    க்கு (மாறிலி auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_செய்தி_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, அளவு(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

நிலையான வெறுமை * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    எனில் (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        திரும்பு nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    க்கு (size_t i = 0; i < n_backends; i++) {
        எனில் (!ggml_backend_is_rpc(backends[i])) {
            திரும்பு nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        க்கு (மாறிலி auto & rank : ranks) {
            எனில் (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                திரும்பு nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    முழு port0;
    எனில் (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        திரும்பு nullptr;
    }
    மாறிலி uint32_t comm_port = (uint32_t) port0 + 1000;
    எனில் (host0.size() >= 64) {
        திரும்பு nullptr;
    }

    க்கு (மாறிலி auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_செய்தி_comm_init_rsp> responses(n_backends);
    க்கு (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_செய்தி_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        எனில் (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, அளவு(*request), &responses[i], அளவு(responses[i]));
    }
    க்கு (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    தருக்க ok = உண்மை;
    க்கு (size_t i = 0; i < n_backends; i++) {
        எனில் (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = பொய்;
        }
    }
    எனில் (!ok) {
        க்கு (மாறிலி auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        திரும்பு nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    திரும்பு புதிய ggml_backend_rpc_comm_context{std::move(ranks)};
}

நிலையான தருக்க ggml_backend_rpc_comm_allreduce_tensor(வெறுமை * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    எனில் (comm_ctx == nullptr) {
        திரும்பு பொய்;
    }
    மாறிலி size_t n_ranks = comm_ctx->ranks.size();
    மாறிலி int64_t ne = ggml_nelements(tensors[0]);
    எனில் (ne == 0) {
        திரும்பு உண்மை;
    }
    க்கு (size_t i = 0; i < n_ranks; i++) {
        எனில் (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            திரும்பு பொய்;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        எனில் ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            திரும்பு பொய்;
        }
    }
    க்கு (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_செய்தி_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, அளவு(*request));
    }
    திரும்பு உண்மை;
}

நிலையான வெறுமை * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    எனில் (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_rpc_add_server;
    }
    எனில் (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_rpc_start_server;
    }
    எனில் (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_rpc_comm_init;
    }
    எனில் (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_rpc_comm_free;
    }
    எனில் (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        திரும்பு (வெறுமை *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    திரும்பு வெற்று;

    GGML_UNUSED(reg);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(வெறுமை) {
    நிலையான கட்டமைப்பு ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ வெற்று,
    };

    திரும்பு &ggml_backend_rpc_reg;
}

நிலையான uint32_t ggml_backend_rpc_get_device_மொத்தம்(மாறிலி எழுத்து * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_செய்தி_device_மொத்தம்_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_மொத்தம், nullptr, 0, &response, அளவு(response));
    திரும்பு response.device_மொத்தம்;
}

நிலையான மாறிலி ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_மொத்தம்,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(மாறிலி எழுத்து * endpoint) {
    நிலையான std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    நிலையான std::mutex mutex;
    நிலையான uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    எனில் (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        திரும்பு reg_map[endpoint];
    }
    uint32_t dev_மொத்தம் = ggml_backend_rpc_get_device_மொத்தம்(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_மொத்தம்);
    எனில் (dev_மொத்தம் == 0) {
        திரும்பு nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = புதிய ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    க்கு (uint32_t ind = 0; ind < dev_மொத்தம்; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = புதிய ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = புதிய ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = புதிய ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    திரும்பு reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
