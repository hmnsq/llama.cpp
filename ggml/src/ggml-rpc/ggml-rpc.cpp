#অন্তর্ভাব "ggml-rpc.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-cpp.h"
#অন্তর্ভাব "transport.h"
#অন্তর্ভাব "log.h"

#অন্তর্ভাব <array>
#অন্তর্ভাব <chrono>
#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <optional>
#অন্তর্ভাব <string>
#অন্তর্ভাব <thread>
#অন্তর্ভাব <vector>
#অন্তর্ভাব <queue>
#অন্তর্ভাব <condition_variable>
#অন্তর্ভাব <future>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <fstream>
#অন্তর্ভাব <filesystem>
#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <atomic>

নামাকাশ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) যদি (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(ঠেলো, 1)
// ggml_tensor is serialized into rpc_tensor
কাঠামো rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t বাফার;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / আকার(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t তথ্য;
    অক্ষর name[GGML_MAX_NAME];

    int32_t use_গুনতি;
};

static_assert(আকার(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
গণনা rpc_cmd {
    RPC_CMD_ALLOC_বাফার = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_আকার,
    RPC_CMD_বাফার_GET_BASE,
    RPC_CMD_FREE_বাফার,
    RPC_CMD_বাফার_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_আকার,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_গুনতি,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_গুনতি,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

স্থির ধ্রুবক অক্ষর * rpc_cmd_name(গণনা rpc_cmd cmd) {
    নির্বাচন (cmd) {
        ক্ষেত্র RPC_CMD_ALLOC_বাফার:      ফেরত "ALLOC_BUFFER";
        ক্ষেত্র RPC_CMD_GET_ALIGNMENT:     ফেরত "GET_ALIGNMENT";
        ক্ষেত্র RPC_CMD_GET_MAX_আকার:      ফেরত "GET_MAX_SIZE";
        ক্ষেত্র RPC_CMD_বাফার_GET_BASE:   ফেরত "BUFFER_GET_BASE";
        ক্ষেত্র RPC_CMD_FREE_বাফার:       ফেরত "FREE_BUFFER";
        ক্ষেত্র RPC_CMD_বাফার_CLEAR:      ফেরত "BUFFER_CLEAR";
        ক্ষেত্র RPC_CMD_SET_TENSOR:        ফেরত "SET_TENSOR";
        ক্ষেত্র RPC_CMD_SET_TENSOR_HASH:   ফেরত "SET_TENSOR_HASH";
        ক্ষেত্র RPC_CMD_GET_TENSOR:        ফেরত "GET_TENSOR";
        ক্ষেত্র RPC_CMD_COPY_TENSOR:       ফেরত "COPY_TENSOR";
        ক্ষেত্র RPC_CMD_GRAPH_COMPUTE:     ফেরত "GRAPH_COMPUTE";
        ক্ষেত্র RPC_CMD_GET_DEVICE_MEMORY: ফেরত "GET_DEVICE_MEMORY";
        ক্ষেত্র RPC_CMD_INIT_TENSOR:       ফেরত "INIT_TENSOR";
        ক্ষেত্র RPC_CMD_GET_ALLOC_আকার:    ফেরত "GET_ALLOC_SIZE";
        ক্ষেত্র RPC_CMD_HELLO:             ফেরত "HELLO";
        ক্ষেত্র RPC_CMD_DEVICE_গুনতি:      ফেরত "DEVICE_COUNT";
        ক্ষেত্র RPC_CMD_GRAPH_RECOMPUTE:   ফেরত "GRAPH_RECOMPUTE";
        ক্ষেত্র RPC_CMD_MEMSET_TENSOR:     ফেরত "MEMSET_TENSOR";
        ক্ষেত্র RPC_CMD_SET_TENSOR_2D:     ফেরত "SET_TENSOR_2D";
        ক্ষেত্র RPC_CMD_GET_TENSOR_2D:     ফেরত "GET_TENSOR_2D";
        ক্ষেত্র RPC_CMD_COMM_INIT:         ফেরত "COMM_INIT";
        ক্ষেত্র RPC_CMD_COMM_ALLREDUCE:    ফেরত "COMM_ALLREDUCE";
        ক্ষেত্র RPC_CMD_COMM_FREE:         ফেরত "COMM_FREE";
        ক্ষেত্র RPC_CMD_NONE:              ফেরত "NONE";
        পূর্বনির্ধারিত:                        ফেরত "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ধ্রুবক size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ধ্রুবক size_t GRAPH_CACHE_MAX = 1024;

কাঠামো rpc_বার্তা_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_আকার];
};

কাঠামো rpc_বার্তা_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_আকার];
};

কাঠামো rpc_বার্তা_device_গুনতি_rsp {
    uint32_t device_গুনতি;
};

কাঠামো rpc_বার্তা_get_alloc_আকার_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

কাঠামো rpc_বার্তা_get_alloc_আকার_rsp {
    uint64_t alloc_আকার;
};

কাঠামো rpc_বার্তা_init_tensor_req {
    rpc_tensor tensor;
};

কাঠামো rpc_বার্তা_alloc_বাফার_req {
    uint32_t device;
    uint64_t size;
};

কাঠামো rpc_বার্তা_alloc_বাফার_rsp {
    uint64_t remote_ptr;
    uint64_t remote_আকার;
};

কাঠামো rpc_বার্তা_get_alignment_req {
    uint32_t device;
};

কাঠামো rpc_বার্তা_get_alignment_rsp {
    uint64_t alignment;
};

কাঠামো rpc_বার্তা_get_max_আকার_req {
    uint32_t device;
};

কাঠামো rpc_বার্তা_get_max_আকার_rsp {
    uint64_t max_size;
};

কাঠামো rpc_বার্তা_বাফার_get_base_req {
    uint64_t remote_ptr;
};

কাঠামো rpc_বার্তা_বাফার_get_base_rsp {
    uint64_t base_ptr;
};

কাঠামো rpc_বার্তা_free_বাফার_req {
    uint64_t remote_ptr;
};

কাঠামো rpc_বার্তা_বাফার_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

কাঠামো rpc_বার্তা_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

কাঠামো rpc_বার্তা_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

কাঠামো rpc_বার্তা_set_tensor_hash_rsp {
    uint8_t ফলাফল;
};

কাঠামো rpc_বার্তা_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

কাঠামো rpc_বার্তা_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

কাঠামো rpc_বার্তা_copy_tensor_rsp {
    uint8_t ফলাফল;
};

কাঠামো rpc_বার্তা_get_device_memory_req {
    uint32_t device;
};

কাঠামো rpc_বার্তা_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t যোগ_mem;
};

কাঠামো rpc_বার্তা_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

কাঠামো rpc_বার্তা_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

কাঠামো rpc_বার্তা_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    অক্ষর     host[64];  // rank > 0: rank 0's host
};

কাঠামো rpc_বার্তা_comm_init_rsp {
    uint8_t ok;
};

কাঠামো rpc_বার্তা_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

কাঠামো rpc_বার্তা_comm_free_req {
    uint32_t device;
};

#pragma pack(বের_করো)

// RPC data structures

স্থির ggml_guid_t ggml_backend_rpc_guid() {
    স্থির ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ফেরত &guid;
}

কাঠামো ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

কাঠামো ggml_backend_rpc_বাফার_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

শ্রেণী rpc_dispatcher;
কাঠামো ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

কাঠামো ggml_backend_rpc_বাফার_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    শূন্য                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
স্থির uint64_t fnv_hash(ধ্রুবক uint8_t * তথ্য, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ধ্রুবক uint64_t fnv_মৌলিক = 0x100000001b3ULL;

    জন্য (size_t i = 0; i < len; ++i) {
        hash ^= তথ্য[i];
        hash *= fnv_মৌলিক;
    }
    ফেরত hash;
}

স্থির যৌক্তিক send_বার্তা(socket_ptr sock, ধ্রুবক শূন্য * বার্তা, size_t বার্তা_আকার) {
    যদি (!sock->send_তথ্য(&বার্তা_আকার, আকার(বার্তা_আকার))) {
        ফেরত মিথ্যা;
    }
    যদি (!sock->send_তথ্য(বার্তা, বার্তা_আকার)) {
        ফেরত মিথ্যা;
    }
    ফেরত sock->flush();
}

স্থির যৌক্তিক recv_বার্তা(socket_ptr sock, শূন্য * বার্তা, size_t বার্তা_আকার) {
    uint64_t size;
    যদি (!sock->recv_তথ্য(&size, আকার(size))) {
        ফেরত মিথ্যা;
    }
    যদি (size != বার্তা_আকার) {
        ফেরত মিথ্যা;
    }
    ফেরত sock->recv_তথ্য(বার্তা, বার্তা_আকার);
}

স্থির যৌক্তিক recv_বার্তা(socket_ptr sock, std::vector<uint8_t> & প্রবেশ) {
    uint64_t size;
    যদি (!sock->recv_তথ্য(&size, আকার(size))) {
        ফেরত মিথ্যা;
    }
    চেষ্টা {
        প্রবেশ.resize(size);
    } ধরো (ধ্রুবক std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ফেরত মিথ্যা;
    }
    ফেরত sock->recv_তথ্য(প্রবেশ.তথ্য(), size);
}

স্থির যৌক্তিক parse_endpoint(ধ্রুবক std::string & endpoint, std::string & host, পূর্ণসংখ্যা & port) {
    size_t pos = endpoint.find(':');
    যদি (pos == std::string::npos) {
        ফেরত মিথ্যা;
    }
    host = endpoint.substr(0, pos);
    চেষ্টা {
        port = std::stoi(endpoint.substr(pos + 1));
    } ধরো (...) {
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
স্থির যৌক্তিক send_rpc_cmd(socket_ptr sock, গণনা rpc_cmd cmd, ধ্রুবক শূন্য * প্রবেশ, size_t প্রবেশ_আকার) {
    uint8_t cmd_byte = cmd;
    যদি (!sock->send_তথ্য(&cmd_byte, আকার(cmd_byte))) {
        ফেরত মিথ্যা;
    }
    যদি (!sock->send_তথ্য(&প্রবেশ_আকার, আকার(প্রবেশ_আকার))) {
        ফেরত মিথ্যা;
    }
    যদি (!sock->send_তথ্য(প্রবেশ, প্রবেশ_আকার)) {
        ফেরত মিথ্যা;
    }
    ফেরত sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
স্থির যৌক্তিক send_rpc_cmd(socket_ptr sock, গণনা rpc_cmd cmd, ধ্রুবক শূন্য * প্রবেশ, size_t প্রবেশ_আকার, শূন্য * output, size_t output_আকার) {
    যদি (!send_rpc_cmd(sock, cmd, প্রবেশ, প্রবেশ_আকার)) {
        ফেরত মিথ্যা;
    }
    uint64_t out_আকার;
    যদি (!sock->recv_তথ্য(&out_আকার, আকার(out_আকার))) {
        ফেরত মিথ্যা;
    }
    যদি (out_আকার != output_আকার) {
        ফেরত মিথ্যা;
    }
    যদি (!sock->recv_তথ্য(output, output_আকার)) {
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
স্থির constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

স্থির অন্তঃরেখ শূন্য rpc_cpu_relax() {
#যদি defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ অস্থির("yield" ::: "memory");
#অন্যথা
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
স্থির যৌক্তিক negotiate_hello(ধ্রুবক std::shared_ptr<socket_t> & sock) {
    rpc_বার্তা_hello_req request = {};
    rpc_বার্তা_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    যৌক্তিক status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, আকার(request), &response, আকার(response));
    RPC_STATUS_ASSERT(status);

    যদি (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ফেরত মিথ্যা;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ফেরত সত্য;
}

template <typename T>
শ্রেণী message_queue {
সর্বজনীন:
    message_queue() {}

    যৌক্তিক ঠেলো(ধ্রুবক T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        যদি (interrupted) {
            ফেরত মিথ্যা;
        }
        queue.ঠেলো(value);
        cvar.notify_all();
        ফেরত সত্য;
    }

    যৌক্তিক বের_করো(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [এই] { ফেরত !queue.empty() || interrupted; });
        যদি (interrupted) {
            ফেরত মিথ্যা;
        }
        *out = queue.front();
        queue.বের_করো();
        ফেরত সত্য;
    }

    যৌক্তিক try_বের_করো(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        যদি (interrupted || queue.empty()) {
            ফেরত মিথ্যা;
        }
        *out = queue.front();
        queue.বের_করো();
        ফেরত সত্য;
    }

    শূন্য interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = সত্য;
        lock.unlock();
        cvar.notify_all();
    }

ব্যক্তিগত:
    যৌক্তিক interrupted = মিথ্যা;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

শ্রেণী rpc_dispatcher {
সর্বজনীন:
    rpc_dispatcher() {
    }

    শূন্য send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার);
    শূন্য send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার, শূন্য * output, size_t output_আকার);
    শূন্য send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার);
    শূন্য send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার, শূন্য * output, size_t output_আকার);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    শূন্য event_free(ggml_backend_event_t event);
    শূন্য event_synchronize(ggml_backend_event_t event);
    শূন্য event_record(ggml_backend_event_t event);
    শূন্য synchronize();
    শূন্য busy_spin_acquire();
    শূন্য busy_spin_release();
    শূন্য graph_compute(uint32_t device, ধ্রুবক ggml_cgraph * cgraph);

    শূন্য start(ধ্রুবক std::string & endpoint);
    শূন্য work();

    ~rpc_dispatcher();

ব্যক্তিগত:
    কাঠামো rpc_বার্তা {
        rpc_cmd                       cmd;
        std::shared_ptr<ধ্রুবক শূন্য>   প্রবেশ;
        size_t                        প্রবেশ_আকার;
        শূন্য                        * output;
        size_t                        output_আকার;
        std::promise<শূন্য>            completion;
    };
    using rpc_বার্তা_ptr   = std::shared_ptr<rpc_বার্তা>;
    using rpc_বার্তা_queue = message_queue<rpc_বার্তা_ptr>;
    কাঠামো rpc_event {
        rpc_বার্তা_ptr              বার্তা;
        std::shared_future<শূন্য> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_বার্তা_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

স্থির শূন্য rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

শূন্য rpc_dispatcher::send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার) {
    auto বার্তা = std::make_shared<rpc_বার্তা>();
    বার্তা->cmd = cmd;
    বার্তা->প্রবেশ = প্রবেশ;
    বার্তা->প্রবেশ_আকার = প্রবেশ_আকার;
    বার্তা->output = nullptr;
    বার্তা->output_আকার = 0;
    GGML_ASSERT(queue.ঠেলো(বার্তা));
    auto future = বার্তা->completion.get_future();
    future.wait();
}

শূন্য rpc_dispatcher::send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার) {
    auto বার্তা = std::make_shared<rpc_বার্তা>();
    বার্তা->cmd = cmd;
    বার্তা->প্রবেশ = প্রবেশ;
    বার্তা->প্রবেশ_আকার = প্রবেশ_আকার;
    বার্তা->output = nullptr;
    বার্তা->output_আকার = 0;
    GGML_ASSERT(queue.ঠেলো(বার্তা));
}

শূন্য rpc_dispatcher::send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার, শূন্য * output, size_t output_আকার) {
    auto বার্তা = std::make_shared<rpc_বার্তা>();
    বার্তা->cmd = cmd;
    বার্তা->প্রবেশ = প্রবেশ;
    বার্তা->প্রবেশ_আকার = প্রবেশ_আকার;
    বার্তা->output = output;
    বার্তা->output_আকার = output_আকার;
    GGML_ASSERT(queue.ঠেলো(বার্তা));
    auto future = বার্তা->completion.get_future();
    future.wait();
}

শূন্য rpc_dispatcher::send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুবক শূন্য> প্রবেশ, size_t প্রবেশ_আকার, শূন্য * output, size_t output_আকার) {
    auto বার্তা = std::make_shared<rpc_বার্তা>();
    বার্তা->cmd = cmd;
    বার্তা->প্রবেশ = প্রবেশ;
    বার্তা->প্রবেশ_আকার = প্রবেশ_আকার;
    বার্তা->output = output;
    বার্তা->output_আকার = output_আকার;
    GGML_ASSERT(queue.ঠেলো(বার্তা));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = নতুন rpc_event;
    ev->বার্তা = std::make_shared<rpc_বার্তা>();
    ev->বার্তা->cmd = RPC_CMD_NONE;
    ev->sf = ev->বার্তা->completion.get_future().share();
    GGML_ASSERT(queue.ঠেলো(ev->বার্তা));
    ফেরত নতুন ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

শূন্য rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

শূন্য rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

শূন্য rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->বার্তা = std::make_shared<rpc_বার্তা>();
    ev->বার্তা->cmd = RPC_CMD_NONE;
    ev->sf = ev->বার্তা->completion.get_future().share();
    GGML_ASSERT(queue.ঠেলো(ev->বার্তা));
}

শূন্য rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto বার্তা = std::make_shared<rpc_বার্তা>();
    বার্তা->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ঠেলো(বার্তা));
    বার্তা->completion.get_future().wait();
}

শূন্য rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

শূন্য rpc_dispatcher::busy_spin_release() {
    ধ্রুবক unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

শূন্য rpc_dispatcher::start(ধ্রুবক std::string & endpoint) {
    std::string host;
    পূর্ণসংখ্যা port;
    যদি (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    যদি (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    যদি (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    যদি (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = সত্য;
    thread = std::thread(rpc_dispatcher_trampoline, এই);
}

শূন্য rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    যখন (running) {
        rpc_বার্তা_ptr বার্তা_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ধ্রুবক যৌক্তিক spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        যদি (spin) {
            যদি (!queue.try_বের_করো(&বার্তা_ptr)) {
                rpc_cpu_relax();
                চালু;
            }
        } অন্যথা যদি (!queue.বের_করো(&বার্তা_ptr)) {
            বিরতি;
        }
        যদি (বার্তা_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(বার্তা_ptr->cmd),
                     rpc_log_format_আকার(বার্তা_ptr->প্রবেশ_আকার).c_str(), rpc_log_format_আকার(বার্তা_ptr->output_আকার).c_str());
            ধ্রুবক যৌক্তিক trace = rpc_log_debug_level() >= 3;
            ধ্রুবক auto t0 = std::chrono::steady_clock::now();
            যদি (বার্তা_ptr->output) {
                যৌক্তিক status = send_rpc_cmd(sock, বার্তা_ptr->cmd, বার্তা_ptr->প্রবেশ.get(), বার্তা_ptr->প্রবেশ_আকার, বার্তা_ptr->output, বার্তা_ptr->output_আকার);
                RPC_STATUS_ASSERT(status);
            } অন্যথা {
                যৌক্তিক status = send_rpc_cmd(sock, বার্তা_ptr->cmd, বার্তা_ptr->প্রবেশ.get(), বার্তা_ptr->প্রবেশ_আকার);
                RPC_STATUS_ASSERT(status);
            }
            যদি (trace) {
                ধ্রুবক std::chrono::duration<দ্বিগুণ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(বার্তা_ptr->cmd), elapsed.count());
            }
        } অন্যথা {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        বার্তা_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = মিথ্যা;
    queue.interrupt();
    sock = nullptr;
    যদি (thread.joinable()) {
        thread.join();
    }
}

স্থির std::shared_ptr<rpc_dispatcher> get_dispatcher(ধ্রুবক std::string & endpoint) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    স্থির std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    যদি (it != dispatchers.end()) {
        যদি (auto dispatcher = it->second.lock()) {
            ফেরত dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ফেরত dispatcher;
}

স্থির শূন্য ggml_backend_rpc_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    auto request = std::make_shared<rpc_বার্তা_free_বাফার_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_বাফার, request, আকার(*request));
    delete ctx;
}

স্থির শূন্য * ggml_backend_rpc_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    যদি (ctx->base_ptr != nullptr) {
        ফেরত ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_বার্তা_বাফার_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_বার্তা_বাফার_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_বাফার_GET_BASE, request, আকার(*request), &response, আকার(response));
    ctx->base_ptr = reinterpret_cast<শূন্য *>(response.base_ptr);
    ফেরত ctx->base_ptr;
}

স্থির যৌক্তিক ggml_backend_বাফার_is_rpc(ggml_backend_বাফার_t বাফার) {
    ফেরত বাফার->iface.free_বাফার == ggml_backend_rpc_বাফার_free_বাফার;
}

স্থির rpc_tensor serialize_tensor(ধ্রুবক ggml_tensor * tensor, ধ্রুবক rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor ফলাফল;
    যদি (!tensor) {
        memset(&ফলাফল, 0, আকার(ফলাফল));
        ফেরত ফলাফল;
    }

    ফলাফল.id = reinterpret_cast<uint64_t>(tensor);
    ফলাফল.type = tensor->type;
    যদি (tensor->বাফার && ggml_backend_বাফার_is_rpc(tensor->বাফার)) {
        ggml_backend_বাফার_t বাফার = tensor->বাফার;
        ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        যদি (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            ফলাফল.বাফার = ctx->remote_ptr;
            ফলাফল.তথ্য = reinterpret_cast<uint64_t>(tensor->তথ্য);
        } অন্যথা {
            ফলাফল.বাফার = 0;
            ফলাফল.তথ্য = 0;
        }
    } অন্যথা {
        ফলাফল.বাফার = 0;
        ফলাফল.তথ্য   = 0;
    }
    জন্য (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ফলাফল.ne[i] = tensor->ne[i];
        ফলাফল.nb[i] = tensor->nb[i];
    }
    ফলাফল.op = tensor->op;
    জন্য (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / আকার(int32_t); i++) {
        ফলাফল.op_params[i] = tensor->op_params[i];
    }
    ফলাফল.flags = tensor->flags;
    জন্য (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ফলাফল.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ফলাফল.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    ফলাফল.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(ফলাফল.name, 0, আকার(ফলাফল.name));
    ফলাফল.use_গুনতি = 0;

    snprintf(ফলাফল.name, GGML_MAX_NAME, "%s", tensor->name);
    ফেরত ফলাফল;
}

স্থির গণনা ggml_status ggml_backend_rpc_বাফার_init_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    যদি (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_বার্তা_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, আকার(*request));
    }
    ফেরত GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_rpc_বাফার_memset_tensor(
        ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    auto request = std::make_shared<rpc_বার্তা_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_আকার(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, আকার(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
স্থির std::shared_ptr<uint8_t> serialize_set_tensor(ধ্রুবক rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, ধ্রুবক শূন্য * তথ্য, size_t size, size_t & প্রবেশ_আকার) {
    প্রবেশ_আকার = আকার(rpc_tensor) + আকার(cache_flag) + আকার(offset) + size;
    uint8_t * প্রবেশ = নতুন uint8_t[প্রবেশ_আকার]();
    uint8_t * p = প্রবেশ;
    memcpy(p, &rpc_tensor, আকার(rpc_tensor)); p += আকার(rpc_tensor);
    memcpy(p, &cache_flag, আকার(cache_flag)); p += আকার(cache_flag);
    memcpy(p, &offset,     আকার(offset));     p += আকার(offset);
    memcpy(p, তথ্য, size);
    ফেরত std::shared_ptr<uint8_t>(প্রবেশ, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
স্থির যৌক্তিক rpc_use_hash_cache(ধ্রুবক ggml_tensor * tensor, size_t size) {
    ফেরত size > HASH_THRESHOLD && tensor->বাফার->usage == GGML_BACKEND_বাফার_USAGE_WEIGHTS;
}

স্থির শূন্য ggml_backend_rpc_বাফার_set_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_আকার(size).c_str());
    যদি (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_বার্তা_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ধ্রুবক uint8_t*)তথ্য, size);
        rpc_বার্তা_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, আকার(*request), &response, আকার(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ফলাফল ? "hit" : "miss");
        যদি (response.ফলাফল) {
            // the server has the same data, no need to send it
            ফেরত;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t প্রবেশ_আকার;
    auto প্রবেশ = serialize_set_tensor(rpc_tensor, cache_flag, offset, তথ্য, size, প্রবেশ_আকার);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, প্রবেশ, প্রবেশ_আকার);
}

স্থির শূন্য ggml_backend_rpc_বাফার_set_tensor_2d(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_আকার(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t প্রবেশ_আকার = আকার(rpc_tensor) + 4*আকার(uint64_t) + size*n_copies;
    uint8_t * প্রবেশ = নতুন uint8_t[প্রবেশ_আকার]();
    uint8_t * dest = প্রবেশ;
    memcpy(dest, &rpc_tensor, আকার(rpc_tensor));
    dest += আকার(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, আকার(header));
    dest += আকার(header);
    জন্য (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (ধ্রুবক অক্ষর *)তথ্য + i*stride_তথ্য, size);
    }
    std::shared_ptr<uint8_t> প্রবেশ_ptr(প্রবেশ, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, প্রবেশ_ptr, প্রবেশ_আকার);
}

স্থির শূন্য ggml_backend_rpc_বাফার_get_tensor_2d(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_আকার(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_বার্তা_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    যদি (stride_তথ্য == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, আকার(*request), তথ্য, size*n_copies);
    } অন্যথা {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, আকার(*request), packed.তথ্য(), packed.size());
        জন্য (size_t i = 0; i < n_copies; i++) {
            memcpy((অক্ষর *)তথ্য + i*stride_তথ্য, packed.তথ্য() + i*size, size);
        }
    }
}

স্থির শূন্য ggml_backend_rpc_বাফার_get_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_আকার(size).c_str());
    auto request = std::make_shared<rpc_বার্তা_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, আকার(*request), তথ্য, size);
}

স্থির যৌক্তিক ggml_backend_rpc_বাফার_cpy_tensor(ggml_backend_বাফার_t বাফার, ধ্রুবক ggml_tensor * src, ggml_tensor * dst) {
    যদি (ggml_backend_বাফার_is_rpc(src->বাফার)) {
        // check if src and dst are on the same server
        ggml_backend_বাফার_t src_বাফার = src->বাফার;
        ggml_backend_rpc_বাফার_context * src_ctx = (ggml_backend_rpc_বাফার_context *)src_বাফার->context;
        ggml_backend_বাফার_t dst_বাফার = dst->বাফার;
        ggml_backend_rpc_বাফার_context * dst_ctx = (ggml_backend_rpc_বাফার_context *)dst_বাফার->context;
        যদি (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ফেরত মিথ্যা;
        }
        ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_বার্তা_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_বার্তা_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, আকার(*request), &response, আকার(response));
        ফেরত response.ফলাফল;
    }
    ফেরত মিথ্যা;
}

স্থির শূন্য ggml_backend_rpc_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    ggml_backend_rpc_বাফার_context * ctx = (ggml_backend_rpc_বাফার_context *)বাফার->context;
    auto request = std::make_shared<rpc_বার্তা_বাফার_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_বাফার_CLEAR, request, আকার(*request));
}

স্থির ggml_backend_বাফার_i ggml_backend_rpc_বাফার_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_rpc_বাফার_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_বাফার_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_বাফার_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_বাফার_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_বাফার_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_rpc_বাফার_type_name(ggml_backend_বাফার_type_t buft) {
    ggml_backend_rpc_বাফার_type_context * buft_ctx = (ggml_backend_rpc_বাফার_type_context *)buft->context;
    ফেরত buft_ctx->name.c_str();
}

স্থির ggml_backend_বাফার_t ggml_backend_rpc_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    ggml_backend_rpc_বাফার_type_context * buft_ctx = (ggml_backend_rpc_বাফার_type_context *)buft->context;
    auto request = std::make_shared<rpc_বার্তা_alloc_বাফার_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_বার্তা_alloc_বাফার_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_বাফার, request, আকার(*request), &response, আকার(response));
    যদি (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_আকার(request->size).c_str(),
                response.remote_ptr, rpc_log_format_আকার(response.remote_আকার).c_str());
        ggml_backend_বাফার_t বাফার = ggml_backend_বাফার_init(buft,
            ggml_backend_rpc_বাফার_interface,
            নতুন ggml_backend_rpc_বাফার_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_আকার);
        ফেরত বাফার;
    } অন্যথা {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_আকার(request->size).c_str());
        ফেরত nullptr;
    }
}

স্থির size_t get_alignment(ধ্রুবক std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_বার্তা_get_alignment_req>();
    request->device = device;
    rpc_বার্তা_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, আকার(*request), &response, আকার(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_আকার(response.alignment).c_str());
    ফেরত response.alignment;
}

স্থির size_t ggml_backend_rpc_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    ggml_backend_rpc_বাফার_type_context * buft_ctx = (ggml_backend_rpc_বাফার_type_context *)buft->context;
    ফেরত buft_ctx->alignment;
}

স্থির size_t get_max_আকার(ধ্রুবক std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_বার্তা_get_max_আকার_req>();
    request->device = device;
    rpc_বার্তা_get_max_আকার_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_আকার, request, আকার(*request), &response, আকার(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_আকার(response.max_size).c_str());
    ফেরত response.max_size;
}

স্থির size_t ggml_backend_rpc_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    ggml_backend_rpc_বাফার_type_context * buft_ctx = (ggml_backend_rpc_বাফার_type_context *)buft->context;
    ফেরত buft_ctx->max_size;
}

স্থির size_t ggml_backend_rpc_বাফার_type_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    যৌক্তিক rpc_get = মিথ্যা;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_আকার_may_expand(tensor->op);

    যদি (rpc_get) {
        ggml_backend_rpc_বাফার_type_context * buft_ctx = (ggml_backend_rpc_বাফার_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ধ্রুবক size_t min_আকার = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        কাঠামো alloc_আকার_cache_চাবি {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / আকার(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_আকার_cache_চাবি চাবি = {};
        চাবি.device = buft_ctx->device;
        চাবি.type = tensor->type;
        চাবি.op = tensor->op;
        memcpy(চাবি.op_params, tensor->op_params, আকার(চাবি.op_params));
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; i++) {
            চাবি.ne[i] = (uint32_t)tensor->ne[i];
            চাবি.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ধ্রুবক uint8_t *)&চাবি, আকার(চাবি));
        cache_hash = fnv_hash((ধ্রুবক uint8_t *)buft_ctx->endpoint.তথ্য(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        স্থির std::mutex cache_mutex;
        স্থির std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            যদি (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_আকার(it->second).c_str());
                ফেরত std::max<size_t>(it->second, min_আকার);
            }
        }

        auto request = std::make_shared<rpc_বার্তা_get_alloc_আকার_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_বার্তা_get_alloc_আকার_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_আকার, request, আকার(*request), &response, আকার(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_আকার;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_আকার(response.alloc_আকার).c_str());

        ফেরত std::max<size_t>(response.alloc_আকার, min_আকার);
    }

    ফেরত ggml_nbytes(tensor);
}

স্থির ggml_backend_বাফার_type_i ggml_backend_rpc_বাফার_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_বাফার_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_বাফার_type_alloc_বাফার,
    /* .alloc_buffer_n      = */ ফাঁকা,
    /* .get_alignment       = */ ggml_backend_rpc_বাফার_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_আকার,
    /* .get_alloc_size      = */ ggml_backend_rpc_বাফার_type_get_alloc_আকার,
    /* .get_alloc_size_n    = */ ফাঁকা,
    /* .is_host             = */ ফাঁকা,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ফেরত rpc_ctx->name.c_str();
}

স্থির শূন্য ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

স্থির শূন্য ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_আকার(size).c_str());
    যদি (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_বার্তা_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ধ্রুবক uint8_t*)তথ্য, size);
        rpc_বার্তা_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, আকার(*request), &response, আকার(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ফলাফল ? "hit" : "miss");
        যদি (response.ফলাফল) {
            // the server has the same data, no need to send it
            ফেরত;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t প্রবেশ_আকার;
    auto প্রবেশ = serialize_set_tensor(rpc_tensor, cache_flag, offset, তথ্য, size, প্রবেশ_আকার);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, প্রবেশ, প্রবেশ_আকার);
}

স্থির শূন্য ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_আকার(size).c_str());
    auto request = std::make_shared<rpc_বার্তা_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, আকার(*request), তথ্য, size);
}

স্থির শূন্য ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

স্থির শূন্য add_tensor(ggml_tensor * tensor, ধ্রুবক ggml_cgraph * cgraph, ধ্রুবক rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    যদি (tensor == nullptr) {
        ফেরত;
    }
    যদি (visited.find(tensor) != visited.end()) {
        ফেরত;
    }
    visited.insert(tensor);
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor ফলাফল = serialize_tensor(tensor, dispatcher);
    ধ্রুবক size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    যদি (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        ফলাফল.use_গুনতি = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(ফলাফল);
}

স্থির uint8_t * serialize_graph(uint32_t device, ধ্রুবক ggml_cgraph * cgraph, ধ্রুবক rpc_dispatcher * dispatcher, size_t * output_আকার) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    জন্য (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_আকার = 2*আকার(uint32_t) + আকার(uint64_t) + n_nodes * আকার(uint64_t) + আকার(uint32_t) + n_tensors * আকার(rpc_tensor);
    uint8_t * output = নতুন uint8_t[*output_আকার]();
    uint8_t * dest = output;
    memcpy(dest, &device, আকার(device));
    dest += আকার(device);
    memcpy(dest, &cgraph->uid, আকার(cgraph->uid));
    dest += আকার(cgraph->uid);
    memcpy(dest, &n_nodes, আকার(n_nodes));
    dest += আকার(n_nodes);
    জন্য (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * আকার(uint64_t), &cgraph->nodes[i], আকার(uint64_t));
    }
    dest += n_nodes * আকার(uint64_t);
    memcpy(dest, &n_tensors, আকার(n_tensors));
    dest += আকার(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.তথ্য(), n_tensors * আকার(rpc_tensor));
    ফেরত output;
}

শূন্য rpc_dispatcher::graph_compute(uint32_t device, ধ্রুবক ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    যৌক্তিক reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    যদি (reuse) {
        auto request = std::make_shared<rpc_বার্তা_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, আকার(*request));
    } অন্যথা {
        যদি (cgraph->uid != 0) {
            যদি (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t প্রবেশ_আকার = 0;
        uint8_t * প্রবেশ = serialize_graph(device, cgraph, এই, &প্রবেশ_আকার);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_আকার(প্রবেশ_আকার).c_str());
        std::shared_ptr<uint8_t> প্রবেশ_ptr(প্রবেশ, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, প্রবেশ_ptr, প্রবেশ_আকার);
    }
}

স্থির গণনা ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    ফেরত GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

স্থির শূন্য ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

স্থির শূন্য ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    ggml_backend_tensor_set_2d(tensor, তথ্য, offset, size, n_copies, stride_tensor, stride_তথ্য);
    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ধ্রুবক ggml_tensor * tensor, শূন্য * তথ্য,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_তথ্য) {
    ggml_backend_tensor_get_2d(tensor, তথ্য, offset, size, n_copies, stride_tensor, stride_তথ্য);
    GGML_UNUSED(backend);
}

স্থির ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ফাঁকা,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ফাঁকা,
    /* .graph_plan_free         = */ ফাঁকা,
    /* .graph_plan_update       = */ ফাঁকা,
    /* .graph_plan_compute      = */ ফাঁকা,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ফাঁকা,
};

ggml_backend_বাফার_type_t ggml_backend_rpc_বাফার_type(ধ্রুবক অক্ষর * endpoint, uint32_t device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    স্থির std::unordered_map<std::string, ggml_backend_বাফার_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    যদি (it != buft_map.end()) {
        ফেরত it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_আকার(dispatcher, device);
    ggml_backend_rpc_বাফার_type_context * buft_ctx = নতুন ggml_backend_rpc_বাফার_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_বাফার_type_t buft = নতুন ggml_backend_বাফার_type {
        /* .iface   = */ ggml_backend_rpc_বাফার_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_আকার(alignment).c_str(), rpc_log_format_আকার(max_size).c_str());
    ফেরত buft;
}

ggml_backend_t ggml_backend_rpc_init(ধ্রুবক অক্ষর * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = নতুন ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = নতুন ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ফেরত backend;
}

যৌক্তিক ggml_backend_is_rpc(ggml_backend_t backend) {
    ফেরত backend != ফাঁকা && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

শূন্য ggml_backend_rpc_get_device_memory(ধ্রুবক অক্ষর * endpoint, uint32_t device, size_t * মুক্ত, size_t * যোগ) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_বার্তা_get_device_memory_req>();
    request->device = device;
    rpc_বার্তা_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, আকার(*request), &response, আকার(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_আকার(response.free_mem).c_str(), rpc_log_format_আকার(response.যোগ_mem).c_str());
    *মুক্ত = response.free_mem;
    *যোগ = response.যোগ_mem;
}

// RPC server-side implementation

শ্রেণী rpc_server {
সর্বজনীন:
    rpc_server(std::vector<ggml_backend_t> all_backends, ধ্রুবক অক্ষর * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    শূন্য hello(rpc_বার্তা_hello_rsp & response);
    যৌক্তিক alloc_বাফার(ধ্রুবক rpc_বার্তা_alloc_বাফার_req & request, rpc_বার্তা_alloc_বাফার_rsp & response);
    যৌক্তিক get_alignment(ধ্রুবক rpc_বার্তা_get_alignment_req & request, rpc_বার্তা_get_alignment_rsp & response);
    যৌক্তিক get_max_আকার(ধ্রুবক rpc_বার্তা_get_max_আকার_req & request, rpc_বার্তা_get_max_আকার_rsp & response);
    যৌক্তিক বাফার_get_base(ধ্রুবক rpc_বার্তা_বাফার_get_base_req & request, rpc_বার্তা_বাফার_get_base_rsp & response);
    যৌক্তিক free_বাফার(ধ্রুবক rpc_বার্তা_free_বাফার_req & request);
    যৌক্তিক বাফার_clear(ধ্রুবক rpc_বার্তা_বাফার_clear_req & request);
    যৌক্তিক memset_tensor(ধ্রুবক rpc_বার্তা_memset_tensor_req & request);
    যৌক্তিক set_tensor(ধ্রুবক std::vector<uint8_t> & প্রবেশ);
    যৌক্তিক set_tensor_2d(ধ্রুবক std::vector<uint8_t> & প্রবেশ);
    যৌক্তিক set_tensor_hash(ধ্রুবক rpc_বার্তা_set_tensor_hash_req & request, rpc_বার্তা_set_tensor_hash_rsp & response);
    যৌক্তিক get_tensor(ধ্রুবক rpc_বার্তা_get_tensor_req & request, std::vector<uint8_t> & response);
    যৌক্তিক get_tensor_2d(ধ্রুবক rpc_বার্তা_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    যৌক্তিক copy_tensor(ধ্রুবক rpc_বার্তা_copy_tensor_req & request, rpc_বার্তা_copy_tensor_rsp & response);
    যৌক্তিক graph_compute(ধ্রুবক std::vector<uint8_t> & প্রবেশ);
    যৌক্তিক graph_recompute(ধ্রুবক rpc_বার্তা_graph_recompute_req & request);
    যৌক্তিক comm_init(ধ্রুবক rpc_বার্তা_comm_init_req & request, rpc_বার্তা_comm_init_rsp & response);
    যৌক্তিক comm_allreduce(ধ্রুবক rpc_বার্তা_comm_allreduce_req & request);
    যৌক্তিক comm_free(ধ্রুবক rpc_বার্তা_comm_free_req & request);
    যৌক্তিক init_tensor(ধ্রুবক rpc_বার্তা_init_tensor_req & request);
    যৌক্তিক get_alloc_আকার(ধ্রুবক rpc_বার্তা_get_alloc_আকার_req & request, rpc_বার্তা_get_alloc_আকার_rsp & response);
    যৌক্তিক get_device_memory(ধ্রুবক rpc_বার্তা_get_device_memory_req & request, rpc_বার্তা_get_device_memory_rsp & response);

    কাঠামো stored_graph {
        std::vector<uint8_t>   বাফার;
        ggml_cgraph          * graph;
    };

ব্যক্তিগত:
    শূন্য sync_all_backends();
    যৌক্তিক get_cached_file(uint64_t hash, std::vector<uint8_t> & তথ্য);
    ggml_tensor * deserialize_tensor(কাঠামো ggml_context * ctx, ধ্রুবক rpc_tensor * tensor);
    ggml_tensor * create_গ্রন্থি(uint64_t id,
                              কাঠামো ggml_context * ctx,
                              ধ্রুবক std::unordered_map<uint64_t, ধ্রুবক rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, কাঠামো ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    কাঠামো comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_বাফার_ptr scratch;
        size_t                  scratch_আকার = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ধ্রুবক অক্ষর * cache_dir;
    std::unordered_set<ggml_backend_বাফার_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

শূন্য rpc_server::hello(rpc_বার্তা_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

যৌক্তিক rpc_server::get_alloc_আকার(ধ্রুবক rpc_বার্তা_get_alloc_আকার_req & request, rpc_বার্তা_get_alloc_আকার_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ফেরত মিথ্যা;
    }
    ggml_backend_বাফার_type_t buft;
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ফেরত মিথ্যা;
    }
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        যদি (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (শূন্য*)tensor->বাফার, tensor->তথ্য);
    যদি (tensor->বাফার == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_বাফার_type(backends[dev_id]);
    } অন্যথা {
        buft = tensor->বাফার->buft;
    }

    response.alloc_আকার = ggml_backend_buft_get_alloc_আকার(buft, tensor);

    ফেরত সত্য;
}

যৌক্তিক rpc_server::alloc_বাফার(ধ্রুবক rpc_বার্তা_alloc_বাফার_req & request, rpc_বার্তা_alloc_বাফার_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ফেরত মিথ্যা;
    }
    ggml_backend_বাফার_type_t buft = ggml_backend_get_default_বাফার_type(backends[dev_id]);
    ggml_backend_বাফার_t বাফার = ggml_backend_buft_alloc_বাফার(buft, request.size);
    response.remote_ptr = 0;
    response.remote_আকার = 0;
    যদি (বাফার != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(বাফার);
        response.remote_আকার = বাফার->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_আকার(request.size).c_str(), response.remote_ptr,
                rpc_log_format_আকার(response.remote_আকার).c_str());
        buffers.insert(বাফার);
    } অন্যথা {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_আকার(request.size).c_str());
    }
    ফেরত সত্য;
}

যৌক্তিক rpc_server::get_alignment(ধ্রুবক rpc_বার্তা_get_alignment_req & request, rpc_বার্তা_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ফেরত মিথ্যা;
    }
    ggml_backend_বাফার_type_t buft = ggml_backend_get_default_বাফার_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_আকার(alignment).c_str());
    response.alignment = alignment;
    ফেরত সত্য;
}

যৌক্তিক rpc_server::get_max_আকার(ধ্রুবক rpc_বার্তা_get_max_আকার_req & request, rpc_বার্তা_get_max_আকার_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ফেরত মিথ্যা;
    }
    ggml_backend_বাফার_type_t buft = ggml_backend_get_default_বাফার_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_আকার(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_আকার(max_size).c_str());
    response.max_size = max_size;
    ফেরত সত্য;
}

যৌক্তিক rpc_server::বাফার_get_base(ধ্রুবক rpc_বার্তা_বাফার_get_base_req & request, rpc_বার্তা_বাফার_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_বাফার_t বাফার = reinterpret_cast<ggml_backend_বাফার_t>(request.remote_ptr);
    যদি (buffers.find(বাফার) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ফেরত মিথ্যা;
    }
    শূন্য * base = ggml_backend_বাফার_get_base(বাফার);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::free_বাফার(ধ্রুবক rpc_বার্তা_free_বাফার_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_বাফার_t বাফার = reinterpret_cast<ggml_backend_বাফার_t>(request.remote_ptr);
    যদি (buffers.find(বাফার) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ফেরত মিথ্যা;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    জন্য (auto & sgs : stored_graphs) {
        জন্য (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_বাফার_free(বাফার);
    buffers.erase(বাফার);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::বাফার_clear(ধ্রুবক rpc_বার্তা_বাফার_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_বাফার_t বাফার = reinterpret_cast<ggml_backend_বাফার_t>(request.remote_ptr);
    যদি (buffers.find(বাফার) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ফেরত মিথ্যা;
    }
    ggml_backend_বাফার_clear(বাফার, request.value);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::memset_tensor(ধ্রুবক rpc_বার্তা_memset_tensor_req & request) {
    sync_all_backends();
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }

    ধ্রুবক uint64_t tensor_আকার = ggml_nbytes(tensor);
    যদি (request.offset > tensor_আকার || request.size > tensor_আকার - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_আকার);
        ফেরত মিথ্যা;
    }

    ধ্রুবক uint64_t বাফার_start = (uint64_t) ggml_backend_বাফার_get_base(tensor->বাফার);
    ধ্রুবক uint64_t বাফার_আকার = ggml_backend_বাফার_get_আকার(tensor->বাফার);
    যদি (request.tensor.তথ্য < বাফার_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ফেরত মিথ্যা;
    }
    ধ্রুবক uint64_t তথ্য_offset = request.tensor.তথ্য - বাফার_start;
    যদি (তথ্য_offset > বাফার_আকার ||
        request.offset > বাফার_আকার - তথ্য_offset ||
        request.size > বাফার_আকার - তথ্য_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ফেরত মিথ্যা;
    }
    যদি (tensor->বাফার->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ফেরত মিথ্যা;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য *) tensor->বাফার, tensor->তথ্য, request.offset,
            rpc_log_format_আকার(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    ফেরত সত্য;
}

ggml_tensor * rpc_server::deserialize_tensor(কাঠামো ggml_context * ctx, ধ্রুবক rpc_tensor * tensor) {
    // Validate tensor type before using it
    যদি (tensor->type >= GGML_TYPE_গুনতি) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ফেরত nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    যদি (ggml_blck_আকার((গণনা ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ফেরত nullptr;
    }

    ggml_tensor * ফলাফল = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    যদি (ফলাফল == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ফেরত nullptr;
    }

    জন্য (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ফলাফল->nb[i] = tensor->nb[i];
    }
    ফলাফল->বাফার = reinterpret_cast<ggml_backend_বাফার_t>(tensor->বাফার);
    যদি (ফলাফল->বাফার && buffers.find(ফলাফল->বাফার) == buffers.end()) {
        ফলাফল->বাফার = nullptr;
    }

    যদি (ফলাফল->বাফার && ggml_nelements(ফলাফল) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_আকার = (uint64_t) ggml_nbytes(ফলাফল);
        uint64_t বাফার_start = (uint64_t) ggml_backend_বাফার_get_base(ফলাফল->বাফার);
        uint64_t বাফার_আকার = (uint64_t) ggml_backend_বাফার_get_আকার(ফলাফল->বাফার);
        যদি (tensor->তথ্য + tensor_আকার < tensor->তথ্য ||
            tensor->তথ্য < বাফার_start || tensor->তথ্য + tensor_আকার > বাফার_start + বাফার_আকার) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(ফলাফল->type),
                      ফলাফল->ne[0], ফলাফল->ne[1], ফলাফল->ne[2], ফলাফল->ne[3],
                      tensor->তথ্য, tensor->তথ্য + tensor_আকার, বাফার_start, বাফার_start + বাফার_আকার);
            ফেরত nullptr;
        }
    }

    ফলাফল->op = (ggml_op) tensor->op;
    জন্য (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / আকার(int32_t); i++) {
        ফলাফল->op_params[i] = tensor->op_params[i];
    }
    ফলাফল->flags = tensor->flags;
    ফলাফল->তথ্য = reinterpret_cast<শূন্য *>(tensor->তথ্য);
    ggml_set_name(ফলাফল, tensor->name);
    ফেরত ফলাফল;
}


যৌক্তিক rpc_server::set_tensor(ধ্রুবক std::vector<uint8_t> & প্রবেশ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    ধ্রুবক size_t header_আকার = আকার(rpc_tensor) + আকার(cache_flag) + আকার(offset);
    যদি (প্রবেশ.size() < header_আকার) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক rpc_tensor * in_tensor = (ধ্রুবক rpc_tensor *)প্রবেশ.তথ্য();
    memcpy(&cache_flag, প্রবেশ.তথ্য() + আকার(rpc_tensor), আকার(cache_flag));
    memcpy(&offset,     প্রবেশ.তথ্য() + আকার(rpc_tensor) + আকার(cache_flag), আকার(offset));
    ধ্রুবক size_t size = প্রবেশ.size() - header_আকার;

    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    যদি (tensor == nullptr || tensor->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->বাফার, tensor->তথ্য, offset,
            rpc_log_format_আকার(size).c_str());

    // sanitize tensor->data
    {
        ধ্রুবক size_t p0 = (size_t) ggml_backend_বাফার_get_base(tensor->বাফার);
        ধ্রুবক size_t p1 = p0 + ggml_backend_বাফার_get_আকার(tensor->বাফার);

        যদি (in_tensor->তথ্য + offset < p0 || in_tensor->তথ্য + offset >= p1 || size > (p1 - in_tensor->তথ্য - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->তথ্য, offset, size, p0, p1);
            ফেরত মিথ্যা;
        }
    }

    ধ্রুবক শূন্য * তথ্য = প্রবেশ.তথ্য() + header_আকার;
    যদি (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ধ্রুবক uint8_t*)তথ্য, size);
        অক্ষর hash_str[17];
        snprintf(hash_str, আকার(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.লিখ((ধ্রুবক অক্ষর *)তথ্য, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, তথ্য, offset, size);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::set_tensor_2d(ধ্রুবক std::vector<uint8_t> & প্রবেশ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    যদি (প্রবেশ.size() < আকার(rpc_tensor) + 4*আকার(uint64_t)) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক rpc_tensor * in_tensor = (ধ্রুবক rpc_tensor *)প্রবেশ.তথ্য();
    uint64_t header[4];
    memcpy(header, প্রবেশ.তথ্য() + আকার(rpc_tensor), আকার(header));
    ধ্রুবক uint64_t offset   = header[0];
    ধ্রুবক uint64_t size     = header[1];
    ধ্রুবক uint64_t n_copies = header[2];
    ধ্রুবক uint64_t stride   = header[3];

    ধ্রুবক uint64_t তথ্য_আকার = প্রবেশ.size() - আকার(rpc_tensor) - 4*আকার(uint64_t);
    যদি (n_copies == 0 || size == 0 || size > তথ্য_আকার / n_copies || size * n_copies != তথ্য_আকার) {
        ফেরত মিথ্যা;
    }

    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    যদি (tensor == nullptr || tensor->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->বাফার, tensor->তথ্য, offset,
            rpc_log_format_আকার(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        যদি (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ফেরত মিথ্যা;
        }
        ধ্রুবক uint64_t span = (n_copies - 1)*stride + size;
        ধ্রুবক uint64_t p0 = (uint64_t) ggml_backend_বাফার_get_base(tensor->বাফার);
        ধ্রুবক uint64_t p1 = p0 + ggml_backend_বাফার_get_আকার(tensor->বাফার);

        যদি (in_tensor->তথ্য < p0 || in_tensor->তথ্য > p1 || offset > p1 - in_tensor->তথ্য || span > p1 - in_tensor->তথ্য - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->তথ্য, offset, span, p0, p1);
            ফেরত মিথ্যা;
        }
        যদি (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            ফেরত মিথ্যা;
        }
    }

    ধ্রুবক শূন্য * তথ্য = প্রবেশ.তথ্য() + আকার(rpc_tensor) + 4*আকার(uint64_t);
    ggml_backend_tensor_set_2d(tensor, তথ্য, offset, size, n_copies, stride, size);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & তথ্য) {
    যদি (!cache_dir) {
        ফেরত মিথ্যা;
    }
    অক্ষর hash_str[17];
    snprintf(hash_str, আকার(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    যদি (!fs::exists(cache_file, ec)) {
        ফেরত মিথ্যা;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    তথ্য.resize(size);
    ifs.পঠ((অক্ষর *)তথ্য.তথ্য(), size);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::set_tensor_hash(ধ্রুবক rpc_বার্তা_set_tensor_hash_req & request, rpc_বার্তা_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }

    std::vector<uint8_t> cached_file;
    যদি (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.ফলাফল = 0;
        ফেরত সত্য;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->বাফার, tensor->তথ্য, request.offset,
            rpc_log_format_আকার(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ধ্রুবক size_t p0 = (size_t) ggml_backend_বাফার_get_base(tensor->বাফার);
        ধ্রুবক size_t p1 = p0 + ggml_backend_বাফার_get_আকার(tensor->বাফার);

        যদি (request.tensor.তথ্য + request.offset < p0
         || request.tensor.তথ্য + request.offset >= p1
         || size > (p1 - request.tensor.তথ্য - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.তথ্য, request.offset, size, request.hash, p0, p1);
            ফেরত মিথ্যা;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.তথ্য(), request.offset, size);
    response.ফলাফল = 1;
    ফেরত সত্য;
}

যৌক্তিক rpc_server::init_tensor(ধ্রুবক rpc_বার্তা_init_tensor_req & request) {
    sync_all_backends();
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ফেরত মিথ্যা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (শূন্য*)tensor->বাফার, tensor->তথ্য);
    // Call the backend's buffer_init_tensor function
    ggml_backend_বাফার_t বাফার = tensor->বাফার;
    যদি (বাফার && বাফার->iface.init_tensor) {
        বাফার->iface.init_tensor(বাফার, tensor);
    } অন্যথা {
        যদি (!বাফার) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    যদি (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ফেরত মিথ্যা;
    }

    ফেরত সত্য;
}

যৌক্তিক rpc_server::get_tensor(ধ্রুবক rpc_বার্তা_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->বাফার, tensor->তথ্য, request.offset,
             rpc_log_format_আকার(request.size).c_str());

    // sanitize tensor->data
    {
        ধ্রুবক size_t p0 = (size_t) ggml_backend_বাফার_get_base(tensor->বাফার);
        ধ্রুবক size_t p1 = p0 + ggml_backend_বাফার_get_আকার(tensor->বাফার);

        যদি (request.tensor.তথ্য + request.offset < p0 ||
            request.tensor.তথ্য + request.offset >= p1 ||
            request.size > (p1 - request.tensor.তথ্য - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.তথ্য, request.offset, request.size, p0, p1);
                ফেরত মিথ্যা;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.তথ্য(), request.offset, request.size);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::get_tensor_2d(ধ্রুবক rpc_বার্তা_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->বাফার, tensor->তথ্য, request.offset,
            rpc_log_format_আকার(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        যদি (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ফেরত মিথ্যা;
        }
        যদি (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ফেরত মিথ্যা;
        }
        ধ্রুবক uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        ধ্রুবক uint64_t p0 = (uint64_t) ggml_backend_বাফার_get_base(tensor->বাফার);
        ধ্রুবক uint64_t p1 = p0 + ggml_backend_বাফার_get_আকার(tensor->বাফার);

        যদি (request.tensor.তথ্য < p0 || request.tensor.তথ্য > p1 || request.offset > p1 - request.tensor.তথ্য ||
                span > p1 - request.tensor.তথ্য - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.তথ্য, request.offset, span, p0, p1);
            ফেরত মিথ্যা;
        }
        যদি (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            ফেরত মিথ্যা;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.তথ্য(), request.offset, request.size, request.n_copies, request.stride, request.size);
    ফেরত সত্য;
}

যৌক্তিক rpc_server::copy_tensor(ধ্রুবক rpc_বার্তা_copy_tensor_req & request, rpc_বার্তা_copy_tensor_rsp & response) {
    sync_all_backends();
    কাঠামো ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    যদি (src == nullptr || dst == nullptr || src->বাফার == nullptr || dst->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ফেরত মিথ্যা;
    }

    uint64_t src_আকার   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_তথ্য   = (uint64_t) dst->তথ্য;
    uint64_t dst_base   = (uint64_t) ggml_backend_বাফার_get_base(dst->বাফার);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_বাফার_get_আকার(dst->বাফার);

    যদি (dst_তথ্য + src_আকার > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_তথ্য,
                         dst_তথ্য + src_আকার,
                         dst_base,
                         dst_base + dst_buf_sz);
        ফেরত মিথ্যা;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.ফলাফল = ggml_backend_বাফার_copy_tensor(src, dst);
    ফেরত সত্য;
}

ggml_tensor * rpc_server::create_গ্রন্থি(uint64_t id,
                                      কাঠামো ggml_context * ctx,
                                      ধ্রুবক std::unordered_map<uint64_t, ধ্রুবক rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, কাঠামো ggml_tensor*> & tensor_map) {
    যদি (tensor_map.find(id) != tensor_map.end()) {
        ফেরত tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    যদি (it_ptr == tensor_ptrs.end()) {
        ফেরত nullptr;
    }
    ধ্রুবক rpc_tensor * tensor = it_ptr->second;

    কাঠামো ggml_tensor * ফলাফল = deserialize_tensor(ctx, tensor);
    যদি (ফলাফল == nullptr) {
        ফেরত nullptr;
    }
    যদি (ফলাফল->বাফার == nullptr && ফলাফল->তথ্য != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ফেরত nullptr;
    }
    tensor_map[id] = ফলাফল;
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        যদি (tensor->src[i] == 0) {
            ফলাফল->src[i] = nullptr;
        } অন্যথা {
            ফলাফল->src[i] = create_গ্রন্থি(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            যদি (ফলাফল->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ফেরত nullptr;
            }
        }
    }

    // Handle view_src similarly
    যদি (tensor->view_src == 0) {
        ফলাফল->view_src = nullptr;
    } অন্যথা {
        ফলাফল->view_src = create_গ্রন্থি(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        যদি (ফলাফল->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ফেরত nullptr;
        }
    }
    ফলাফল->view_offs = tensor->view_offs;
    ফেরত ফলাফল;
}

যৌক্তিক rpc_server::graph_compute(ধ্রুবক std::vector<uint8_t> & প্রবেশ) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    যদি (প্রবেশ.size() < 2*আকার(uint32_t) + আকার(uint64_t)) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক uint8_t * src = প্রবেশ.তথ্য();
    uint32_t device;
    memcpy(&device, src, আকার(device));
    src += আকার(device);
    যদি (device >= backends.size()) {
        ফেরত মিথ্যা;
    }
    uint64_t uid;
    memcpy(&uid, src, আকার(uid));
    src += আকার(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, আকার(n_nodes));
    src += আকার(n_nodes);
    যদি (প্রবেশ.size() < 2*আকার(uint32_t) + আকার(uint64_t) + n_nodes*আকার(uint64_t) + আকার(uint32_t)) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক uint64_t * nodes = (ধ্রুবক uint64_t *)src;
    src += n_nodes*আকার(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, আকার(n_tensors));
    src += আকার(n_tensors);
    যদি (প্রবেশ.size() < 2*আকার(uint32_t) + আকার(uint64_t) + n_nodes*আকার(uint64_t) + আকার(uint32_t) + n_tensors*আকার(rpc_tensor)) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক rpc_tensor * tensors = (ধ্রুবক rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    যদি (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_আকার = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, মিথ্যা);
    যদি (sg.বাফার.size() < buf_আকার) {
        sg.বাফার.resize(buf_আকার);
    }
    কাঠামো ggml_init_params params = {
        /*.mem_size   =*/ buf_আকার,
        /*.mem_buffer =*/ sg.বাফার.তথ্য(),
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    কাঠামো ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, মিথ্যা);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ধ্রুবক rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    জন্য (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ধ্রুবক auto t_deserialize = std::chrono::steady_clock::now();
    জন্য (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], আকার(id));
        graph->nodes[i] = create_গ্রন্থি(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        যদি (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ফেরত মিথ্যা;
        }
        যদি (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ধ্রুবক size_t hash_pos = ggml_hash_ঢোকাও(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_গুনতি;
        }
    }
    ধ্রুবক std::chrono::duration<দ্বিগুণ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ফেরত সত্য;
}

যৌক্তিক rpc_server::graph_recompute(ধ্রুবক rpc_বার্তা_graph_recompute_req & request) {
    uint32_t device = request.device;
    যদি (device >= backends.size()) {
        ফেরত মিথ্যা;
    }
    auto it = stored_graphs[device].find(request.uid);
    যদি (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ফেরত মিথ্যা;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ফেরত সত্য;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
শূন্য rpc_server::sync_all_backends() {
    জন্য (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
যৌক্তিক rpc_server::comm_init(ধ্রুবক rpc_বার্তা_comm_init_req & request, rpc_বার্তা_comm_init_rsp & response) {
    response.ok = 0;
    যদি (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ফেরত সত্য;
    }
    comm_state & state = comm_states[request.device];
    যদি (state.peer != nullptr) {
        response.ok = 1;
        ফেরত সত্য;
    }
    uint8_t local_caps[RPC_CONN_CAPS_আকার] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_আকার] = {};
    যদি (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        যদি (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            ফেরত সত্য;
        }
        state.peer = srv->accept();
        যদি (state.peer == nullptr) {
            ফেরত সত্য;
        }
        যদি (!state.peer->recv_তথ্য(remote_caps, আকার(remote_caps))) {
            state.peer = nullptr;
            ফেরত সত্য;
        }
        state.peer->get_caps(local_caps);
        যদি (!state.peer->send_তথ্য(local_caps, আকার(local_caps))) {
            state.peer = nullptr;
            ফেরত সত্য;
        }
        state.peer->update_caps(remote_caps);
    } অন্যথা {
        ধ্রুবক std::string host(request.host, strnlen(request.host, আকার(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        জন্য (পূর্ণসংখ্যা i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            যদি (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        যদি (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            ফেরত সত্য;
        }
        state.peer->get_caps(local_caps);
        যদি (!state.peer->send_তথ্য(local_caps, আকার(local_caps)) ||
            !state.peer->recv_তথ্য(remote_caps, আকার(remote_caps))) {
            state.peer = nullptr;
            ফেরত সত্য;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ফেরত সত্য;
}

যৌক্তিক rpc_server::comm_allreduce(ধ্রুবক rpc_বার্তা_comm_allreduce_req & request) {
    যদি (request.device >= backends.size()) {
        ফেরত মিথ্যা;
    }
    comm_state & state = comm_states[request.device];
    যদি (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ফেরত মিথ্যা;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_আকার = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, মিথ্যা);
    কাঠামো ggml_init_params params = {
        /*.mem_size   =*/ ctx_আকার,
        /*.mem_buffer =*/ ফাঁকা,
        /*.no_alloc   =*/ সত্য,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    যদি (t_dst == nullptr || t_dst->বাফার == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ফেরত মিথ্যা;
    }
    ধ্রুবক size_t  nbytes = ggml_nbytes(t_dst);
    ধ্রুবক int64_t ne     = ggml_nelements(t_dst);
    যদি (nbytes == 0) {
        ফেরত সত্য;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ধ্রুবক যৌক্তিক   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ধ্রুবক size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    ধ্রুবক size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    যদি (state.scratch_আকার < need) {
        state.scratch.reset(ggml_backend_alloc_বাফার(backend, need));
        state.scratch_আকার = need;
    }
    অক্ষর * scratch_base = (অক্ষর *) ggml_backend_বাফার_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->বাফার = state.scratch.get();
        t->তথ্য   = scratch_base + offset;
        ফেরত t;
    };
    auto new_cpy_গ্রন্থি = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->বাফার = dst->বাফার;
        t->তথ্য   = dst->তথ্য;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ফেরত t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, মিথ্যা);
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
    যদি (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_গ্রন্থি(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.তথ্য(), 0, wire_bytes);
    } অন্যথা {
        ggml_backend_tensor_get(t_dst, state.send_buf.তথ্য(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    যদি (state.rank == 0) {
        যদি (!state.peer->send_তথ্য(state.send_buf.তথ্য(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_তথ্য(state.recv_buf.তথ্য(), wire_bytes)) {
            ফেরত মিথ্যা;
        }
    } অন্যথা {
        যদি (!state.peer->recv_তথ্য(state.recv_buf.তথ্য(), wire_bytes) ||
            !state.peer->send_তথ্য(state.send_buf.তথ্য(), wire_bytes) || !state.peer->flush()) {
            ফেরত মিথ্যা;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    যদি (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.তথ্য(), 0, wire_bytes);
        t_cast = new_cpy_গ্রন্থি(t_wire_recv, t_peer);
    } অন্যথা {
        ggml_backend_tensor_set(t_peer, state.recv_buf.তথ্য(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->বাফার = t_dst->বাফার;
    t_red->তথ্য   = t_dst->তথ্য;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    যদি (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } অন্যথা {
        compute_nodes(t_red, nullptr);
    }
    ফেরত সত্য;
}

যৌক্তিক rpc_server::comm_free(ধ্রুবক rpc_বার্তা_comm_free_req & request) {
    যদি (request.device >= backends.size()) {
        ফেরত মিথ্যা;
    }
    comm_states[request.device] = comm_state();
    ফেরত সত্য;
}

যৌক্তিক rpc_server::get_device_memory(ধ্রুবক rpc_বার্তা_get_device_memory_req & request, rpc_বার্তা_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ফেরত মিথ্যা;
    }
    size_t মুক্ত, যোগ;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &মুক্ত, &যোগ);
    response.free_mem = মুক্ত;
    response.যোগ_mem = যোগ;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_আকার(response.free_mem).c_str(), rpc_log_format_আকার(response.যোগ_mem).c_str());
    ফেরত সত্য;
}

rpc_server::~rpc_server() {
    জন্য (auto বাফার : buffers) {
        ggml_backend_বাফার_free(বাফার);
    }
}

স্থির শূন্য rpc_serve_client(ধ্রুবক std::vector<ggml_backend_t> & backends, ধ্রুবক অক্ষর * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    যদি (!sock->recv_তথ্য(&cmd, 1)) {
        ফেরত;
    }
    যদি (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ফেরত;
    }

    // Read input_size and validate protocol version
    uint64_t hello_প্রবেশ_আকার;
    যদি (!sock->recv_তথ্য(&hello_প্রবেশ_আকার, আকার(hello_প্রবেশ_আকার))) {
        ফেরত;
    }

    যদি (hello_প্রবেশ_আকার != আকার(rpc_বার্তা_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_প্রবেশ_আকার, আকার(rpc_বার্তা_hello_req), RPC_PROTO_MAJOR_VERSION);
        ফেরত;
    }

    rpc_বার্তা_hello_req req = {};
    যদি (!sock->recv_তথ্য(&req, আকার(req))) {
        ফেরত;
    }

    rpc_বার্তা_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    যদি (!send_বার্তা(sock, &rsp, আকার(rsp))) {
        ফেরত;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    যখন (সত্য) {
        যদি (!sock->recv_তথ্য(&cmd, 1)) {
            বিরতি;
        }
        যদি (cmd >= RPC_CMD_গুনতি) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            বিরতি;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((গণনা rpc_cmd)cmd));
        নির্বাচন (cmd) {
            ক্ষেত্র RPC_CMD_HELLO: {
                // HELLO command is handled above
                ফেরত;
            }
            ক্ষেত্র RPC_CMD_DEVICE_গুনতি: {
                যদি (!recv_বার্তা(sock, nullptr, 0)) {
                    ফেরত;
                }
                rpc_বার্তা_device_গুনতি_rsp response;
                response.device_গুনতি = backends.size();
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_ALLOC_বাফার: {
                rpc_বার্তা_alloc_বাফার_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_alloc_বাফার_rsp response;
                যদি (!server.alloc_বাফার(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GET_ALLOC_আকার: {
                rpc_বার্তা_get_alloc_আকার_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_get_alloc_আকার_rsp response;
                যদি (!server.get_alloc_আকার(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GET_ALIGNMENT: {
                rpc_বার্তা_get_alignment_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_get_alignment_rsp response;
                যদি (!server.get_alignment(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GET_MAX_আকার: {
                rpc_বার্তা_get_max_আকার_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_get_max_আকার_rsp response;
                যদি (!server.get_max_আকার(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_বাফার_GET_BASE: {
                rpc_বার্তা_বাফার_get_base_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_বাফার_get_base_rsp response;
                যদি (!server.বাফার_get_base(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_FREE_বাফার: {
                rpc_বার্তা_free_বাফার_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                যদি (!server.free_বাফার(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_বাফার_CLEAR: {
                rpc_বার্তা_বাফার_clear_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                যদি (!server.বাফার_clear(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_MEMSET_TENSOR: {
                rpc_বার্তা_memset_tensor_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                যদি (!server.memset_tensor(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> প্রবেশ;
                যদি (!recv_বার্তা(sock, প্রবেশ)) {
                    ফেরত;
                }
                যদি (!server.set_tensor(প্রবেশ)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> প্রবেশ;
                যদি (!recv_বার্তা(sock, প্রবেশ)) {
                    ফেরত;
                }
                যদি (!server.set_tensor_2d(প্রবেশ)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GET_TENSOR_2D: {
                rpc_বার্তা_get_tensor_2d_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                std::vector<uint8_t> response;
                যদি (!server.get_tensor_2d(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, response.তথ্য(), response.size())) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_SET_TENSOR_HASH: {
                rpc_বার্তা_set_tensor_hash_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_set_tensor_hash_rsp response;
                যদি (!server.set_tensor_hash(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_INIT_TENSOR: {
                rpc_বার্তা_init_tensor_req request;
                যদি (!recv_বার্তা(sock, &request,আকার(request))) {
                    ফেরত;
                }
                যদি (!server.init_tensor(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GET_TENSOR: {
                rpc_বার্তা_get_tensor_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                std::vector<uint8_t> response;
                যদি (!server.get_tensor(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, response.তথ্য(), response.size())) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_COPY_TENSOR: {
                rpc_বার্তা_copy_tensor_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_copy_tensor_rsp response;
                যদি (!server.copy_tensor(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> প্রবেশ;
                যদি (!recv_বার্তা(sock, প্রবেশ)) {
                    ফেরত;
                }
                যদি (!server.graph_compute(প্রবেশ)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_বার্তা_graph_recompute_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                যদি (!server.graph_recompute(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_COMM_INIT: {
                rpc_বার্তা_comm_init_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_comm_init_rsp response;
                যদি (!server.comm_init(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_COMM_ALLREDUCE: {
                rpc_বার্তা_comm_allreduce_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                যদি (!server.comm_allreduce(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_COMM_FREE: {
                rpc_বার্তা_comm_free_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                যদি (!server.comm_free(request)) {
                    ফেরত;
                }
                বিরতি;
            }
            ক্ষেত্র RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_বার্তা_get_device_memory_req request;
                যদি (!recv_বার্তা(sock, &request, আকার(request))) {
                    ফেরত;
                }
                rpc_বার্তা_get_device_memory_rsp response;
                যদি (!server.get_device_memory(request, response)) {
                    ফেরত;
                }
                যদি (!send_বার্তা(sock, &response, আকার(response))) {
                    ফেরত;
                }
                বিরতি;
            }
            পূর্বনির্ধারিত: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((গণনা rpc_cmd)cmd));
                ফেরত;
            }
        }
    }
}

শূন্য ggml_backend_rpc_start_server(ধ্রুবক অক্ষর * endpoint, ধ্রুবক অক্ষর * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    যদি (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ফেরত;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    জন্য (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t মুক্ত, যোগ;
        ggml_backend_dev_memory(dev, &মুক্ত, &যোগ);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 যোগ / 1024 / 1024, মুক্ত / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        যদি (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ফেরত;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        যদি (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            যদি (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    পূর্ণসংখ্যা port;
    যদি (!parse_endpoint(endpoint, host, port)) {
        ফেরত;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#অন্যথা
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    যদি (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ফেরত;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    যদি (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        ফেরত;
    }
    যখন (সত্য) {
        auto client_socket = server_socket->accept();
        যদি (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            ফেরত;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    জন্য (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

স্থির ধ্রুবক অক্ষর * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ফেরত ctx->name.c_str();
}

স্থির ধ্রুবক অক্ষর * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ফেরত ctx->description.c_str();
}

স্থির শূন্য ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, মুক্ত, যোগ);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ফেরত GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, কাঠামো ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);
    props->caps = {
        /* .async                 = */ সত্য,
        /* .host_buffer           = */ মিথ্যা,
        /* .buffer_from_host_ptr  = */ মিথ্যা,
        /* .events                = */ সত্য,
        /* .mmap_support          = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ফেরত ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_rpc_device_get_বাফার_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ফেরত ggml_backend_rpc_বাফার_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

স্থির যৌক্তিক ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক কাঠামো ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    যদি (!buft || buft->iface.get_name != ggml_backend_rpc_বাফার_type_name) {
        ফেরত মিথ্যা;
    }
    ggml_backend_rpc_বাফার_type_context * buft_ctx = (ggml_backend_rpc_বাফার_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ফেরত buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

স্থির ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    ফেরত dispatcher->event_new(dev);
}

স্থির শূন্য ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

স্থির শূন্য ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

স্থির ধ্রুবক কাঠামো ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ফাঁকা,
    /* .buffer_from_host_ptr = */ ফাঁকা,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ফাঁকা,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

কাঠামো ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

স্থির ধ্রুবক অক্ষর * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ফেরত ctx ? ctx->name.c_str() : "RPC";
}

স্থির size_t ggml_backend_rpc_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ফেরত ctx ? ctx->devices.size() : 0;
}

স্থির ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t সূচক) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    যদি (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } অন্যথা {
        GGML_ASSERT(সূচক < ctx->devices.size());
        ফেরত ctx->devices[সূচক];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
কাঠামো ggml_backend_rpc_comm_context {
    কাঠামো rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

স্থির শূন্য ggml_backend_rpc_comm_free(শূন্য * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    যদি (comm_ctx == nullptr) {
        ফেরত;
    }
    জন্য (ধ্রুবক auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_বার্তা_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, আকার(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

স্থির শূন্য * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    যদি (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        ফেরত nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    জন্য (size_t i = 0; i < n_backends; i++) {
        যদি (!ggml_backend_is_rpc(backends[i])) {
            ফেরত nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        জন্য (ধ্রুবক auto & rank : ranks) {
            যদি (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ফেরত nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    পূর্ণসংখ্যা port0;
    যদি (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ফেরত nullptr;
    }
    ধ্রুবক uint32_t comm_port = (uint32_t) port0 + 1000;
    যদি (host0.size() >= 64) {
        ফেরত nullptr;
    }

    জন্য (ধ্রুবক auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_বার্তা_comm_init_rsp> responses(n_backends);
    জন্য (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_বার্তা_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        যদি (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, আকার(*request), &responses[i], আকার(responses[i]));
    }
    জন্য (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    যৌক্তিক ok = সত্য;
    জন্য (size_t i = 0; i < n_backends; i++) {
        যদি (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = মিথ্যা;
        }
    }
    যদি (!ok) {
        জন্য (ধ্রুবক auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ফেরত nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ফেরত নতুন ggml_backend_rpc_comm_context{std::move(ranks)};
}

স্থির যৌক্তিক ggml_backend_rpc_comm_allreduce_tensor(শূন্য * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    যদি (comm_ctx == nullptr) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক size_t n_ranks = comm_ctx->ranks.size();
    ধ্রুবক int64_t ne = ggml_nelements(tensors[0]);
    যদি (ne == 0) {
        ফেরত সত্য;
    }
    জন্য (size_t i = 0; i < n_ranks; i++) {
        যদি (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->বাফার == nullptr || !ggml_backend_বাফার_is_rpc(tensors[i]->বাফার)) {
            ফেরত মিথ্যা;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        যদি ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ফেরত মিথ্যা;
        }
    }
    জন্য (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_বার্তা_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, আকার(*request));
    }
    ফেরত সত্য;
}

স্থির শূন্য * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    যদি (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ফেরত (শূন্য *)ggml_backend_rpc_add_server;
    }
    যদি (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ফেরত (শূন্য *)ggml_backend_rpc_start_server;
    }
    যদি (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ফেরত (শূন্য *)ggml_backend_rpc_comm_init;
    }
    যদি (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ফেরত (শূন্য *)ggml_backend_rpc_comm_free;
    }
    যদি (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ফেরত (শূন্য *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ফেরত ফাঁকা;

    GGML_UNUSED(reg);
}

স্থির ধ্রুবক কাঠামো ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(শূন্য) {
    স্থির কাঠামো ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ফাঁকা,
    };

    ফেরত &ggml_backend_rpc_reg;
}

স্থির uint32_t ggml_backend_rpc_get_device_গুনতি(ধ্রুবক অক্ষর * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_বার্তা_device_গুনতি_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_গুনতি, nullptr, 0, &response, আকার(response));
    ফেরত response.device_গুনতি;
}

স্থির ধ্রুবক ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_গুনতি,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(ধ্রুবক অক্ষর * endpoint) {
    স্থির std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    স্থির std::mutex mutex;
    স্থির uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    যদি (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ফেরত reg_map[endpoint];
    }
    uint32_t dev_গুনতি = ggml_backend_rpc_get_device_গুনতি(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_গুনতি);
    যদি (dev_গুনতি == 0) {
        ফেরত nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = নতুন ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    জন্য (uint32_t ind = 0; ind < dev_গুনতি; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = নতুন ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = নতুন ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = নতুন ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ফেরত reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
