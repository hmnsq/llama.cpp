#অন্তর্ভাৱ "ggml-rpc.h"
#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-cpp.h"
#অন্তর্ভাৱ "transport.h"
#অন্তর্ভাৱ "log.h"

#অন্তর্ভাৱ <array>
#অন্তর্ভাৱ <chrono>
#অন্তর্ভাৱ <cinttypes>
#অন্তর্ভাৱ <optional>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <thread>
#অন্তর্ভাৱ <vector>
#অন্তর্ভাৱ <queue>
#অন্তর্ভাৱ <condition_variable>
#অন্তর্ভাৱ <future>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <mutex>
#অন্তর্ভাৱ <unordered_map>
#অন্তর্ভাৱ <unordered_set>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <fstream>
#অন্তর্ভাৱ <filesystem>
#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <atomic>

নামাকাশ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) যদি (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(ঠেলো, 1)
// ggml_tensor is serialized into rpc_tensor
সংরচনা rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / পরিমাণ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    ৱর্ণ name[GGML_MAX_NAME];

    int32_t use_গুনতি;
};

static_assert(পরিমাণ(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
গণনা rpc_cmd {
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

স্থির ধ্রুৱ ৱর্ণ * rpc_cmd_name(গণনা rpc_cmd cmd) {
    ৱিকল্প (cmd) {
        পক্ষ RPC_CMD_ALLOC_BUFFER:      ঘূৰাই "ALLOC_BUFFER";
        পক্ষ RPC_CMD_GET_ALIGNMENT:     ঘূৰাই "GET_ALIGNMENT";
        পক্ষ RPC_CMD_GET_MAX_SIZE:      ঘূৰাই "GET_MAX_SIZE";
        পক্ষ RPC_CMD_BUFFER_GET_BASE:   ঘূৰাই "BUFFER_GET_BASE";
        পক্ষ RPC_CMD_FREE_BUFFER:       ঘূৰাই "FREE_BUFFER";
        পক্ষ RPC_CMD_BUFFER_CLEAR:      ঘূৰাই "BUFFER_CLEAR";
        পক্ষ RPC_CMD_SET_TENSOR:        ঘূৰাই "SET_TENSOR";
        পক্ষ RPC_CMD_SET_TENSOR_HASH:   ঘূৰাই "SET_TENSOR_HASH";
        পক্ষ RPC_CMD_GET_TENSOR:        ঘূৰাই "GET_TENSOR";
        পক্ষ RPC_CMD_COPY_TENSOR:       ঘূৰাই "COPY_TENSOR";
        পক্ষ RPC_CMD_GRAPH_COMPUTE:     ঘূৰাই "GRAPH_COMPUTE";
        পক্ষ RPC_CMD_GET_DEVICE_MEMORY: ঘূৰাই "GET_DEVICE_MEMORY";
        পক্ষ RPC_CMD_INIT_TENSOR:       ঘূৰাই "INIT_TENSOR";
        পক্ষ RPC_CMD_GET_ALLOC_SIZE:    ঘূৰাই "GET_ALLOC_SIZE";
        পক্ষ RPC_CMD_HELLO:             ঘূৰাই "HELLO";
        পক্ষ RPC_CMD_DEVICE_গুনতি:      ঘূৰাই "DEVICE_COUNT";
        পক্ষ RPC_CMD_GRAPH_RECOMPUTE:   ঘূৰাই "GRAPH_RECOMPUTE";
        পক্ষ RPC_CMD_MEMSET_TENSOR:     ঘূৰাই "MEMSET_TENSOR";
        পক্ষ RPC_CMD_SET_TENSOR_2D:     ঘূৰাই "SET_TENSOR_2D";
        পক্ষ RPC_CMD_GET_TENSOR_2D:     ঘূৰাই "GET_TENSOR_2D";
        পক্ষ RPC_CMD_COMM_INIT:         ঘূৰাই "COMM_INIT";
        পক্ষ RPC_CMD_COMM_ALLREDUCE:    ঘূৰাই "COMM_ALLREDUCE";
        পক্ষ RPC_CMD_COMM_FREE:         ঘূৰাই "COMM_FREE";
        পক্ষ RPC_CMD_NONE:              ঘূৰাই "NONE";
        স্ৱভাৱ:                        ঘূৰাই "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ধ্রুৱ size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ধ্রুৱ size_t GRAPH_CACHE_MAX = 1024;

সংরচনা rpc_বাৰ্তা_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

সংরচনা rpc_বাৰ্তা_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

সংরচনা rpc_বাৰ্তা_device_গুনতি_rsp {
    uint32_t device_গুনতি;
};

সংরচনা rpc_বাৰ্তা_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

সংরচনা rpc_বাৰ্তা_get_alloc_size_rsp {
    uint64_t alloc_size;
};

সংরচনা rpc_বাৰ্তা_init_tensor_req {
    rpc_tensor tensor;
};

সংরচনা rpc_বাৰ্তা_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

সংরচনা rpc_বাৰ্তা_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

সংরচনা rpc_বাৰ্তা_get_alignment_req {
    uint32_t device;
};

সংরচনা rpc_বাৰ্তা_get_alignment_rsp {
    uint64_t alignment;
};

সংরচনা rpc_বাৰ্তা_get_max_size_req {
    uint32_t device;
};

সংরচনা rpc_বাৰ্তা_get_max_size_rsp {
    uint64_t max_size;
};

সংরচনা rpc_বাৰ্তা_buffer_get_base_req {
    uint64_t remote_ptr;
};

সংরচনা rpc_বাৰ্তা_buffer_get_base_rsp {
    uint64_t base_ptr;
};

সংরচনা rpc_বাৰ্তা_free_buffer_req {
    uint64_t remote_ptr;
};

সংরচনা rpc_বাৰ্তা_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

সংরচনা rpc_বাৰ্তা_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

সংরচনা rpc_বাৰ্তা_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

সংরচনা rpc_বাৰ্তা_set_tensor_hash_rsp {
    uint8_t result;
};

সংরচনা rpc_বাৰ্তা_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

সংরচনা rpc_বাৰ্তা_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

সংরচনা rpc_বাৰ্তা_copy_tensor_rsp {
    uint8_t result;
};

সংরচনা rpc_বাৰ্তা_get_device_memory_req {
    uint32_t device;
};

সংরচনা rpc_বাৰ্তা_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

সংরচনা rpc_বাৰ্তা_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

সংরচনা rpc_বাৰ্তা_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

সংরচনা rpc_বাৰ্তা_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    ৱর্ণ     host[64];  // rank > 0: rank 0's host
};

সংরচনা rpc_বাৰ্তা_comm_init_rsp {
    uint8_t ok;
};

সংরচনা rpc_বাৰ্তা_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

সংরচনা rpc_বাৰ্তা_comm_free_req {
    uint32_t device;
};

#pragma pack(বেৰ_কৰো)

// RPC data structures

স্থির ggml_guid_t ggml_backend_rpc_guid() {
    স্থির ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ঘূৰাই &guid;
}

সংরচনা ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

সংরচনা ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

শ্ৰেণী rpc_dispatcher;
সংরচনা ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

সংরচনা ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    শূন্য                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
স্থির uint64_t fnv_hash(ধ্রুৱ uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ধ্রুৱ uint64_t fnv_মৌলিক = 0x100000001b3ULL;

    বাবে (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_মৌলিক;
    }
    ঘূৰাই hash;
}

স্থির তার্কিক send_বাৰ্তা(socket_ptr sock, ধ্রুৱ শূন্য * বাৰ্তা, size_t বাৰ্তা_size) {
    যদি (!sock->send_data(&বাৰ্তা_size, পরিমাণ(বাৰ্তা_size))) {
        ঘূৰাই মিছা;
    }
    যদি (!sock->send_data(বাৰ্তা, বাৰ্তা_size)) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই sock->flush();
}

স্থির তার্কিক recv_বাৰ্তা(socket_ptr sock, শূন্য * বাৰ্তা, size_t বাৰ্তা_size) {
    uint64_t size;
    যদি (!sock->recv_data(&size, পরিমাণ(size))) {
        ঘূৰাই মিছা;
    }
    যদি (size != বাৰ্তা_size) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই sock->recv_data(বাৰ্তা, বাৰ্তা_size);
}

স্থির তার্কিক recv_বাৰ্তা(socket_ptr sock, std::vector<uint8_t> & প্ৰবেশ) {
    uint64_t size;
    যদি (!sock->recv_data(&size, পরিমাণ(size))) {
        ঘূৰাই মিছা;
    }
    যত্ন {
        প্ৰবেশ.resize(size);
    } গৃহাণ (ধ্রুৱ std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ঘূৰাই মিছা;
    }
    ঘূৰাই sock->recv_data(প্ৰবেশ.data(), size);
}

স্থির তার্কিক parse_endpoint(ধ্রুৱ std::string & endpoint, std::string & host, পূর্ণাঙ্ক & port) {
    size_t pos = endpoint.find(':');
    যদি (pos == std::string::npos) {
        ঘূৰাই মিছা;
    }
    host = endpoint.substr(0, pos);
    যত্ন {
        port = std::stoi(endpoint.substr(pos + 1));
    } গৃহাণ (...) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই সঁচা;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
স্থির তার্কিক send_rpc_cmd(socket_ptr sock, গণনা rpc_cmd cmd, ধ্রুৱ শূন্য * প্ৰবেশ, size_t প্ৰবেশ_size) {
    uint8_t cmd_byte = cmd;
    যদি (!sock->send_data(&cmd_byte, পরিমাণ(cmd_byte))) {
        ঘূৰাই মিছা;
    }
    যদি (!sock->send_data(&প্ৰবেশ_size, পরিমাণ(প্ৰবেশ_size))) {
        ঘূৰাই মিছা;
    }
    যদি (!sock->send_data(প্ৰবেশ, প্ৰবেশ_size)) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
স্থির তার্কিক send_rpc_cmd(socket_ptr sock, গণনা rpc_cmd cmd, ধ্রুৱ শূন্য * প্ৰবেশ, size_t প্ৰবেশ_size, শূন্য * output, size_t output_size) {
    যদি (!send_rpc_cmd(sock, cmd, প্ৰবেশ, প্ৰবেশ_size)) {
        ঘূৰাই মিছা;
    }
    uint64_t out_size;
    যদি (!sock->recv_data(&out_size, পরিমাণ(out_size))) {
        ঘূৰাই মিছা;
    }
    যদি (out_size != output_size) {
        ঘূৰাই মিছা;
    }
    যদি (!sock->recv_data(output, output_size)) {
        ঘূৰাই মিছা;
    }
    ঘূৰাই সঁচা;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
স্থির constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

স্থির অন্তঃস্থ শূন্য rpc_cpu_relax() {
#যদি defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ চঞ্চল("yield" ::: "memory");
#অন্যথা
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
স্থির তার্কিক negotiate_hello(ধ্রুৱ std::shared_ptr<socket_t> & sock) {
    rpc_বাৰ্তা_hello_req request = {};
    rpc_বাৰ্তা_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    তার্কিক status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, পরিমাণ(request), &response, পরিমাণ(response));
    RPC_STATUS_ASSERT(status);

    যদি (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ঘূৰাই মিছা;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ঘূৰাই সঁচা;
}

template <typename T>
শ্ৰেণী message_queue {
সার্ৱজনিক:
    message_queue() {}

    তার্কিক ঠেলো(ধ্রুৱ T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        যদি (interrupted) {
            ঘূৰাই মিছা;
        }
        queue.ঠেলো(value);
        cvar.notify_all();
        ঘূৰাই সঁচা;
    }

    তার্কিক বেৰ_কৰো(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [এতত্] { ঘূৰাই !queue.empty() || interrupted; });
        যদি (interrupted) {
            ঘূৰাই মিছা;
        }
        *out = queue.front();
        queue.বেৰ_কৰো();
        ঘূৰাই সঁচা;
    }

    তার্কিক try_বেৰ_কৰো(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        যদি (interrupted || queue.empty()) {
            ঘূৰাই মিছা;
        }
        *out = queue.front();
        queue.বেৰ_কৰো();
        ঘূৰাই সঁচা;
    }

    শূন্য interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = সঁচা;
        lock.unlock();
        cvar.notify_all();
    }

গুপ্ত:
    তার্কিক interrupted = মিছা;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

শ্ৰেণী rpc_dispatcher {
সার্ৱজনিক:
    rpc_dispatcher() {
    }

    শূন্য send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size);
    শূন্য send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size, শূন্য * output, size_t output_size);
    শূন্য send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size);
    শূন্য send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size, শূন্য * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    শূন্য event_free(ggml_backend_event_t event);
    শূন্য event_synchronize(ggml_backend_event_t event);
    শূন্য event_record(ggml_backend_event_t event);
    শূন্য synchronize();
    শূন্য busy_spin_acquire();
    শূন্য busy_spin_release();
    শূন্য graph_compute(uint32_t device, ধ্রুৱ ggml_cgraph * cgraph);

    শূন্য start(ধ্রুৱ std::string & endpoint);
    শূন্য work();

    ~rpc_dispatcher();

গুপ্ত:
    সংরচনা rpc_বাৰ্তা {
        rpc_cmd                       cmd;
        std::shared_ptr<ধ্রুৱ শূন্য>   প্ৰবেশ;
        size_t                        প্ৰবেশ_size;
        শূন্য                        * output;
        size_t                        output_size;
        std::promise<শূন্য>            completion;
    };
    using rpc_বাৰ্তা_ptr   = std::shared_ptr<rpc_বাৰ্তা>;
    using rpc_বাৰ্তা_queue = message_queue<rpc_বাৰ্তা_ptr>;
    সংরচনা rpc_event {
        rpc_বাৰ্তা_ptr              বাৰ্তা;
        std::shared_future<শূন্য> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_বাৰ্তা_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

স্থির শূন্য rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

শূন্য rpc_dispatcher::send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size) {
    auto বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    বাৰ্তা->cmd = cmd;
    বাৰ্তা->প্ৰবেশ = প্ৰবেশ;
    বাৰ্তা->প্ৰবেশ_size = প্ৰবেশ_size;
    বাৰ্তা->output = nullptr;
    বাৰ্তা->output_size = 0;
    GGML_ASSERT(queue.ঠেলো(বাৰ্তা));
    auto future = বাৰ্তা->completion.get_future();
    future.wait();
}

শূন্য rpc_dispatcher::send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size) {
    auto বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    বাৰ্তা->cmd = cmd;
    বাৰ্তা->প্ৰবেশ = প্ৰবেশ;
    বাৰ্তা->প্ৰবেশ_size = প্ৰবেশ_size;
    বাৰ্তা->output = nullptr;
    বাৰ্তা->output_size = 0;
    GGML_ASSERT(queue.ঠেলো(বাৰ্তা));
}

শূন্য rpc_dispatcher::send(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size, শূন্য * output, size_t output_size) {
    auto বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    বাৰ্তা->cmd = cmd;
    বাৰ্তা->প্ৰবেশ = প্ৰবেশ;
    বাৰ্তা->প্ৰবেশ_size = প্ৰবেশ_size;
    বাৰ্তা->output = output;
    বাৰ্তা->output_size = output_size;
    GGML_ASSERT(queue.ঠেলো(বাৰ্তা));
    auto future = বাৰ্তা->completion.get_future();
    future.wait();
}

শূন্য rpc_dispatcher::send_async(গণনা rpc_cmd cmd, std::shared_ptr<ধ্রুৱ শূন্য> প্ৰবেশ, size_t প্ৰবেশ_size, শূন্য * output, size_t output_size) {
    auto বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    বাৰ্তা->cmd = cmd;
    বাৰ্তা->প্ৰবেশ = প্ৰবেশ;
    বাৰ্তা->প্ৰবেশ_size = প্ৰবেশ_size;
    বাৰ্তা->output = output;
    বাৰ্তা->output_size = output_size;
    GGML_ASSERT(queue.ঠেলো(বাৰ্তা));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = নূতন rpc_event;
    ev->বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    ev->বাৰ্তা->cmd = RPC_CMD_NONE;
    ev->sf = ev->বাৰ্তা->completion.get_future().share();
    GGML_ASSERT(queue.ঠেলো(ev->বাৰ্তা));
    ঘূৰাই নূতন ggml_backend_event {
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
    ev->বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    ev->বাৰ্তা->cmd = RPC_CMD_NONE;
    ev->sf = ev->বাৰ্তা->completion.get_future().share();
    GGML_ASSERT(queue.ঠেলো(ev->বাৰ্তা));
}

শূন্য rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto বাৰ্তা = std::make_shared<rpc_বাৰ্তা>();
    বাৰ্তা->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ঠেলো(বাৰ্তা));
    বাৰ্তা->completion.get_future().wait();
}

শূন্য rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

শূন্য rpc_dispatcher::busy_spin_release() {
    ধ্রুৱ unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

শূন্য rpc_dispatcher::start(ধ্রুৱ std::string & endpoint) {
    std::string host;
    পূর্ণাঙ্ক port;
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
    running = সঁচা;
    thread = std::thread(rpc_dispatcher_trampoline, এতত্);
}

শূন্য rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    যেতিয়ালৈ (running) {
        rpc_বাৰ্তা_ptr বাৰ্তা_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ধ্রুৱ তার্কিক spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        যদি (spin) {
            যদি (!queue.try_বেৰ_কৰো(&বাৰ্তা_ptr)) {
                rpc_cpu_relax();
                অনুৱর্ত;
            }
        } অন্যথা যদি (!queue.বেৰ_কৰো(&বাৰ্তা_ptr)) {
            ভঙ্গ;
        }
        যদি (বাৰ্তা_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(বাৰ্তা_ptr->cmd),
                     rpc_log_format_size(বাৰ্তা_ptr->প্ৰবেশ_size).c_str(), rpc_log_format_size(বাৰ্তা_ptr->output_size).c_str());
            ধ্রুৱ তার্কিক trace = rpc_log_debug_level() >= 3;
            ধ্রুৱ auto t0 = std::chrono::steady_clock::now();
            যদি (বাৰ্তা_ptr->output) {
                তার্কিক status = send_rpc_cmd(sock, বাৰ্তা_ptr->cmd, বাৰ্তা_ptr->প্ৰবেশ.get(), বাৰ্তা_ptr->প্ৰবেশ_size, বাৰ্তা_ptr->output, বাৰ্তা_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } অন্যথা {
                তার্কিক status = send_rpc_cmd(sock, বাৰ্তা_ptr->cmd, বাৰ্তা_ptr->প্ৰবেশ.get(), বাৰ্তা_ptr->প্ৰবেশ_size);
                RPC_STATUS_ASSERT(status);
            }
            যদি (trace) {
                ধ্রুৱ std::chrono::duration<দ্ৱিগুণ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(বাৰ্তা_ptr->cmd), elapsed.count());
            }
        } অন্যথা {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        বাৰ্তা_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = মিছা;
    queue.interrupt();
    sock = nullptr;
    যদি (thread.joinable()) {
        thread.join();
    }
}

স্থির std::shared_ptr<rpc_dispatcher> get_dispatcher(ধ্রুৱ std::string & endpoint) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    স্থির std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    যদি (it != dispatchers.end()) {
        যদি (auto dispatcher = it->second.lock()) {
            ঘূৰাই dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ঘূৰাই dispatcher;
}

স্থির শূন্য ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_বাৰ্তা_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, পরিমাণ(*request));
    delete ctx;
}

স্থির শূন্য * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    যদি (ctx->base_ptr != nullptr) {
        ঘূৰাই ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_বাৰ্তা_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_বাৰ্তা_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, পরিমাণ(*request), &response, পরিমাণ(response));
    ctx->base_ptr = reinterpret_cast<শূন্য *>(response.base_ptr);
    ঘূৰাই ctx->base_ptr;
}

স্থির তার্কিক ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    ঘূৰাই buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

স্থির rpc_tensor serialize_tensor(ধ্রুৱ ggml_tensor * tensor, ধ্রুৱ rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    যদি (!tensor) {
        memset(&result, 0, পরিমাণ(result));
        ঘূৰাই result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    যদি (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        যদি (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } অন্যথা {
            result.buffer = 0;
            result.data = 0;
        }
    } অন্যথা {
        result.buffer = 0;
        result.data   = 0;
    }
    বাবে (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    বাবে (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / পরিমাণ(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    বাবে (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, পরিমাণ(result.name));
    result.use_গুনতি = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ঘূৰাই result;
}

স্থির গণনা ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    যদি (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_বাৰ্তা_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, পরিমাণ(*request));
    }
    ঘূৰাই GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_বাৰ্তা_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, পরিমাণ(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
স্থির std::shared_ptr<uint8_t> serialize_set_tensor(ধ্রুৱ rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, ধ্রুৱ শূন্য * data, size_t size, size_t & প্ৰবেশ_size) {
    প্ৰবেশ_size = পরিমাণ(rpc_tensor) + পরিমাণ(cache_flag) + পরিমাণ(offset) + size;
    uint8_t * প্ৰবেশ = নূতন uint8_t[প্ৰবেশ_size]();
    uint8_t * p = প্ৰবেশ;
    memcpy(p, &rpc_tensor, পরিমাণ(rpc_tensor)); p += পরিমাণ(rpc_tensor);
    memcpy(p, &cache_flag, পরিমাণ(cache_flag)); p += পরিমাণ(cache_flag);
    memcpy(p, &offset,     পরিমাণ(offset));     p += পরিমাণ(offset);
    memcpy(p, data, size);
    ঘূৰাই std::shared_ptr<uint8_t>(প্ৰবেশ, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
স্থির তার্কিক rpc_use_hash_cache(ধ্রুৱ ggml_tensor * tensor, size_t size) {
    ঘূৰাই size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

স্থির শূন্য ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    যদি (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_বাৰ্তা_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ধ্রুৱ uint8_t*)data, size);
        rpc_বাৰ্তা_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, পরিমাণ(*request), &response, পরিমাণ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        যদি (response.result) {
            // the server has the same data, no need to send it
            ঘূৰাই;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t প্ৰবেশ_size;
    auto প্ৰবেশ = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, প্ৰবেশ_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, প্ৰবেশ, প্ৰবেশ_size);
}

স্থির শূন্য ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ধ্রুৱ শূন্য * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t প্ৰবেশ_size = পরিমাণ(rpc_tensor) + 4*পরিমাণ(uint64_t) + size*n_copies;
    uint8_t * প্ৰবেশ = নূতন uint8_t[প্ৰবেশ_size]();
    uint8_t * dest = প্ৰবেশ;
    memcpy(dest, &rpc_tensor, পরিমাণ(rpc_tensor));
    dest += পরিমাণ(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, পরিমাণ(header));
    dest += পরিমাণ(header);
    বাবে (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (ধ্রুৱ ৱর্ণ *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> প্ৰবেশ_ptr(প্ৰবেশ, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, প্ৰবেশ_ptr, প্ৰবেশ_size);
}

স্থির শূন্য ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, ধ্রুৱ ggml_tensor * tensor, শূন্য * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_বাৰ্তা_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    যদি (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, পরিমাণ(*request), data, size*n_copies);
    } অন্যথা {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, পরিমাণ(*request), packed.data(), packed.size());
        বাবে (size_t i = 0; i < n_copies; i++) {
            memcpy((ৱর্ণ *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

স্থির শূন্য ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, ধ্রুৱ ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_বাৰ্তা_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, পরিমাণ(*request), data, size);
}

স্থির তার্কিক ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ধ্রুৱ ggml_tensor * src, ggml_tensor * dst) {
    যদি (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        যদি (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ঘূৰাই মিছা;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_বাৰ্তা_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_বাৰ্তা_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, পরিমাণ(*request), &response, পরিমাণ(response));
        ঘূৰাই response.result;
    }
    ঘূৰাই মিছা;
}

স্থির শূন্য ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_বাৰ্তা_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, পরিমাণ(*request));
}

স্থির ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ রিক্ত,
};

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ঘূৰাই buft_ctx->name.c_str();
}

স্থির ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_বাৰ্তা_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_বাৰ্তা_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, পরিমাণ(*request), &response, পরিমাণ(response));
    যদি (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            নূতন ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        ঘূৰাই buffer;
    } অন্যথা {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        ঘূৰাই nullptr;
    }
}

স্থির size_t get_alignment(ধ্রুৱ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_বাৰ্তা_get_alignment_req>();
    request->device = device;
    rpc_বাৰ্তা_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, পরিমাণ(*request), &response, পরিমাণ(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    ঘূৰাই response.alignment;
}

স্থির size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ঘূৰাই buft_ctx->alignment;
}

স্থির size_t get_max_size(ধ্রুৱ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_বাৰ্তা_get_max_size_req>();
    request->device = device;
    rpc_বাৰ্তা_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, পরিমাণ(*request), &response, পরিমাণ(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    ঘূৰাই response.max_size;
}

স্থির size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ঘূৰাই buft_ctx->max_size;
}

স্থির size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুৱ ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    তার্কিক rpc_get = মিছা;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    যদি (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ধ্রুৱ size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        সংরচনা alloc_size_cache_চাবি {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / পরিমাণ(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_চাবি চাবি = {};
        চাবি.device = buft_ctx->device;
        চাবি.type = tensor->type;
        চাবি.op = tensor->op;
        memcpy(চাবি.op_params, tensor->op_params, পরিমাণ(চাবি.op_params));
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_DIMS; i++) {
            চাবি.ne[i] = (uint32_t)tensor->ne[i];
            চাবি.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ধ্রুৱ uint8_t *)&চাবি, পরিমাণ(চাবি));
        cache_hash = fnv_hash((ধ্রুৱ uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        স্থির std::mutex cache_mutex;
        স্থির std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            যদি (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                ঘূৰাই std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_বাৰ্তা_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_বাৰ্তা_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, পরিমাণ(*request), &response, পরিমাণ(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        ঘূৰাই std::max<size_t>(response.alloc_size, min_size);
    }

    ঘূৰাই ggml_nbytes(tensor);
}

স্থির ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ রিক্ত,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ রিক্ত,
    /* .is_host             = */ রিক্ত,
};

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ঘূৰাই rpc_ctx->name.c_str();
}

স্থির শূন্য ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

স্থির শূন্য ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ধ্রুৱ শূন্য * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    যদি (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_বাৰ্তা_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ধ্রুৱ uint8_t*)data, size);
        rpc_বাৰ্তা_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, পরিমাণ(*request), &response, পরিমাণ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        যদি (response.result) {
            // the server has the same data, no need to send it
            ঘূৰাই;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t প্ৰবেশ_size;
    auto প্ৰবেশ = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, প্ৰবেশ_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, প্ৰবেশ, প্ৰবেশ_size);
}

স্থির শূন্য ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ধ্রুৱ ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_বাৰ্তা_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, পরিমাণ(*request), data, size);
}

স্থির শূন্য ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

স্থির শূন্য add_tensor(ggml_tensor * tensor, ধ্রুৱ ggml_cgraph * cgraph, ধ্রুৱ rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    যদি (tensor == nullptr) {
        ঘূৰাই;
    }
    যদি (visited.find(tensor) != visited.end()) {
        ঘূৰাই;
    }
    visited.insert(tensor);
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    ধ্রুৱ size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    যদি (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_গুনতি = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

স্থির uint8_t * serialize_graph(uint32_t device, ধ্রুৱ ggml_cgraph * cgraph, ধ্রুৱ rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    বাবে (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*পরিমাণ(uint32_t) + পরিমাণ(uint64_t) + n_nodes * পরিমাণ(uint64_t) + পরিমাণ(uint32_t) + n_tensors * পরিমাণ(rpc_tensor);
    uint8_t * output = নূতন uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, পরিমাণ(device));
    dest += পরিমাণ(device);
    memcpy(dest, &cgraph->uid, পরিমাণ(cgraph->uid));
    dest += পরিমাণ(cgraph->uid);
    memcpy(dest, &n_nodes, পরিমাণ(n_nodes));
    dest += পরিমাণ(n_nodes);
    বাবে (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * পরিমাণ(uint64_t), &cgraph->nodes[i], পরিমাণ(uint64_t));
    }
    dest += n_nodes * পরিমাণ(uint64_t);
    memcpy(dest, &n_tensors, পরিমাণ(n_tensors));
    dest += পরিমাণ(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * পরিমাণ(rpc_tensor));
    ঘূৰাই output;
}

শূন্য rpc_dispatcher::graph_compute(uint32_t device, ধ্রুৱ ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    তার্কিক reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    যদি (reuse) {
        auto request = std::make_shared<rpc_বাৰ্তা_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, পরিমাণ(*request));
    } অন্যথা {
        যদি (cgraph->uid != 0) {
            যদি (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t প্ৰবেশ_size = 0;
        uint8_t * প্ৰবেশ = serialize_graph(device, cgraph, এতত্, &প্ৰবেশ_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(প্ৰবেশ_size).c_str());
        std::shared_ptr<uint8_t> প্ৰবেশ_ptr(প্ৰবেশ, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, প্ৰবেশ_ptr, প্ৰবেশ_size);
    }
}

স্থির গণনা ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    ঘূৰাই GGML_STATUS_SUCCESS;
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

স্থির শূন্য ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ধ্রুৱ শূন্য * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

স্থির শূন্য ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ধ্রুৱ ggml_tensor * tensor, শূন্য * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

স্থির ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ রিক্ত,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ রিক্ত,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(ধ্রুৱ ৱর্ণ * endpoint, uint32_t device) {
    স্থির std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    স্থির std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    যদি (it != buft_map.end()) {
        ঘূৰাই it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = নূতন ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = নূতন ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    ঘূৰাই buft;
}

ggml_backend_t ggml_backend_rpc_init(ধ্রুৱ ৱর্ণ * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = নূতন ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = নূতন ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ঘূৰাই backend;
}

তার্কিক ggml_backend_is_rpc(ggml_backend_t backend) {
    ঘূৰাই backend != রিক্ত && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

শূন্য ggml_backend_rpc_get_device_memory(ধ্রুৱ ৱর্ণ * endpoint, uint32_t device, size_t * মুক্ত, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_বাৰ্তা_get_device_memory_req>();
    request->device = device;
    rpc_বাৰ্তা_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, পরিমাণ(*request), &response, পরিমাণ(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *মুক্ত = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

শ্ৰেণী rpc_server {
সার্ৱজনিক:
    rpc_server(std::vector<ggml_backend_t> all_backends, ধ্রুৱ ৱর্ণ * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    শূন্য hello(rpc_বাৰ্তা_hello_rsp & response);
    তার্কিক alloc_buffer(ধ্রুৱ rpc_বাৰ্তা_alloc_buffer_req & request, rpc_বাৰ্তা_alloc_buffer_rsp & response);
    তার্কিক get_alignment(ধ্রুৱ rpc_বাৰ্তা_get_alignment_req & request, rpc_বাৰ্তা_get_alignment_rsp & response);
    তার্কিক get_max_size(ধ্রুৱ rpc_বাৰ্তা_get_max_size_req & request, rpc_বাৰ্তা_get_max_size_rsp & response);
    তার্কিক buffer_get_base(ধ্রুৱ rpc_বাৰ্তা_buffer_get_base_req & request, rpc_বাৰ্তা_buffer_get_base_rsp & response);
    তার্কিক free_buffer(ধ্রুৱ rpc_বাৰ্তা_free_buffer_req & request);
    তার্কিক buffer_clear(ধ্রুৱ rpc_বাৰ্তা_buffer_clear_req & request);
    তার্কিক memset_tensor(ধ্রুৱ rpc_বাৰ্তা_memset_tensor_req & request);
    তার্কিক set_tensor(ধ্রুৱ std::vector<uint8_t> & প্ৰবেশ);
    তার্কিক set_tensor_2d(ধ্রুৱ std::vector<uint8_t> & প্ৰবেশ);
    তার্কিক set_tensor_hash(ধ্রুৱ rpc_বাৰ্তা_set_tensor_hash_req & request, rpc_বাৰ্তা_set_tensor_hash_rsp & response);
    তার্কিক get_tensor(ধ্রুৱ rpc_বাৰ্তা_get_tensor_req & request, std::vector<uint8_t> & response);
    তার্কিক get_tensor_2d(ধ্রুৱ rpc_বাৰ্তা_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    তার্কিক copy_tensor(ধ্রুৱ rpc_বাৰ্তা_copy_tensor_req & request, rpc_বাৰ্তা_copy_tensor_rsp & response);
    তার্কিক graph_compute(ধ্রুৱ std::vector<uint8_t> & প্ৰবেশ);
    তার্কিক graph_recompute(ধ্রুৱ rpc_বাৰ্তা_graph_recompute_req & request);
    তার্কিক comm_init(ধ্রুৱ rpc_বাৰ্তা_comm_init_req & request, rpc_বাৰ্তা_comm_init_rsp & response);
    তার্কিক comm_allreduce(ধ্রুৱ rpc_বাৰ্তা_comm_allreduce_req & request);
    তার্কিক comm_free(ধ্রুৱ rpc_বাৰ্তা_comm_free_req & request);
    তার্কিক init_tensor(ধ্রুৱ rpc_বাৰ্তা_init_tensor_req & request);
    তার্কিক get_alloc_size(ধ্রুৱ rpc_বাৰ্তা_get_alloc_size_req & request, rpc_বাৰ্তা_get_alloc_size_rsp & response);
    তার্কিক get_device_memory(ধ্রুৱ rpc_বাৰ্তা_get_device_memory_req & request, rpc_বাৰ্তা_get_device_memory_rsp & response);

    সংরচনা stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

গুপ্ত:
    শূন্য sync_all_backends();
    তার্কিক get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(সংরচনা ggml_context * ctx, ধ্রুৱ rpc_tensor * tensor);
    ggml_tensor * create_গ্ৰন্থি(uint64_t id,
                              সংরচনা ggml_context * ctx,
                              ধ্রুৱ std::unordered_map<uint64_t, ধ্রুৱ rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, সংরচনা ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    সংরচনা comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ধ্রুৱ ৱর্ণ * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

শূন্য rpc_server::hello(rpc_বাৰ্তা_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

তার্কিক rpc_server::get_alloc_size(ধ্রুৱ rpc_বাৰ্তা_get_alloc_size_req & request, rpc_বাৰ্তা_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ঘূৰাই মিছা;
    }
    ggml_backend_buffer_type_t buft;
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ঘূৰাই মিছা;
    }
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        যদি (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (শূন্য*)tensor->buffer, tensor->data);
    যদি (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } অন্যথা {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::alloc_buffer(ধ্রুৱ rpc_বাৰ্তা_alloc_buffer_req & request, rpc_বাৰ্তা_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ঘূৰাই মিছা;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    যদি (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } অন্যথা {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::get_alignment(ধ্রুৱ rpc_বাৰ্তা_get_alignment_req & request, rpc_বাৰ্তা_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ঘূৰাই মিছা;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::get_max_size(ধ্রুৱ rpc_বাৰ্তা_get_max_size_req & request, rpc_বাৰ্তা_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ঘূৰাই মিছা;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::buffer_get_base(ধ্রুৱ rpc_বাৰ্তা_buffer_get_base_req & request, rpc_বাৰ্তা_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    যদি (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ঘূৰাই মিছা;
    }
    শূন্য * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::free_buffer(ধ্রুৱ rpc_বাৰ্তা_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    যদি (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ঘূৰাই মিছা;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    বাবে (auto & sgs : stored_graphs) {
        বাবে (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::buffer_clear(ধ্রুৱ rpc_বাৰ্তা_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    যদি (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ঘূৰাই মিছা;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::memset_tensor(ধ্রুৱ rpc_বাৰ্তা_memset_tensor_req & request) {
    sync_all_backends();
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }

    ধ্রুৱ uint64_t tensor_size = ggml_nbytes(tensor);
    যদি (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        ঘূৰাই মিছা;
    }

    ধ্রুৱ uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    ধ্রুৱ uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    যদি (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ঘূৰাই মিছা;
    }
    ধ্রুৱ uint64_t data_offset = request.tensor.data - buffer_start;
    যদি (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ঘূৰাই মিছা;
    }
    যদি (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ঘূৰাই মিছা;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    ঘূৰাই সঁচা;
}

ggml_tensor * rpc_server::deserialize_tensor(সংরচনা ggml_context * ctx, ধ্রুৱ rpc_tensor * tensor) {
    // Validate tensor type before using it
    যদি (tensor->type >= GGML_TYPE_গুনতি) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ঘূৰাই nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    যদি (ggml_blck_size((গণনা ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ঘূৰাই nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    যদি (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ঘূৰাই nullptr;
    }

    বাবে (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    যদি (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    যদি (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        যদি (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            ঘূৰাই nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    বাবে (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / পরিমাণ(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<শূন্য *>(tensor->data);
    ggml_set_name(result, tensor->name);
    ঘূৰাই result;
}


তার্কিক rpc_server::set_tensor(ধ্রুৱ std::vector<uint8_t> & প্ৰবেশ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    ধ্রুৱ size_t header_size = পরিমাণ(rpc_tensor) + পরিমাণ(cache_flag) + পরিমাণ(offset);
    যদি (প্ৰবেশ.size() < header_size) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ rpc_tensor * in_tensor = (ধ্রুৱ rpc_tensor *)প্ৰবেশ.data();
    memcpy(&cache_flag, প্ৰবেশ.data() + পরিমাণ(rpc_tensor), পরিমাণ(cache_flag));
    memcpy(&offset,     প্ৰবেশ.data() + পরিমাণ(rpc_tensor) + পরিমাণ(cache_flag), পরিমাণ(offset));
    ধ্রুৱ size_t size = প্ৰবেশ.size() - header_size;

    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    যদি (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        ধ্রুৱ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ধ্রুৱ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        যদি (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            ঘূৰাই মিছা;
        }
    }

    ধ্রুৱ শূন্য * data = প্ৰবেশ.data() + header_size;
    যদি (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ধ্রুৱ uint8_t*)data, size);
        ৱর্ণ hash_str[17];
        snprintf(hash_str, পরিমাণ(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.লিখ((ধ্রুৱ ৱর্ণ *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::set_tensor_2d(ধ্রুৱ std::vector<uint8_t> & প্ৰবেশ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    যদি (প্ৰবেশ.size() < পরিমাণ(rpc_tensor) + 4*পরিমাণ(uint64_t)) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ rpc_tensor * in_tensor = (ধ্রুৱ rpc_tensor *)প্ৰবেশ.data();
    uint64_t header[4];
    memcpy(header, প্ৰবেশ.data() + পরিমাণ(rpc_tensor), পরিমাণ(header));
    ধ্রুৱ uint64_t offset   = header[0];
    ধ্রুৱ uint64_t size     = header[1];
    ধ্রুৱ uint64_t n_copies = header[2];
    ধ্রুৱ uint64_t stride   = header[3];

    ধ্রুৱ uint64_t data_size = প্ৰবেশ.size() - পরিমাণ(rpc_tensor) - 4*পরিমাণ(uint64_t);
    যদি (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        ঘূৰাই মিছা;
    }

    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    যদি (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        যদি (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ঘূৰাই মিছা;
        }
        ধ্রুৱ uint64_t span = (n_copies - 1)*stride + size;
        ধ্রুৱ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ধ্রুৱ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        যদি (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            ঘূৰাই মিছা;
        }
        যদি (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            ঘূৰাই মিছা;
        }
    }

    ধ্রুৱ শূন্য * data = প্ৰবেশ.data() + পরিমাণ(rpc_tensor) + 4*পরিমাণ(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    যদি (!cache_dir) {
        ঘূৰাই মিছা;
    }
    ৱর্ণ hash_str[17];
    snprintf(hash_str, পরিমাণ(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    যদি (!fs::exists(cache_file, ec)) {
        ঘূৰাই মিছা;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.পঠ((ৱর্ণ *)data.data(), size);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::set_tensor_hash(ধ্রুৱ rpc_বাৰ্তা_set_tensor_hash_req & request, rpc_বাৰ্তা_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }

    std::vector<uint8_t> cached_file;
    যদি (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        ঘূৰাই সঁচা;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ধ্রুৱ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ধ্রুৱ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        যদি (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            ঘূৰাই মিছা;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::init_tensor(ধ্রুৱ rpc_বাৰ্তা_init_tensor_req & request) {
    sync_all_backends();
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ঘূৰাই মিছা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (শূন্য*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    যদি (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } অন্যথা {
        যদি (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    যদি (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ঘূৰাই মিছা;
    }

    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::get_tensor(ধ্রুৱ rpc_বাৰ্তা_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        ধ্রুৱ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ধ্রুৱ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        যদি (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                ঘূৰাই মিছা;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::get_tensor_2d(ধ্রুৱ rpc_বাৰ্তা_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    যদি (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (শূন্য*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        যদি (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ঘূৰাই মিছা;
        }
        যদি (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ঘূৰাই মিছা;
        }
        ধ্রুৱ uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        ধ্রুৱ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ধ্রুৱ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        যদি (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            ঘূৰাই মিছা;
        }
        যদি (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            ঘূৰাই মিছা;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::copy_tensor(ধ্রুৱ rpc_বাৰ্তা_copy_tensor_req & request, rpc_বাৰ্তা_copy_tensor_rsp & response) {
    sync_all_backends();
    সংরচনা ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    যদি (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ঘূৰাই মিছা;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    যদি (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        ঘূৰাই মিছা;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ঘূৰাই সঁচা;
}

ggml_tensor * rpc_server::create_গ্ৰন্থি(uint64_t id,
                                      সংরচনা ggml_context * ctx,
                                      ধ্রুৱ std::unordered_map<uint64_t, ধ্রুৱ rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, সংরচনা ggml_tensor*> & tensor_map) {
    যদি (tensor_map.find(id) != tensor_map.end()) {
        ঘূৰাই tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    যদি (it_ptr == tensor_ptrs.end()) {
        ঘূৰাই nullptr;
    }
    ধ্রুৱ rpc_tensor * tensor = it_ptr->second;

    সংরচনা ggml_tensor * result = deserialize_tensor(ctx, tensor);
    যদি (result == nullptr) {
        ঘূৰাই nullptr;
    }
    যদি (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ঘূৰাই nullptr;
    }
    tensor_map[id] = result;
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        যদি (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } অন্যথা {
            result->src[i] = create_গ্ৰন্থি(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            যদি (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ঘূৰাই nullptr;
            }
        }
    }

    // Handle view_src similarly
    যদি (tensor->view_src == 0) {
        result->view_src = nullptr;
    } অন্যথা {
        result->view_src = create_গ্ৰন্থি(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        যদি (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ঘূৰাই nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    ঘূৰাই result;
}

তার্কিক rpc_server::graph_compute(ধ্রুৱ std::vector<uint8_t> & প্ৰবেশ) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    যদি (প্ৰবেশ.size() < 2*পরিমাণ(uint32_t) + পরিমাণ(uint64_t)) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ uint8_t * src = প্ৰবেশ.data();
    uint32_t device;
    memcpy(&device, src, পরিমাণ(device));
    src += পরিমাণ(device);
    যদি (device >= backends.size()) {
        ঘূৰাই মিছা;
    }
    uint64_t uid;
    memcpy(&uid, src, পরিমাণ(uid));
    src += পরিমাণ(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, পরিমাণ(n_nodes));
    src += পরিমাণ(n_nodes);
    যদি (প্ৰবেশ.size() < 2*পরিমাণ(uint32_t) + পরিমাণ(uint64_t) + n_nodes*পরিমাণ(uint64_t) + পরিমাণ(uint32_t)) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ uint64_t * nodes = (ধ্রুৱ uint64_t *)src;
    src += n_nodes*পরিমাণ(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, পরিমাণ(n_tensors));
    src += পরিমাণ(n_tensors);
    যদি (প্ৰবেশ.size() < 2*পরিমাণ(uint32_t) + পরিমাণ(uint64_t) + n_nodes*পরিমাণ(uint64_t) + পরিমাণ(uint32_t) + n_tensors*পরিমাণ(rpc_tensor)) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ rpc_tensor * tensors = (ধ্রুৱ rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    যদি (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, মিছা);
    যদি (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    সংরচনা ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    সংরচনা ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, মিছা);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ধ্রুৱ rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    বাবে (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ধ্রুৱ auto t_deserialize = std::chrono::steady_clock::now();
    বাবে (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], পরিমাণ(id));
        graph->nodes[i] = create_গ্ৰন্থি(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        যদি (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ঘূৰাই মিছা;
        }
        যদি (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ধ্রুৱ size_t hash_pos = ggml_hash_ঢোকাও(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_গুনতি;
        }
    }
    ধ্রুৱ std::chrono::duration<দ্ৱিগুণ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::graph_recompute(ধ্রুৱ rpc_বাৰ্তা_graph_recompute_req & request) {
    uint32_t device = request.device;
    যদি (device >= backends.size()) {
        ঘূৰাই মিছা;
    }
    auto it = stored_graphs[device].find(request.uid);
    যদি (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ঘূৰাই মিছা;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ঘূৰাই সঁচা;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
শূন্য rpc_server::sync_all_backends() {
    বাবে (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
তার্কিক rpc_server::comm_init(ধ্রুৱ rpc_বাৰ্তা_comm_init_req & request, rpc_বাৰ্তা_comm_init_rsp & response) {
    response.ok = 0;
    যদি (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ঘূৰাই সঁচা;
    }
    comm_state & state = comm_states[request.device];
    যদি (state.peer != nullptr) {
        response.ok = 1;
        ঘূৰাই সঁচা;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    যদি (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        যদি (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            ঘূৰাই সঁচা;
        }
        state.peer = srv->accept();
        যদি (state.peer == nullptr) {
            ঘূৰাই সঁচা;
        }
        যদি (!state.peer->recv_data(remote_caps, পরিমাণ(remote_caps))) {
            state.peer = nullptr;
            ঘূৰাই সঁচা;
        }
        state.peer->get_caps(local_caps);
        যদি (!state.peer->send_data(local_caps, পরিমাণ(local_caps))) {
            state.peer = nullptr;
            ঘূৰাই সঁচা;
        }
        state.peer->update_caps(remote_caps);
    } অন্যথা {
        ধ্রুৱ std::string host(request.host, strnlen(request.host, পরিমাণ(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        বাবে (পূর্ণাঙ্ক i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            যদি (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        যদি (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            ঘূৰাই সঁচা;
        }
        state.peer->get_caps(local_caps);
        যদি (!state.peer->send_data(local_caps, পরিমাণ(local_caps)) ||
            !state.peer->recv_data(remote_caps, পরিমাণ(remote_caps))) {
            state.peer = nullptr;
            ঘূৰাই সঁচা;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::comm_allreduce(ধ্রুৱ rpc_বাৰ্তা_comm_allreduce_req & request) {
    যদি (request.device >= backends.size()) {
        ঘূৰাই মিছা;
    }
    comm_state & state = comm_states[request.device];
    যদি (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ঘূৰাই মিছা;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, মিছা);
    সংরচনা ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ রিক্ত,
        /*.no_alloc   =*/ সঁচা,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    যদি (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ঘূৰাই মিছা;
    }
    ধ্রুৱ size_t  nbytes = ggml_nbytes(t_dst);
    ধ্রুৱ int64_t ne     = ggml_nelements(t_dst);
    যদি (nbytes == 0) {
        ঘূৰাই সঁচা;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ধ্রুৱ তার্কিক   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ধ্রুৱ size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    ধ্রুৱ size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    যদি (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    ৱর্ণ * scratch_base = (ৱর্ণ *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        ঘূৰাই t;
    };
    auto new_cpy_গ্ৰন্থি = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ঘূৰাই t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, মিছা);
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
        compute_nodes(new_cpy_গ্ৰন্থি(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } অন্যথা {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    যদি (state.rank == 0) {
        যদি (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            ঘূৰাই মিছা;
        }
    } অন্যথা {
        যদি (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            ঘূৰাই মিছা;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    যদি (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_গ্ৰন্থি(t_wire_recv, t_peer);
    } অন্যথা {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    যদি (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } অন্যথা {
        compute_nodes(t_red, nullptr);
    }
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::comm_free(ধ্রুৱ rpc_বাৰ্তা_comm_free_req & request) {
    যদি (request.device >= backends.size()) {
        ঘূৰাই মিছা;
    }
    comm_states[request.device] = comm_state();
    ঘূৰাই সঁচা;
}

তার্কিক rpc_server::get_device_memory(ধ্রুৱ rpc_বাৰ্তা_get_device_memory_req & request, rpc_বাৰ্তা_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    যদি (dev_id >= backends.size()) {
        ঘূৰাই মিছা;
    }
    size_t মুক্ত, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &মুক্ত, &total);
    response.free_mem = মুক্ত;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    ঘূৰাই সঁচা;
}

rpc_server::~rpc_server() {
    বাবে (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

স্থির শূন্য rpc_serve_client(ধ্রুৱ std::vector<ggml_backend_t> & backends, ধ্রুৱ ৱর্ণ * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    যদি (!sock->recv_data(&cmd, 1)) {
        ঘূৰাই;
    }
    যদি (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ঘূৰাই;
    }

    // Read input_size and validate protocol version
    uint64_t hello_প্ৰবেশ_size;
    যদি (!sock->recv_data(&hello_প্ৰবেশ_size, পরিমাণ(hello_প্ৰবেশ_size))) {
        ঘূৰাই;
    }

    যদি (hello_প্ৰবেশ_size != পরিমাণ(rpc_বাৰ্তা_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_প্ৰবেশ_size, পরিমাণ(rpc_বাৰ্তা_hello_req), RPC_PROTO_MAJOR_VERSION);
        ঘূৰাই;
    }

    rpc_বাৰ্তা_hello_req req = {};
    যদি (!sock->recv_data(&req, পরিমাণ(req))) {
        ঘূৰাই;
    }

    rpc_বাৰ্তা_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    যদি (!send_বাৰ্তা(sock, &rsp, পরিমাণ(rsp))) {
        ঘূৰাই;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    যেতিয়ালৈ (সঁচা) {
        যদি (!sock->recv_data(&cmd, 1)) {
            ভঙ্গ;
        }
        যদি (cmd >= RPC_CMD_গুনতি) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            ভঙ্গ;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((গণনা rpc_cmd)cmd));
        ৱিকল্প (cmd) {
            পক্ষ RPC_CMD_HELLO: {
                // HELLO command is handled above
                ঘূৰাই;
            }
            পক্ষ RPC_CMD_DEVICE_গুনতি: {
                যদি (!recv_বাৰ্তা(sock, nullptr, 0)) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_device_গুনতি_rsp response;
                response.device_গুনতি = backends.size();
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_ALLOC_BUFFER: {
                rpc_বাৰ্তা_alloc_buffer_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_alloc_buffer_rsp response;
                যদি (!server.alloc_buffer(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GET_ALLOC_SIZE: {
                rpc_বাৰ্তা_get_alloc_size_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_get_alloc_size_rsp response;
                যদি (!server.get_alloc_size(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GET_ALIGNMENT: {
                rpc_বাৰ্তা_get_alignment_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_get_alignment_rsp response;
                যদি (!server.get_alignment(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GET_MAX_SIZE: {
                rpc_বাৰ্তা_get_max_size_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_get_max_size_rsp response;
                যদি (!server.get_max_size(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_BUFFER_GET_BASE: {
                rpc_বাৰ্তা_buffer_get_base_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_buffer_get_base_rsp response;
                যদি (!server.buffer_get_base(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_FREE_BUFFER: {
                rpc_বাৰ্তা_free_buffer_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.free_buffer(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_BUFFER_CLEAR: {
                rpc_বাৰ্তা_buffer_clear_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.buffer_clear(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_MEMSET_TENSOR: {
                rpc_বাৰ্তা_memset_tensor_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.memset_tensor(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> প্ৰবেশ;
                যদি (!recv_বাৰ্তা(sock, প্ৰবেশ)) {
                    ঘূৰাই;
                }
                যদি (!server.set_tensor(প্ৰবেশ)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> প্ৰবেশ;
                যদি (!recv_বাৰ্তা(sock, প্ৰবেশ)) {
                    ঘূৰাই;
                }
                যদি (!server.set_tensor_2d(প্ৰবেশ)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GET_TENSOR_2D: {
                rpc_বাৰ্তা_get_tensor_2d_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                std::vector<uint8_t> response;
                যদি (!server.get_tensor_2d(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, response.data(), response.size())) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_SET_TENSOR_HASH: {
                rpc_বাৰ্তা_set_tensor_hash_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_set_tensor_hash_rsp response;
                যদি (!server.set_tensor_hash(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_INIT_TENSOR: {
                rpc_বাৰ্তা_init_tensor_req request;
                যদি (!recv_বাৰ্তা(sock, &request,পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.init_tensor(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GET_TENSOR: {
                rpc_বাৰ্তা_get_tensor_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                std::vector<uint8_t> response;
                যদি (!server.get_tensor(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, response.data(), response.size())) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_COPY_TENSOR: {
                rpc_বাৰ্তা_copy_tensor_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_copy_tensor_rsp response;
                যদি (!server.copy_tensor(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> প্ৰবেশ;
                যদি (!recv_বাৰ্তা(sock, প্ৰবেশ)) {
                    ঘূৰাই;
                }
                যদি (!server.graph_compute(প্ৰবেশ)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_বাৰ্তা_graph_recompute_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.graph_recompute(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_COMM_INIT: {
                rpc_বাৰ্তা_comm_init_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_comm_init_rsp response;
                যদি (!server.comm_init(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_COMM_ALLREDUCE: {
                rpc_বাৰ্তা_comm_allreduce_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.comm_allreduce(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_COMM_FREE: {
                rpc_বাৰ্তা_comm_free_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                যদি (!server.comm_free(request)) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            পক্ষ RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_বাৰ্তা_get_device_memory_req request;
                যদি (!recv_বাৰ্তা(sock, &request, পরিমাণ(request))) {
                    ঘূৰাই;
                }
                rpc_বাৰ্তা_get_device_memory_rsp response;
                যদি (!server.get_device_memory(request, response)) {
                    ঘূৰাই;
                }
                যদি (!send_বাৰ্তা(sock, &response, পরিমাণ(response))) {
                    ঘূৰাই;
                }
                ভঙ্গ;
            }
            স্ৱভাৱ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((গণনা rpc_cmd)cmd));
                ঘূৰাই;
            }
        }
    }
}

শূন্য ggml_backend_rpc_start_server(ধ্রুৱ ৱর্ণ * endpoint, ধ্রুৱ ৱর্ণ * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    যদি (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ঘূৰাই;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    বাবে (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t মুক্ত, total;
        ggml_backend_dev_memory(dev, &মুক্ত, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, মুক্ত / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        যদি (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ঘূৰাই;
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
    পূর্ণাঙ্ক port;
    যদি (!parse_endpoint(endpoint, host, port)) {
        ঘূৰাই;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#অন্যথা
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    যদি (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ঘূৰাই;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    যদি (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        ঘূৰাই;
    }
    যেতিয়ালৈ (সঁচা) {
        auto client_socket = server_socket->accept();
        যদি (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            ঘূৰাই;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    বাবে (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ঘূৰাই ctx->name.c_str();
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ঘূৰাই ctx->description.c_str();
}

স্থির শূন্য ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, মুক্ত, total);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ঘূৰাই GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, সংরচনা ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ সঁচা,
        /* .host_buffer           = */ মিছা,
        /* .buffer_from_host_ptr  = */ মিছা,
        /* .events                = */ সঁচা,
        /* .mmap_support          = */ সঁচা,
    };
}

স্থির ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ধ্রুৱ ৱর্ণ * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ঘূৰাই ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

স্থির ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ঘূৰাই ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

স্থির তার্কিক ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ধ্রুৱ সংরচনা ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    ঘূৰাই সঁচা;
}

স্থির তার্কিক ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    যদি (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        ঘূৰাই মিছা;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ঘূৰাই buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

স্থির ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    ঘূৰাই dispatcher->event_new(dev);
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

স্থির ধ্রুৱ সংরচনা ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ রিক্ত,
    /* .buffer_from_host_ptr = */ রিক্ত,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ রিক্ত,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

সংরচনা ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ঘূৰাই ctx ? ctx->name.c_str() : "RPC";
}

স্থির size_t ggml_backend_rpc_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ঘূৰাই ctx ? ctx->devices.size() : 0;
}

স্থির ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    যদি (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } অন্যথা {
        GGML_ASSERT(index < ctx->devices.size());
        ঘূৰাই ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
সংরচনা ggml_backend_rpc_comm_context {
    সংরচনা rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

স্থির শূন্য ggml_backend_rpc_comm_free(শূন্য * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    যদি (comm_ctx == nullptr) {
        ঘূৰাই;
    }
    বাবে (ধ্রুৱ auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_বাৰ্তা_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, পরিমাণ(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

স্থির শূন্য * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    যদি (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        ঘূৰাই nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    বাবে (size_t i = 0; i < n_backends; i++) {
        যদি (!ggml_backend_is_rpc(backends[i])) {
            ঘূৰাই nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        বাবে (ধ্রুৱ auto & rank : ranks) {
            যদি (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ঘূৰাই nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    পূর্ণাঙ্ক port0;
    যদি (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ঘূৰাই nullptr;
    }
    ধ্রুৱ uint32_t comm_port = (uint32_t) port0 + 1000;
    যদি (host0.size() >= 64) {
        ঘূৰাই nullptr;
    }

    বাবে (ধ্রুৱ auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_বাৰ্তা_comm_init_rsp> responses(n_backends);
    বাবে (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_বাৰ্তা_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        যদি (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, পরিমাণ(*request), &responses[i], পরিমাণ(responses[i]));
    }
    বাবে (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    তার্কিক ok = সঁচা;
    বাবে (size_t i = 0; i < n_backends; i++) {
        যদি (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = মিছা;
        }
    }
    যদি (!ok) {
        বাবে (ধ্রুৱ auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ঘূৰাই nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ঘূৰাই নূতন ggml_backend_rpc_comm_context{std::move(ranks)};
}

স্থির তার্কিক ggml_backend_rpc_comm_allreduce_tensor(শূন্য * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    যদি (comm_ctx == nullptr) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ size_t n_ranks = comm_ctx->ranks.size();
    ধ্রুৱ int64_t ne = ggml_nelements(tensors[0]);
    যদি (ne == 0) {
        ঘূৰাই সঁচা;
    }
    বাবে (size_t i = 0; i < n_ranks; i++) {
        যদি (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            ঘূৰাই মিছা;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        যদি ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ঘূৰাই মিছা;
        }
    }
    বাবে (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_বাৰ্তা_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, পরিমাণ(*request));
    }
    ঘূৰাই সঁচা;
}

স্থির শূন্য * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, ধ্রুৱ ৱর্ণ * name) {
    যদি (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ঘূৰাই (শূন্য *)ggml_backend_rpc_add_server;
    }
    যদি (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ঘূৰাই (শূন্য *)ggml_backend_rpc_start_server;
    }
    যদি (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ঘূৰাই (শূন্য *)ggml_backend_rpc_comm_init;
    }
    যদি (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ঘূৰাই (শূন্য *)ggml_backend_rpc_comm_free;
    }
    যদি (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ঘূৰাই (শূন্য *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ঘূৰাই রিক্ত;

    GGML_UNUSED(reg);
}

স্থির ধ্রুৱ সংরচনা ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(শূন্য) {
    স্থির সংরচনা ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ রিক্ত,
    };

    ঘূৰাই &ggml_backend_rpc_reg;
}

স্থির uint32_t ggml_backend_rpc_get_device_গুনতি(ধ্রুৱ ৱর্ণ * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_বাৰ্তা_device_গুনতি_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_গুনতি, nullptr, 0, &response, পরিমাণ(response));
    ঘূৰাই response.device_গুনতি;
}

স্থির ধ্রুৱ ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_গুনতি,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(ধ্রুৱ ৱর্ণ * endpoint) {
    স্থির std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    স্থির std::mutex mutex;
    স্থির uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    যদি (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ঘূৰাই reg_map[endpoint];
    }
    uint32_t dev_গুনতি = ggml_backend_rpc_get_device_গুনতি(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_গুনতি);
    যদি (dev_গুনতি == 0) {
        ঘূৰাই nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = নূতন ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    বাবে (uint32_t ind = 0; ind < dev_গুনতি; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = নূতন ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = নূতন ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = নূতন ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ঘূৰাই reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
