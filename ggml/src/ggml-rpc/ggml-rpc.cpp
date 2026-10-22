#شامل "ggml-rpc.h"
#شامل "ggml-impl.h"
#شامل "ggml-backend-impl.h"
#شامل "ggml-cpp.h"
#شامل "transport.h"
#شامل "log.h"

#شامل <array>
#شامل <chrono>
#شامل <cinttypes>
#شامل <optional>
#شامل <string>
#شامل <thread>
#شامل <vector>
#شامل <queue>
#شامل <condition_variable>
#شامل <future>
#شامل <memory>
#شامل <mutex>
#شامل <unordered_map>
#شامل <unordered_set>
#شامل <cstring>
#شامل <fstream>
#شامل <filesystem>
#شامل <algorithm>
#شامل <atomic>

فضاینام fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) اگر (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(افزودن, 1)
// ggml_tensor is serialized into rpc_tensor
ساختار rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / اندازه(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    نویسه name[GGML_MAX_NAME];

    int32_t use_شمار;
};

static_assert(اندازه(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
شمارش rpc_cmd {
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
    RPC_CMD_DEVICE_شمار,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_شمار,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ایستا ثابت نویسه * rpc_cmd_name(شمارش rpc_cmd cmd) {
    انتخاب (cmd) {
        حالت RPC_CMD_ALLOC_BUFFER:      بازگشت "ALLOC_BUFFER";
        حالت RPC_CMD_GET_ALIGNMENT:     بازگشت "GET_ALIGNMENT";
        حالت RPC_CMD_GET_MAX_SIZE:      بازگشت "GET_MAX_SIZE";
        حالت RPC_CMD_BUFFER_GET_BASE:   بازگشت "BUFFER_GET_BASE";
        حالت RPC_CMD_FREE_BUFFER:       بازگشت "FREE_BUFFER";
        حالت RPC_CMD_BUFFER_CLEAR:      بازگشت "BUFFER_CLEAR";
        حالت RPC_CMD_SET_TENSOR:        بازگشت "SET_TENSOR";
        حالت RPC_CMD_SET_TENSOR_HASH:   بازگشت "SET_TENSOR_HASH";
        حالت RPC_CMD_GET_TENSOR:        بازگشت "GET_TENSOR";
        حالت RPC_CMD_COPY_TENSOR:       بازگشت "COPY_TENSOR";
        حالت RPC_CMD_GRAPH_COMPUTE:     بازگشت "GRAPH_COMPUTE";
        حالت RPC_CMD_GET_DEVICE_MEMORY: بازگشت "GET_DEVICE_MEMORY";
        حالت RPC_CMD_INIT_TENSOR:       بازگشت "INIT_TENSOR";
        حالت RPC_CMD_GET_ALLOC_SIZE:    بازگشت "GET_ALLOC_SIZE";
        حالت RPC_CMD_HELLO:             بازگشت "HELLO";
        حالت RPC_CMD_DEVICE_شمار:      بازگشت "DEVICE_COUNT";
        حالت RPC_CMD_GRAPH_RECOMPUTE:   بازگشت "GRAPH_RECOMPUTE";
        حالت RPC_CMD_MEMSET_TENSOR:     بازگشت "MEMSET_TENSOR";
        حالت RPC_CMD_SET_TENSOR_2D:     بازگشت "SET_TENSOR_2D";
        حالت RPC_CMD_GET_TENSOR_2D:     بازگشت "GET_TENSOR_2D";
        حالت RPC_CMD_COMM_INIT:         بازگشت "COMM_INIT";
        حالت RPC_CMD_COMM_ALLREDUCE:    بازگشت "COMM_ALLREDUCE";
        حالت RPC_CMD_COMM_FREE:         بازگشت "COMM_FREE";
        حالت RPC_CMD_NONE:              بازگشت "NONE";
        پیشفرض:                        بازگشت "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ثابت size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ثابت size_t GRAPH_CACHE_MAX = 1024;

ساختار rpc_پیام_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ساختار rpc_پیام_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ساختار rpc_پیام_device_شمار_rsp {
    uint32_t device_شمار;
};

ساختار rpc_پیام_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ساختار rpc_پیام_get_alloc_size_rsp {
    uint64_t alloc_size;
};

ساختار rpc_پیام_init_tensor_req {
    rpc_tensor tensor;
};

ساختار rpc_پیام_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

ساختار rpc_پیام_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

ساختار rpc_پیام_get_alignment_req {
    uint32_t device;
};

ساختار rpc_پیام_get_alignment_rsp {
    uint64_t alignment;
};

ساختار rpc_پیام_get_max_size_req {
    uint32_t device;
};

ساختار rpc_پیام_get_max_size_rsp {
    uint64_t max_size;
};

ساختار rpc_پیام_buffer_get_base_req {
    uint64_t remote_ptr;
};

ساختار rpc_پیام_buffer_get_base_rsp {
    uint64_t base_ptr;
};

ساختار rpc_پیام_free_buffer_req {
    uint64_t remote_ptr;
};

ساختار rpc_پیام_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ساختار rpc_پیام_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

ساختار rpc_پیام_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

ساختار rpc_پیام_set_tensor_hash_rsp {
    uint8_t result;
};

ساختار rpc_پیام_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

ساختار rpc_پیام_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ساختار rpc_پیام_copy_tensor_rsp {
    uint8_t result;
};

ساختار rpc_پیام_get_device_memory_req {
    uint32_t device;
};

ساختار rpc_پیام_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

ساختار rpc_پیام_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ساختار rpc_پیام_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ساختار rpc_پیام_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    نویسه     host[64];  // rank > 0: rank 0's host
};

ساختار rpc_پیام_comm_init_rsp {
    uint8_t ok;
};

ساختار rpc_پیام_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ساختار rpc_پیام_comm_free_req {
    uint32_t device;
};

#pragma pack(برداشت)

// RPC data structures

ایستا ggml_guid_t ggml_backend_rpc_guid() {
    ایستا ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    بازگشت &guid;
}

ساختار ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ساختار ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

رده rpc_dispatcher;
ساختار ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ساختار ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    تهی                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ایستا uint64_t fnv_hash(ثابت uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ثابت uint64_t fnv_اول = 0x100000001b3ULL;

    برای (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_اول;
    }
    بازگشت hash;
}

ایستا منطقی send_پیام(socket_ptr sock, ثابت تهی * پیام, size_t پیام_size) {
    اگر (!sock->send_data(&پیام_size, اندازه(پیام_size))) {
        بازگشت نادرست;
    }
    اگر (!sock->send_data(پیام, پیام_size)) {
        بازگشت نادرست;
    }
    بازگشت sock->flush();
}

ایستا منطقی recv_پیام(socket_ptr sock, تهی * پیام, size_t پیام_size) {
    uint64_t size;
    اگر (!sock->recv_data(&size, اندازه(size))) {
        بازگشت نادرست;
    }
    اگر (size != پیام_size) {
        بازگشت نادرست;
    }
    بازگشت sock->recv_data(پیام, پیام_size);
}

ایستا منطقی recv_پیام(socket_ptr sock, std::vector<uint8_t> & درون_داد) {
    uint64_t size;
    اگر (!sock->recv_data(&size, اندازه(size))) {
        بازگشت نادرست;
    }
    تلاش {
        درون_داد.resize(size);
    } بگیر (ثابت std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        بازگشت نادرست;
    }
    بازگشت sock->recv_data(درون_داد.data(), size);
}

ایستا منطقی parse_endpoint(ثابت std::string & endpoint, std::string & host, صحیح & port) {
    size_t pos = endpoint.find(':');
    اگر (pos == std::string::npos) {
        بازگشت نادرست;
    }
    host = endpoint.substr(0, pos);
    تلاش {
        port = std::stoi(endpoint.substr(pos + 1));
    } بگیر (...) {
        بازگشت نادرست;
    }
    بازگشت درست;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ایستا منطقی send_rpc_cmd(socket_ptr sock, شمارش rpc_cmd cmd, ثابت تهی * درون_داد, size_t درون_داد_size) {
    uint8_t cmd_byte = cmd;
    اگر (!sock->send_data(&cmd_byte, اندازه(cmd_byte))) {
        بازگشت نادرست;
    }
    اگر (!sock->send_data(&درون_داد_size, اندازه(درون_داد_size))) {
        بازگشت نادرست;
    }
    اگر (!sock->send_data(درون_داد, درون_داد_size)) {
        بازگشت نادرست;
    }
    بازگشت sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ایستا منطقی send_rpc_cmd(socket_ptr sock, شمارش rpc_cmd cmd, ثابت تهی * درون_داد, size_t درون_داد_size, تهی * output, size_t output_size) {
    اگر (!send_rpc_cmd(sock, cmd, درون_داد, درون_داد_size)) {
        بازگشت نادرست;
    }
    uint64_t out_size;
    اگر (!sock->recv_data(&out_size, اندازه(out_size))) {
        بازگشت نادرست;
    }
    اگر (out_size != output_size) {
        بازگشت نادرست;
    }
    اگر (!sock->recv_data(output, output_size)) {
        بازگشت نادرست;
    }
    بازگشت درست;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ایستا constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

ایستا درونخطی تهی rpc_cpu_relax() {
#اگر defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ فرار("yield" ::: "memory");
#وگرنه
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ایستا منطقی negotiate_hello(ثابت std::shared_ptr<socket_t> & sock) {
    rpc_پیام_hello_req request = {};
    rpc_پیام_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    منطقی status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, اندازه(request), &response, اندازه(response));
    RPC_STATUS_ASSERT(status);

    اگر (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        بازگشت نادرست;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    بازگشت درست;
}

template <typename T>
رده message_queue {
عمومی:
    message_queue() {}

    منطقی افزودن(ثابت T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        اگر (interrupted) {
            بازگشت نادرست;
        }
        queue.افزودن(value);
        cvar.notify_all();
        بازگشت درست;
    }

    منطقی برداشت(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [این] { بازگشت !queue.empty() || interrupted; });
        اگر (interrupted) {
            بازگشت نادرست;
        }
        *out = queue.front();
        queue.برداشت();
        بازگشت درست;
    }

    منطقی try_برداشت(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        اگر (interrupted || queue.empty()) {
            بازگشت نادرست;
        }
        *out = queue.front();
        queue.برداشت();
        بازگشت درست;
    }

    تهی interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = درست;
        lock.unlock();
        cvar.notify_all();
    }

خصوصی:
    منطقی interrupted = نادرست;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

رده rpc_dispatcher {
عمومی:
    rpc_dispatcher() {
    }

    تهی send(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size);
    تهی send(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size, تهی * output, size_t output_size);
    تهی send_async(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size);
    تهی send_async(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size, تهی * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    تهی event_free(ggml_backend_event_t event);
    تهی event_synchronize(ggml_backend_event_t event);
    تهی event_record(ggml_backend_event_t event);
    تهی synchronize();
    تهی busy_spin_acquire();
    تهی busy_spin_release();
    تهی graph_compute(uint32_t device, ثابت ggml_cgraph * cgraph);

    تهی start(ثابت std::string & endpoint);
    تهی work();

    ~rpc_dispatcher();

خصوصی:
    ساختار rpc_پیام {
        rpc_cmd                       cmd;
        std::shared_ptr<ثابت تهی>   درون_داد;
        size_t                        درون_داد_size;
        تهی                        * output;
        size_t                        output_size;
        std::promise<تهی>            completion;
    };
    using rpc_پیام_ptr   = std::shared_ptr<rpc_پیام>;
    using rpc_پیام_queue = message_queue<rpc_پیام_ptr>;
    ساختار rpc_event {
        rpc_پیام_ptr              پیام;
        std::shared_future<تهی> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_پیام_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ایستا تهی rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

تهی rpc_dispatcher::send(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size) {
    auto پیام = std::make_shared<rpc_پیام>();
    پیام->cmd = cmd;
    پیام->درون_داد = درون_داد;
    پیام->درون_داد_size = درون_داد_size;
    پیام->output = nullptr;
    پیام->output_size = 0;
    GGML_ASSERT(queue.افزودن(پیام));
    auto future = پیام->completion.get_future();
    future.wait();
}

تهی rpc_dispatcher::send_async(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size) {
    auto پیام = std::make_shared<rpc_پیام>();
    پیام->cmd = cmd;
    پیام->درون_داد = درون_داد;
    پیام->درون_داد_size = درون_داد_size;
    پیام->output = nullptr;
    پیام->output_size = 0;
    GGML_ASSERT(queue.افزودن(پیام));
}

تهی rpc_dispatcher::send(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size, تهی * output, size_t output_size) {
    auto پیام = std::make_shared<rpc_پیام>();
    پیام->cmd = cmd;
    پیام->درون_داد = درون_داد;
    پیام->درون_داد_size = درون_داد_size;
    پیام->output = output;
    پیام->output_size = output_size;
    GGML_ASSERT(queue.افزودن(پیام));
    auto future = پیام->completion.get_future();
    future.wait();
}

تهی rpc_dispatcher::send_async(شمارش rpc_cmd cmd, std::shared_ptr<ثابت تهی> درون_داد, size_t درون_داد_size, تهی * output, size_t output_size) {
    auto پیام = std::make_shared<rpc_پیام>();
    پیام->cmd = cmd;
    پیام->درون_داد = درون_داد;
    پیام->درون_داد_size = درون_داد_size;
    پیام->output = output;
    پیام->output_size = output_size;
    GGML_ASSERT(queue.افزودن(پیام));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = جدید rpc_event;
    ev->پیام = std::make_shared<rpc_پیام>();
    ev->پیام->cmd = RPC_CMD_NONE;
    ev->sf = ev->پیام->completion.get_future().share();
    GGML_ASSERT(queue.افزودن(ev->پیام));
    بازگشت جدید ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

تهی rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

تهی rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

تهی rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->پیام = std::make_shared<rpc_پیام>();
    ev->پیام->cmd = RPC_CMD_NONE;
    ev->sf = ev->پیام->completion.get_future().share();
    GGML_ASSERT(queue.افزودن(ev->پیام));
}

تهی rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto پیام = std::make_shared<rpc_پیام>();
    پیام->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.افزودن(پیام));
    پیام->completion.get_future().wait();
}

تهی rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

تهی rpc_dispatcher::busy_spin_release() {
    ثابت unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

تهی rpc_dispatcher::start(ثابت std::string & endpoint) {
    std::string host;
    صحیح port;
    اگر (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    اگر (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    اگر (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    اگر (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = درست;
    thread = std::thread(rpc_dispatcher_trampoline, این);
}

تهی rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    وقتی (running) {
        rpc_پیام_ptr پیام_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ثابت منطقی spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        اگر (spin) {
            اگر (!queue.try_برداشت(&پیام_ptr)) {
                rpc_cpu_relax();
                ادامه;
            }
        } وگرنه اگر (!queue.برداشت(&پیام_ptr)) {
            بشکن;
        }
        اگر (پیام_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(پیام_ptr->cmd),
                     rpc_log_format_size(پیام_ptr->درون_داد_size).c_str(), rpc_log_format_size(پیام_ptr->output_size).c_str());
            ثابت منطقی trace = rpc_log_debug_level() >= 3;
            ثابت auto t0 = std::chrono::steady_clock::now();
            اگر (پیام_ptr->output) {
                منطقی status = send_rpc_cmd(sock, پیام_ptr->cmd, پیام_ptr->درون_داد.get(), پیام_ptr->درون_داد_size, پیام_ptr->output, پیام_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } وگرنه {
                منطقی status = send_rpc_cmd(sock, پیام_ptr->cmd, پیام_ptr->درون_داد.get(), پیام_ptr->درون_داد_size);
                RPC_STATUS_ASSERT(status);
            }
            اگر (trace) {
                ثابت std::chrono::duration<دوگانه, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(پیام_ptr->cmd), elapsed.count());
            }
        } وگرنه {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        پیام_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = نادرست;
    queue.interrupt();
    sock = nullptr;
    اگر (thread.joinable()) {
        thread.join();
    }
}

ایستا std::shared_ptr<rpc_dispatcher> get_dispatcher(ثابت std::string & endpoint) {
    ایستا std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    ایستا std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    اگر (it != dispatchers.end()) {
        اگر (auto dispatcher = it->second.lock()) {
            بازگشت dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    بازگشت dispatcher;
}

ایستا تهی ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_پیام_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, اندازه(*request));
    delete ctx;
}

ایستا تهی * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    اگر (ctx->base_ptr != nullptr) {
        بازگشت ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_پیام_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_پیام_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, اندازه(*request), &response, اندازه(response));
    ctx->base_ptr = reinterpret_cast<تهی *>(response.base_ptr);
    بازگشت ctx->base_ptr;
}

ایستا منطقی ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    بازگشت buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

ایستا rpc_tensor serialize_tensor(ثابت ggml_tensor * tensor, ثابت rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    اگر (!tensor) {
        memset(&result, 0, اندازه(result));
        بازگشت result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    اگر (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        اگر (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } وگرنه {
            result.buffer = 0;
            result.data = 0;
        }
    } وگرنه {
        result.buffer = 0;
        result.data   = 0;
    }
    برای (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    برای (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / اندازه(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    برای (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, اندازه(result.name));
    result.use_شمار = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    بازگشت result;
}

ایستا شمارش ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    اگر (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_پیام_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, اندازه(*request));
    }
    بازگشت GGML_STATUS_SUCCESS;
}

ایستا تهی ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_پیام_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, اندازه(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ایستا std::shared_ptr<uint8_t> serialize_set_tensor(ثابت rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, ثابت تهی * data, size_t size, size_t & درون_داد_size) {
    درون_داد_size = اندازه(rpc_tensor) + اندازه(cache_flag) + اندازه(offset) + size;
    uint8_t * درون_داد = جدید uint8_t[درون_داد_size]();
    uint8_t * p = درون_داد;
    memcpy(p, &rpc_tensor, اندازه(rpc_tensor)); p += اندازه(rpc_tensor);
    memcpy(p, &cache_flag, اندازه(cache_flag)); p += اندازه(cache_flag);
    memcpy(p, &offset,     اندازه(offset));     p += اندازه(offset);
    memcpy(p, data, size);
    بازگشت std::shared_ptr<uint8_t>(درون_داد, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ایستا منطقی rpc_use_hash_cache(ثابت ggml_tensor * tensor, size_t size) {
    بازگشت size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ایستا تهی ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    اگر (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_پیام_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ثابت uint8_t*)data, size);
        rpc_پیام_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, اندازه(*request), &response, اندازه(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        اگر (response.result) {
            // the server has the same data, no need to send it
            بازگشت;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t درون_داد_size;
    auto درون_داد = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, درون_داد_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, درون_داد, درون_داد_size);
}

ایستا تهی ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ثابت تهی * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t درون_داد_size = اندازه(rpc_tensor) + 4*اندازه(uint64_t) + size*n_copies;
    uint8_t * درون_داد = جدید uint8_t[درون_داد_size]();
    uint8_t * dest = درون_داد;
    memcpy(dest, &rpc_tensor, اندازه(rpc_tensor));
    dest += اندازه(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, اندازه(header));
    dest += اندازه(header);
    برای (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (ثابت نویسه *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> درون_داد_ptr(درون_داد, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, درون_داد_ptr, درون_داد_size);
}

ایستا تهی ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, ثابت ggml_tensor * tensor, تهی * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_پیام_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    اگر (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, اندازه(*request), data, size*n_copies);
    } وگرنه {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, اندازه(*request), packed.data(), packed.size());
        برای (size_t i = 0; i < n_copies; i++) {
            memcpy((نویسه *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

ایستا تهی ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_پیام_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, اندازه(*request), data, size);
}

ایستا منطقی ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        اگر (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            بازگشت نادرست;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_پیام_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_پیام_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, اندازه(*request), &response, اندازه(response));
        بازگشت response.result;
    }
    بازگشت نادرست;
}

ایستا تهی ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_پیام_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, اندازه(*request));
}

ایستا ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ پوچ,
};

ایستا ثابت نویسه * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    بازگشت buft_ctx->name.c_str();
}

ایستا ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_پیام_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_پیام_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, اندازه(*request), &response, اندازه(response));
    اگر (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            جدید ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        بازگشت buffer;
    } وگرنه {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        بازگشت nullptr;
    }
}

ایستا size_t get_alignment(ثابت std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_پیام_get_alignment_req>();
    request->device = device;
    rpc_پیام_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, اندازه(*request), &response, اندازه(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    بازگشت response.alignment;
}

ایستا size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    بازگشت buft_ctx->alignment;
}

ایستا size_t get_max_size(ثابت std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_پیام_get_max_size_req>();
    request->device = device;
    rpc_پیام_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, اندازه(*request), &response, اندازه(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    بازگشت response.max_size;
}

ایستا size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    بازگشت buft_ctx->max_size;
}

ایستا size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    منطقی rpc_get = نادرست;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    اگر (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ثابت size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ساختار alloc_size_cache_کلید {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / اندازه(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_کلید کلید = {};
        کلید.device = buft_ctx->device;
        کلید.type = tensor->type;
        کلید.op = tensor->op;
        memcpy(کلید.op_params, tensor->op_params, اندازه(کلید.op_params));
        برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
            کلید.ne[i] = (uint32_t)tensor->ne[i];
            کلید.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ثابت uint8_t *)&کلید, اندازه(کلید));
        cache_hash = fnv_hash((ثابت uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ایستا std::mutex cache_mutex;
        ایستا std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            اگر (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                بازگشت std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_پیام_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_پیام_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, اندازه(*request), &response, اندازه(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        بازگشت std::max<size_t>(response.alloc_size, min_size);
    }

    بازگشت ggml_nbytes(tensor);
}

ایستا ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ پوچ,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ پوچ,
    /* .is_host             = */ پوچ,
};

ایستا ثابت نویسه * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    بازگشت rpc_ctx->name.c_str();
}

ایستا تهی ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

ایستا تهی ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    اگر (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_پیام_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ثابت uint8_t*)data, size);
        rpc_پیام_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, اندازه(*request), &response, اندازه(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        اگر (response.result) {
            // the server has the same data, no need to send it
            بازگشت;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t درون_داد_size;
    auto درون_داد = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, درون_داد_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, درون_داد, درون_داد_size);
}

ایستا تهی ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_پیام_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, اندازه(*request), data, size);
}

ایستا تهی ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ایستا تهی add_tensor(ggml_tensor * tensor, ثابت ggml_cgraph * cgraph, ثابت rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    اگر (tensor == nullptr) {
        بازگشت;
    }
    اگر (visited.find(tensor) != visited.end()) {
        بازگشت;
    }
    visited.insert(tensor);
    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    ثابت size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    اگر (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_شمار = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

ایستا uint8_t * serialize_graph(uint32_t device, ثابت ggml_cgraph * cgraph, ثابت rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    برای (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*اندازه(uint32_t) + اندازه(uint64_t) + n_nodes * اندازه(uint64_t) + اندازه(uint32_t) + n_tensors * اندازه(rpc_tensor);
    uint8_t * output = جدید uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, اندازه(device));
    dest += اندازه(device);
    memcpy(dest, &cgraph->uid, اندازه(cgraph->uid));
    dest += اندازه(cgraph->uid);
    memcpy(dest, &n_nodes, اندازه(n_nodes));
    dest += اندازه(n_nodes);
    برای (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * اندازه(uint64_t), &cgraph->nodes[i], اندازه(uint64_t));
    }
    dest += n_nodes * اندازه(uint64_t);
    memcpy(dest, &n_tensors, اندازه(n_tensors));
    dest += اندازه(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * اندازه(rpc_tensor));
    بازگشت output;
}

تهی rpc_dispatcher::graph_compute(uint32_t device, ثابت ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    منطقی reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    اگر (reuse) {
        auto request = std::make_shared<rpc_پیام_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, اندازه(*request));
    } وگرنه {
        اگر (cgraph->uid != 0) {
            اگر (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t درون_داد_size = 0;
        uint8_t * درون_داد = serialize_graph(device, cgraph, این, &درون_داد_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(درون_داد_size).c_str());
        std::shared_ptr<uint8_t> درون_داد_ptr(درون_داد, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, درون_داد_ptr, درون_داد_size);
    }
}

ایستا شمارش ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    بازگشت GGML_STATUS_SUCCESS;
}

ایستا تهی ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

ایستا تهی ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

ایستا تهی ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت تهی * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

ایستا تهی ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, تهی * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

ایستا ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ پوچ,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ پوچ,
    /* .graph_plan_free         = */ پوچ,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ پوچ,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ پوچ,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(ثابت نویسه * endpoint, uint32_t device) {
    ایستا std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ایستا std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    اگر (it != buft_map.end()) {
        بازگشت it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = جدید ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = جدید ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    بازگشت buft;
}

ggml_backend_t ggml_backend_rpc_init(ثابت نویسه * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = جدید ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = جدید ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    بازگشت backend;
}

منطقی ggml_backend_is_rpc(ggml_backend_t backend) {
    بازگشت backend != پوچ && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

تهی ggml_backend_rpc_get_device_memory(ثابت نویسه * endpoint, uint32_t device, size_t * آزاد, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_پیام_get_device_memory_req>();
    request->device = device;
    rpc_پیام_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, اندازه(*request), &response, اندازه(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *آزاد = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

رده rpc_server {
عمومی:
    rpc_server(std::vector<ggml_backend_t> all_backends, ثابت نویسه * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    تهی hello(rpc_پیام_hello_rsp & response);
    منطقی alloc_buffer(ثابت rpc_پیام_alloc_buffer_req & request, rpc_پیام_alloc_buffer_rsp & response);
    منطقی get_alignment(ثابت rpc_پیام_get_alignment_req & request, rpc_پیام_get_alignment_rsp & response);
    منطقی get_max_size(ثابت rpc_پیام_get_max_size_req & request, rpc_پیام_get_max_size_rsp & response);
    منطقی buffer_get_base(ثابت rpc_پیام_buffer_get_base_req & request, rpc_پیام_buffer_get_base_rsp & response);
    منطقی free_buffer(ثابت rpc_پیام_free_buffer_req & request);
    منطقی buffer_clear(ثابت rpc_پیام_buffer_clear_req & request);
    منطقی memset_tensor(ثابت rpc_پیام_memset_tensor_req & request);
    منطقی set_tensor(ثابت std::vector<uint8_t> & درون_داد);
    منطقی set_tensor_2d(ثابت std::vector<uint8_t> & درون_داد);
    منطقی set_tensor_hash(ثابت rpc_پیام_set_tensor_hash_req & request, rpc_پیام_set_tensor_hash_rsp & response);
    منطقی get_tensor(ثابت rpc_پیام_get_tensor_req & request, std::vector<uint8_t> & response);
    منطقی get_tensor_2d(ثابت rpc_پیام_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    منطقی copy_tensor(ثابت rpc_پیام_copy_tensor_req & request, rpc_پیام_copy_tensor_rsp & response);
    منطقی graph_compute(ثابت std::vector<uint8_t> & درون_داد);
    منطقی graph_recompute(ثابت rpc_پیام_graph_recompute_req & request);
    منطقی comm_init(ثابت rpc_پیام_comm_init_req & request, rpc_پیام_comm_init_rsp & response);
    منطقی comm_allreduce(ثابت rpc_پیام_comm_allreduce_req & request);
    منطقی comm_free(ثابت rpc_پیام_comm_free_req & request);
    منطقی init_tensor(ثابت rpc_پیام_init_tensor_req & request);
    منطقی get_alloc_size(ثابت rpc_پیام_get_alloc_size_req & request, rpc_پیام_get_alloc_size_rsp & response);
    منطقی get_device_memory(ثابت rpc_پیام_get_device_memory_req & request, rpc_پیام_get_device_memory_rsp & response);

    ساختار stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

خصوصی:
    تهی sync_all_backends();
    منطقی get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(ساختار ggml_context * ctx, ثابت rpc_tensor * tensor);
    ggml_tensor * create_گره(uint64_t id,
                              ساختار ggml_context * ctx,
                              ثابت std::unordered_map<uint64_t, ثابت rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ساختار ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ساختار comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ثابت نویسه * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

تهی rpc_server::hello(rpc_پیام_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

منطقی rpc_server::get_alloc_size(ثابت rpc_پیام_get_alloc_size_req & request, rpc_پیام_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        بازگشت نادرست;
    }
    ggml_backend_buffer_type_t buft;
    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        بازگشت نادرست;
    }
    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (تهی*)tensor->buffer, tensor->data);
    اگر (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } وگرنه {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    بازگشت درست;
}

منطقی rpc_server::alloc_buffer(ثابت rpc_پیام_alloc_buffer_req & request, rpc_پیام_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        بازگشت نادرست;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    اگر (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } وگرنه {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    بازگشت درست;
}

منطقی rpc_server::get_alignment(ثابت rpc_پیام_get_alignment_req & request, rpc_پیام_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        بازگشت نادرست;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    بازگشت درست;
}

منطقی rpc_server::get_max_size(ثابت rpc_پیام_get_max_size_req & request, rpc_پیام_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        بازگشت نادرست;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    بازگشت درست;
}

منطقی rpc_server::buffer_get_base(ثابت rpc_پیام_buffer_get_base_req & request, rpc_پیام_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    اگر (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        بازگشت نادرست;
    }
    تهی * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    بازگشت درست;
}

منطقی rpc_server::free_buffer(ثابت rpc_پیام_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    اگر (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        بازگشت نادرست;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    برای (auto & sgs : stored_graphs) {
        برای (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    بازگشت درست;
}

منطقی rpc_server::buffer_clear(ثابت rpc_پیام_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    اگر (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        بازگشت نادرست;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    بازگشت درست;
}

منطقی rpc_server::memset_tensor(ثابت rpc_پیام_memset_tensor_req & request) {
    sync_all_backends();
    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }

    ثابت uint64_t tensor_size = ggml_nbytes(tensor);
    اگر (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        بازگشت نادرست;
    }

    ثابت uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    ثابت uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    اگر (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        بازگشت نادرست;
    }
    ثابت uint64_t data_offset = request.tensor.data - buffer_start;
    اگر (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        بازگشت نادرست;
    }
    اگر (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        بازگشت نادرست;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (تهی *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    بازگشت درست;
}

ggml_tensor * rpc_server::deserialize_tensor(ساختار ggml_context * ctx, ثابت rpc_tensor * tensor) {
    // Validate tensor type before using it
    اگر (tensor->type >= GGML_TYPE_شمار) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        بازگشت nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    اگر (ggml_blck_size((شمارش ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        بازگشت nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    اگر (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        بازگشت nullptr;
    }

    برای (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    اگر (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    اگر (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        اگر (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            بازگشت nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    برای (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / اندازه(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<تهی *>(tensor->data);
    ggml_set_name(result, tensor->name);
    بازگشت result;
}


منطقی rpc_server::set_tensor(ثابت std::vector<uint8_t> & درون_داد) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    ثابت size_t header_size = اندازه(rpc_tensor) + اندازه(cache_flag) + اندازه(offset);
    اگر (درون_داد.size() < header_size) {
        بازگشت نادرست;
    }
    ثابت rpc_tensor * in_tensor = (ثابت rpc_tensor *)درون_داد.data();
    memcpy(&cache_flag, درون_داد.data() + اندازه(rpc_tensor), اندازه(cache_flag));
    memcpy(&offset,     درون_داد.data() + اندازه(rpc_tensor) + اندازه(cache_flag), اندازه(offset));
    ثابت size_t size = درون_داد.size() - header_size;

    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    اگر (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (تهی*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ثابت size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        اگر (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            بازگشت نادرست;
        }
    }

    ثابت تهی * data = درون_داد.data() + header_size;
    اگر (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ثابت uint8_t*)data, size);
        نویسه hash_str[17];
        snprintf(hash_str, اندازه(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.اكتب((ثابت نویسه *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    بازگشت درست;
}

منطقی rpc_server::set_tensor_2d(ثابت std::vector<uint8_t> & درون_داد) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    اگر (درون_داد.size() < اندازه(rpc_tensor) + 4*اندازه(uint64_t)) {
        بازگشت نادرست;
    }
    ثابت rpc_tensor * in_tensor = (ثابت rpc_tensor *)درون_داد.data();
    uint64_t header[4];
    memcpy(header, درون_داد.data() + اندازه(rpc_tensor), اندازه(header));
    ثابت uint64_t offset   = header[0];
    ثابت uint64_t size     = header[1];
    ثابت uint64_t n_copies = header[2];
    ثابت uint64_t stride   = header[3];

    ثابت uint64_t data_size = درون_داد.size() - اندازه(rpc_tensor) - 4*اندازه(uint64_t);
    اگر (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        بازگشت نادرست;
    }

    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    اگر (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (تهی*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        اگر (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            بازگشت نادرست;
        }
        ثابت uint64_t span = (n_copies - 1)*stride + size;
        ثابت uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ثابت uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        اگر (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            بازگشت نادرست;
        }
        اگر (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            بازگشت نادرست;
        }
    }

    ثابت تهی * data = درون_داد.data() + اندازه(rpc_tensor) + 4*اندازه(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    بازگشت درست;
}

منطقی rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    اگر (!cache_dir) {
        بازگشت نادرست;
    }
    نویسه hash_str[17];
    snprintf(hash_str, اندازه(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    اگر (!fs::exists(cache_file, ec)) {
        بازگشت نادرست;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.اقرأ((نویسه *)data.data(), size);
    بازگشت درست;
}

منطقی rpc_server::set_tensor_hash(ثابت rpc_پیام_set_tensor_hash_req & request, rpc_پیام_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }

    std::vector<uint8_t> cached_file;
    اگر (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        بازگشت درست;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (تهی*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ثابت size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        اگر (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            بازگشت نادرست;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    بازگشت درست;
}

منطقی rpc_server::init_tensor(ثابت rpc_پیام_init_tensor_req & request) {
    sync_all_backends();
    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        بازگشت نادرست;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (تهی*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    اگر (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } وگرنه {
        اگر (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    اگر (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        بازگشت نادرست;
    }

    بازگشت درست;
}

منطقی rpc_server::get_tensor(ثابت rpc_پیام_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (تهی*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ثابت size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        اگر (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                بازگشت نادرست;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    بازگشت درست;
}

منطقی rpc_server::get_tensor_2d(ثابت rpc_پیام_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ساختار ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (تهی*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        اگر (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            بازگشت نادرست;
        }
        اگر (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            بازگشت نادرست;
        }
        ثابت uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        ثابت uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ثابت uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        اگر (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            بازگشت نادرست;
        }
        اگر (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            بازگشت نادرست;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    بازگشت درست;
}

منطقی rpc_server::copy_tensor(ثابت rpc_پیام_copy_tensor_req & request, rpc_پیام_copy_tensor_rsp & response) {
    sync_all_backends();
    ساختار ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    اگر (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        بازگشت نادرست;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    اگر (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        بازگشت نادرست;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    بازگشت درست;
}

ggml_tensor * rpc_server::create_گره(uint64_t id,
                                      ساختار ggml_context * ctx,
                                      ثابت std::unordered_map<uint64_t, ثابت rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ساختار ggml_tensor*> & tensor_map) {
    اگر (tensor_map.find(id) != tensor_map.end()) {
        بازگشت tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    اگر (it_ptr == tensor_ptrs.end()) {
        بازگشت nullptr;
    }
    ثابت rpc_tensor * tensor = it_ptr->second;

    ساختار ggml_tensor * result = deserialize_tensor(ctx, tensor);
    اگر (result == nullptr) {
        بازگشت nullptr;
    }
    اگر (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        بازگشت nullptr;
    }
    tensor_map[id] = result;
    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        اگر (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } وگرنه {
            result->src[i] = create_گره(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            اگر (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                بازگشت nullptr;
            }
        }
    }

    // Handle view_src similarly
    اگر (tensor->view_src == 0) {
        result->view_src = nullptr;
    } وگرنه {
        result->view_src = create_گره(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        اگر (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            بازگشت nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    بازگشت result;
}

منطقی rpc_server::graph_compute(ثابت std::vector<uint8_t> & درون_داد) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    اگر (درون_داد.size() < 2*اندازه(uint32_t) + اندازه(uint64_t)) {
        بازگشت نادرست;
    }
    ثابت uint8_t * src = درون_داد.data();
    uint32_t device;
    memcpy(&device, src, اندازه(device));
    src += اندازه(device);
    اگر (device >= backends.size()) {
        بازگشت نادرست;
    }
    uint64_t uid;
    memcpy(&uid, src, اندازه(uid));
    src += اندازه(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, اندازه(n_nodes));
    src += اندازه(n_nodes);
    اگر (درون_داد.size() < 2*اندازه(uint32_t) + اندازه(uint64_t) + n_nodes*اندازه(uint64_t) + اندازه(uint32_t)) {
        بازگشت نادرست;
    }
    ثابت uint64_t * nodes = (ثابت uint64_t *)src;
    src += n_nodes*اندازه(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, اندازه(n_tensors));
    src += اندازه(n_tensors);
    اگر (درون_داد.size() < 2*اندازه(uint32_t) + اندازه(uint64_t) + n_nodes*اندازه(uint64_t) + اندازه(uint32_t) + n_tensors*اندازه(rpc_tensor)) {
        بازگشت نادرست;
    }
    ثابت rpc_tensor * tensors = (ثابت rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    اگر (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, نادرست);
    اگر (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    ساختار ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ساختار ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, نادرست);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ثابت rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    برای (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ثابت auto t_deserialize = std::chrono::steady_clock::now();
    برای (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], اندازه(id));
        graph->nodes[i] = create_گره(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        اگر (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            بازگشت نادرست;
        }
        اگر (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ثابت size_t hash_pos = ggml_hash_درج(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_شمار;
        }
    }
    ثابت std::chrono::duration<دوگانه, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    بازگشت درست;
}

منطقی rpc_server::graph_recompute(ثابت rpc_پیام_graph_recompute_req & request) {
    uint32_t device = request.device;
    اگر (device >= backends.size()) {
        بازگشت نادرست;
    }
    auto it = stored_graphs[device].find(request.uid);
    اگر (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        بازگشت نادرست;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    بازگشت درست;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
تهی rpc_server::sync_all_backends() {
    برای (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
منطقی rpc_server::comm_init(ثابت rpc_پیام_comm_init_req & request, rpc_پیام_comm_init_rsp & response) {
    response.ok = 0;
    اگر (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        بازگشت درست;
    }
    comm_state & state = comm_states[request.device];
    اگر (state.peer != nullptr) {
        response.ok = 1;
        بازگشت درست;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    اگر (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        اگر (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            بازگشت درست;
        }
        state.peer = srv->accept();
        اگر (state.peer == nullptr) {
            بازگشت درست;
        }
        اگر (!state.peer->recv_data(remote_caps, اندازه(remote_caps))) {
            state.peer = nullptr;
            بازگشت درست;
        }
        state.peer->get_caps(local_caps);
        اگر (!state.peer->send_data(local_caps, اندازه(local_caps))) {
            state.peer = nullptr;
            بازگشت درست;
        }
        state.peer->update_caps(remote_caps);
    } وگرنه {
        ثابت std::string host(request.host, strnlen(request.host, اندازه(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        برای (صحیح i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            اگر (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        اگر (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            بازگشت درست;
        }
        state.peer->get_caps(local_caps);
        اگر (!state.peer->send_data(local_caps, اندازه(local_caps)) ||
            !state.peer->recv_data(remote_caps, اندازه(remote_caps))) {
            state.peer = nullptr;
            بازگشت درست;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    بازگشت درست;
}

منطقی rpc_server::comm_allreduce(ثابت rpc_پیام_comm_allreduce_req & request) {
    اگر (request.device >= backends.size()) {
        بازگشت نادرست;
    }
    comm_state & state = comm_states[request.device];
    اگر (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        بازگشت نادرست;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, نادرست);
    ساختار ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ پوچ,
        /*.no_alloc   =*/ درست,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    اگر (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        بازگشت نادرست;
    }
    ثابت size_t  nbytes = ggml_nbytes(t_dst);
    ثابت int64_t ne     = ggml_nelements(t_dst);
    اگر (nbytes == 0) {
        بازگشت درست;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ثابت منطقی   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ثابت size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    ثابت size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    اگر (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    نویسه * scratch_base = (نویسه *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        بازگشت t;
    };
    auto new_cpy_گره = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        بازگشت t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, نادرست);
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
    اگر (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_گره(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } وگرنه {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    اگر (state.rank == 0) {
        اگر (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            بازگشت نادرست;
        }
    } وگرنه {
        اگر (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            بازگشت نادرست;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    اگر (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_گره(t_wire_recv, t_peer);
    } وگرنه {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    اگر (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } وگرنه {
        compute_nodes(t_red, nullptr);
    }
    بازگشت درست;
}

منطقی rpc_server::comm_free(ثابت rpc_پیام_comm_free_req & request) {
    اگر (request.device >= backends.size()) {
        بازگشت نادرست;
    }
    comm_states[request.device] = comm_state();
    بازگشت درست;
}

منطقی rpc_server::get_device_memory(ثابت rpc_پیام_get_device_memory_req & request, rpc_پیام_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        بازگشت نادرست;
    }
    size_t آزاد, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &آزاد, &total);
    response.free_mem = آزاد;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    بازگشت درست;
}

rpc_server::~rpc_server() {
    برای (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

ایستا تهی rpc_serve_client(ثابت std::vector<ggml_backend_t> & backends, ثابت نویسه * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    اگر (!sock->recv_data(&cmd, 1)) {
        بازگشت;
    }
    اگر (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        بازگشت;
    }

    // Read input_size and validate protocol version
    uint64_t hello_درون_داد_size;
    اگر (!sock->recv_data(&hello_درون_داد_size, اندازه(hello_درون_داد_size))) {
        بازگشت;
    }

    اگر (hello_درون_داد_size != اندازه(rpc_پیام_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_درون_داد_size, اندازه(rpc_پیام_hello_req), RPC_PROTO_MAJOR_VERSION);
        بازگشت;
    }

    rpc_پیام_hello_req req = {};
    اگر (!sock->recv_data(&req, اندازه(req))) {
        بازگشت;
    }

    rpc_پیام_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    اگر (!send_پیام(sock, &rsp, اندازه(rsp))) {
        بازگشت;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    وقتی (درست) {
        اگر (!sock->recv_data(&cmd, 1)) {
            بشکن;
        }
        اگر (cmd >= RPC_CMD_شمار) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            بشکن;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((شمارش rpc_cmd)cmd));
        انتخاب (cmd) {
            حالت RPC_CMD_HELLO: {
                // HELLO command is handled above
                بازگشت;
            }
            حالت RPC_CMD_DEVICE_شمار: {
                اگر (!recv_پیام(sock, nullptr, 0)) {
                    بازگشت;
                }
                rpc_پیام_device_شمار_rsp response;
                response.device_شمار = backends.size();
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_ALLOC_BUFFER: {
                rpc_پیام_alloc_buffer_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_alloc_buffer_rsp response;
                اگر (!server.alloc_buffer(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GET_ALLOC_SIZE: {
                rpc_پیام_get_alloc_size_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_get_alloc_size_rsp response;
                اگر (!server.get_alloc_size(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GET_ALIGNMENT: {
                rpc_پیام_get_alignment_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_get_alignment_rsp response;
                اگر (!server.get_alignment(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GET_MAX_SIZE: {
                rpc_پیام_get_max_size_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_get_max_size_rsp response;
                اگر (!server.get_max_size(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_BUFFER_GET_BASE: {
                rpc_پیام_buffer_get_base_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_buffer_get_base_rsp response;
                اگر (!server.buffer_get_base(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_FREE_BUFFER: {
                rpc_پیام_free_buffer_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.free_buffer(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_BUFFER_CLEAR: {
                rpc_پیام_buffer_clear_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.buffer_clear(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_MEMSET_TENSOR: {
                rpc_پیام_memset_tensor_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.memset_tensor(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> درون_داد;
                اگر (!recv_پیام(sock, درون_داد)) {
                    بازگشت;
                }
                اگر (!server.set_tensor(درون_داد)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> درون_داد;
                اگر (!recv_پیام(sock, درون_داد)) {
                    بازگشت;
                }
                اگر (!server.set_tensor_2d(درون_داد)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GET_TENSOR_2D: {
                rpc_پیام_get_tensor_2d_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                std::vector<uint8_t> response;
                اگر (!server.get_tensor_2d(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, response.data(), response.size())) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_SET_TENSOR_HASH: {
                rpc_پیام_set_tensor_hash_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_set_tensor_hash_rsp response;
                اگر (!server.set_tensor_hash(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_INIT_TENSOR: {
                rpc_پیام_init_tensor_req request;
                اگر (!recv_پیام(sock, &request,اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.init_tensor(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GET_TENSOR: {
                rpc_پیام_get_tensor_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                std::vector<uint8_t> response;
                اگر (!server.get_tensor(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, response.data(), response.size())) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_COPY_TENSOR: {
                rpc_پیام_copy_tensor_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_copy_tensor_rsp response;
                اگر (!server.copy_tensor(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> درون_داد;
                اگر (!recv_پیام(sock, درون_داد)) {
                    بازگشت;
                }
                اگر (!server.graph_compute(درون_داد)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_پیام_graph_recompute_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.graph_recompute(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_COMM_INIT: {
                rpc_پیام_comm_init_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_comm_init_rsp response;
                اگر (!server.comm_init(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_COMM_ALLREDUCE: {
                rpc_پیام_comm_allreduce_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.comm_allreduce(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_COMM_FREE: {
                rpc_پیام_comm_free_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                اگر (!server.comm_free(request)) {
                    بازگشت;
                }
                بشکن;
            }
            حالت RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_پیام_get_device_memory_req request;
                اگر (!recv_پیام(sock, &request, اندازه(request))) {
                    بازگشت;
                }
                rpc_پیام_get_device_memory_rsp response;
                اگر (!server.get_device_memory(request, response)) {
                    بازگشت;
                }
                اگر (!send_پیام(sock, &response, اندازه(response))) {
                    بازگشت;
                }
                بشکن;
            }
            پیشفرض: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((شمارش rpc_cmd)cmd));
                بازگشت;
            }
        }
    }
}

تهی ggml_backend_rpc_start_server(ثابت نویسه * endpoint, ثابت نویسه * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    اگر (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        بازگشت;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    برای (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t آزاد, total;
        ggml_backend_dev_memory(dev, &آزاد, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, آزاد / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        اگر (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            بازگشت;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        اگر (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            اگر (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    صحیح port;
    اگر (!parse_endpoint(endpoint, host, port)) {
        بازگشت;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#وگرنه
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    اگر (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        بازگشت;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    اگر (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        بازگشت;
    }
    وقتی (درست) {
        auto client_socket = server_socket->accept();
        اگر (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            بازگشت;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    برای (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

ایستا ثابت نویسه * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    بازگشت ctx->name.c_str();
}

ایستا ثابت نویسه * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    بازگشت ctx->description.c_str();
}

ایستا تهی ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * آزاد, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, آزاد, total);
}

ایستا شمارش ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    بازگشت GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ساختار ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ درست,
        /* .host_buffer           = */ نادرست,
        /* .buffer_from_host_ptr  = */ نادرست,
        /* .events                = */ درست,
        /* .mmap_support          = */ درست,
    };
}

ایستا ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ثابت نویسه * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    بازگشت ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

ایستا ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    بازگشت ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

ایستا منطقی ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ثابت ساختار ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    بازگشت درست;
}

ایستا منطقی ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    اگر (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        بازگشت نادرست;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    بازگشت buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ایستا ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    بازگشت dispatcher->event_new(dev);
}

ایستا تهی ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ایستا تهی ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ایستا ثابت ساختار ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ پوچ,
    /* .buffer_from_host_ptr = */ پوچ,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ پوچ,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ساختار ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ایستا ثابت نویسه * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    بازگشت ctx ? ctx->name.c_str() : "RPC";
}

ایستا size_t ggml_backend_rpc_reg_get_device_شمار(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    بازگشت ctx ? ctx->devices.size() : 0;
}

ایستا ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    اگر (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } وگرنه {
        GGML_ASSERT(index < ctx->devices.size());
        بازگشت ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ساختار ggml_backend_rpc_comm_context {
    ساختار rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ایستا تهی ggml_backend_rpc_comm_free(تهی * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    اگر (comm_ctx == nullptr) {
        بازگشت;
    }
    برای (ثابت auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_پیام_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, اندازه(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

ایستا تهی * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    اگر (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        بازگشت nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    برای (size_t i = 0; i < n_backends; i++) {
        اگر (!ggml_backend_is_rpc(backends[i])) {
            بازگشت nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        برای (ثابت auto & rank : ranks) {
            اگر (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                بازگشت nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    صحیح port0;
    اگر (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        بازگشت nullptr;
    }
    ثابت uint32_t comm_port = (uint32_t) port0 + 1000;
    اگر (host0.size() >= 64) {
        بازگشت nullptr;
    }

    برای (ثابت auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_پیام_comm_init_rsp> responses(n_backends);
    برای (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_پیام_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        اگر (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, اندازه(*request), &responses[i], اندازه(responses[i]));
    }
    برای (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    منطقی ok = درست;
    برای (size_t i = 0; i < n_backends; i++) {
        اگر (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = نادرست;
        }
    }
    اگر (!ok) {
        برای (ثابت auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        بازگشت nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    بازگشت جدید ggml_backend_rpc_comm_context{std::move(ranks)};
}

ایستا منطقی ggml_backend_rpc_comm_allreduce_tensor(تهی * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    اگر (comm_ctx == nullptr) {
        بازگشت نادرست;
    }
    ثابت size_t n_ranks = comm_ctx->ranks.size();
    ثابت int64_t ne = ggml_nelements(tensors[0]);
    اگر (ne == 0) {
        بازگشت درست;
    }
    برای (size_t i = 0; i < n_ranks; i++) {
        اگر (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            بازگشت نادرست;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        اگر ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            بازگشت نادرست;
        }
    }
    برای (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_پیام_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, اندازه(*request));
    }
    بازگشت درست;
}

ایستا تهی * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    اگر (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        بازگشت (تهی *)ggml_backend_rpc_add_server;
    }
    اگر (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        بازگشت (تهی *)ggml_backend_rpc_start_server;
    }
    اگر (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        بازگشت (تهی *)ggml_backend_rpc_comm_init;
    }
    اگر (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        بازگشت (تهی *)ggml_backend_rpc_comm_free;
    }
    اگر (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        بازگشت (تهی *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    بازگشت پوچ;

    GGML_UNUSED(reg);
}

ایستا ثابت ساختار ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(تهی) {
    ایستا ساختار ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ پوچ,
    };

    بازگشت &ggml_backend_rpc_reg;
}

ایستا uint32_t ggml_backend_rpc_get_device_شمار(ثابت نویسه * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_پیام_device_شمار_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_شمار, nullptr, 0, &response, اندازه(response));
    بازگشت response.device_شمار;
}

ایستا ثابت ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_شمار,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(ثابت نویسه * endpoint) {
    ایستا std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ایستا std::mutex mutex;
    ایستا uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    اگر (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        بازگشت reg_map[endpoint];
    }
    uint32_t dev_شمار = ggml_backend_rpc_get_device_شمار(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_شمار);
    اگر (dev_شمار == 0) {
        بازگشت nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = جدید ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    برای (uint32_t ind = 0; ind < dev_شمار; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = جدید ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = جدید ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = جدید ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    بازگشت reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
