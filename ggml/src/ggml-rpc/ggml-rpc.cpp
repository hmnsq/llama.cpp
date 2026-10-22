#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "log.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) གལ་སྲིད (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
sa_nracanaa rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / parimaa_na(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    var_na name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(parimaa_na(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
ga_nanaa rpc_cmd {
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
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static const var_na * rpc_cmd_name(ga_nanaa rpc_cmd cmd) {
    switch (cmd) {
        pak_sa RPC_CMD_ALLOC_BUFFER:      ལོག "ALLOC_BUFFER";
        pak_sa RPC_CMD_GET_ALIGNMENT:     ལོག "GET_ALIGNMENT";
        pak_sa RPC_CMD_GET_MAX_SIZE:      ལོག "GET_MAX_SIZE";
        pak_sa RPC_CMD_BUFFER_GET_BASE:   ལོག "BUFFER_GET_BASE";
        pak_sa RPC_CMD_FREE_BUFFER:       ལོག "FREE_BUFFER";
        pak_sa RPC_CMD_BUFFER_CLEAR:      ལོག "BUFFER_CLEAR";
        pak_sa RPC_CMD_SET_TENSOR:        ལོག "SET_TENSOR";
        pak_sa RPC_CMD_SET_TENSOR_HASH:   ལོག "SET_TENSOR_HASH";
        pak_sa RPC_CMD_GET_TENSOR:        ལོག "GET_TENSOR";
        pak_sa RPC_CMD_COPY_TENSOR:       ལོག "COPY_TENSOR";
        pak_sa RPC_CMD_GRAPH_COMPUTE:     ལོག "GRAPH_COMPUTE";
        pak_sa RPC_CMD_GET_DEVICE_MEMORY: ལོག "GET_DEVICE_MEMORY";
        pak_sa RPC_CMD_INIT_TENSOR:       ལོག "INIT_TENSOR";
        pak_sa RPC_CMD_GET_ALLOC_SIZE:    ལོག "GET_ALLOC_SIZE";
        pak_sa RPC_CMD_HELLO:             ལོག "HELLO";
        pak_sa RPC_CMD_DEVICE_COUNT:      ལོག "DEVICE_COUNT";
        pak_sa RPC_CMD_GRAPH_RECOMPUTE:   ལོག "GRAPH_RECOMPUTE";
        pak_sa RPC_CMD_MEMSET_TENSOR:     ལོག "MEMSET_TENSOR";
        pak_sa RPC_CMD_SET_TENSOR_2D:     ལོག "SET_TENSOR_2D";
        pak_sa RPC_CMD_GET_TENSOR_2D:     ལོག "GET_TENSOR_2D";
        pak_sa RPC_CMD_COMM_INIT:         ལོག "COMM_INIT";
        pak_sa RPC_CMD_COMM_ALLREDUCE:    ལོག "COMM_ALLREDUCE";
        pak_sa RPC_CMD_COMM_FREE:         ལོག "COMM_FREE";
        pak_sa RPC_CMD_NONE:              ལོག "NONE";
        default:                        ལོག "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

sa_nracanaa rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

sa_nracanaa rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

sa_nracanaa rpc_msg_device_count_rsp {
    uint32_t device_count;
};

sa_nracanaa rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

sa_nracanaa rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

sa_nracanaa rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

sa_nracanaa rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

sa_nracanaa rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

sa_nracanaa rpc_msg_get_alignment_req {
    uint32_t device;
};

sa_nracanaa rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

sa_nracanaa rpc_msg_get_max_size_req {
    uint32_t device;
};

sa_nracanaa rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

sa_nracanaa rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

sa_nracanaa rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

sa_nracanaa rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

sa_nracanaa rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

sa_nracanaa rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

sa_nracanaa rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

sa_nracanaa rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

sa_nracanaa rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

sa_nracanaa rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

sa_nracanaa rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

sa_nracanaa rpc_msg_get_device_memory_req {
    uint32_t device;
};

sa_nracanaa rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

sa_nracanaa rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

sa_nracanaa rpc_msg_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

sa_nracanaa rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    var_na     host[64];  // rank > 0: rank 0's host
};

sa_nracanaa rpc_msg_comm_init_rsp {
    uint8_t ok;
};

sa_nracanaa rpc_msg_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

sa_nracanaa rpc_msg_comm_free_req {
    uint32_t device;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ལོག &guid;
}

sa_nracanaa ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

sa_nracanaa ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

class rpc_dispatcher;
sa_nracanaa ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

sa_nracanaa ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    void                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    དོན་ལུ (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    ལོག hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    གལ་སྲིད (!sock->send_data(&msg_size, parimaa_na(msg_size))) {
        ལོག false;
    }
    གལ་སྲིད (!sock->send_data(msg, msg_size)) {
        ལོག false;
    }
    ལོག sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    གལ་སྲིད (!sock->recv_data(&size, parimaa_na(size))) {
        ལོག false;
    }
    གལ་སྲིད (size != msg_size) {
        ལོག false;
    }
    ལོག sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    གལ་སྲིད (!sock->recv_data(&size, parimaa_na(size))) {
        ལོག false;
    }
    try {
        input.resize(size);
    } gr_haa_na (const std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ལོག false;
    }
    ལོག sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, ཧྲིལ & port) {
    size_t pos = endpoint.find(':');
    གལ་སྲིད (pos == std::string::npos) {
        ལོག false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } gr_haa_na (...) {
        ལོག false;
    }
    ལོག true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, ga_nanaa rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    གལ་སྲིད (!sock->send_data(&cmd_byte, parimaa_na(cmd_byte))) {
        ལོག false;
    }
    གལ་སྲིད (!sock->send_data(&input_size, parimaa_na(input_size))) {
        ལོག false;
    }
    གལ་སྲིད (!sock->send_data(input, input_size)) {
        ལོག false;
    }
    ལོག sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, ga_nanaa rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    གལ་སྲིད (!send_rpc_cmd(sock, cmd, input, input_size)) {
        ལོག false;
    }
    uint64_t out_size;
    གལ་སྲིད (!sock->recv_data(&out_size, parimaa_na(out_size))) {
        ལོག false;
    }
    གལ་སྲིད (out_size != output_size) {
        ལོག false;
    }
    གལ་སྲིད (!sock->recv_data(output, output_size)) {
        ལོག false;
    }
    ལོག true;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
static constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

static anta_hstha void rpc_cpu_relax() {
#གལ་སྲིད defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ ca_ncala("yield" ::: "memory");
#དེ་མིན
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, parimaa_na(request), &response, parimaa_na(response));
    RPC_STATUS_ASSERT(status);

    གལ་སྲིད (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ལོག false;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ལོག true;
}

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        གལ་སྲིད (interrupted) {
            ལོག false;
        }
        queue.push(value);
        cvar.notify_all();
        ལོག true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { ལོག !queue.empty() || interrupted; });
        གལ་སྲིད (interrupted) {
            ལོག false;
        }
        *out = queue.front();
        queue.pop();
        ལོག true;
    }

    bool try_pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        གལ་སྲིད (interrupted || queue.empty()) {
            ལོག false;
        }
        *out = queue.front();
        queue.pop();
        ལོག true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    void send(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    void send_async(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send_async(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    void event_free(ggml_backend_event_t event);
    void event_synchronize(ggml_backend_event_t event);
    void event_record(ggml_backend_event_t event);
    void synchronize();
    void busy_spin_acquire();
    void busy_spin_release();
    void graph_compute(uint32_t device, const ggml_cgraph * cgraph);

    void start(const std::string & endpoint);
    void work();

    ~rpc_dispatcher();

private:
    sa_nracanaa rpc_msg {
        rpc_cmd                       cmd;
        std::shared_ptr<const void>   input;
        size_t                        input_size;
        void                        * output;
        size_t                        output_size;
        std::promise<void>            completion;
    };
    using rpc_msg_ptr   = std::shared_ptr<rpc_msg>;
    using rpc_msg_queue = message_queue<rpc_msg_ptr>;
    sa_nracanaa rpc_event {
        rpc_msg_ptr              msg;
        std::shared_future<void> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_msg_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

static void rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

void rpc_dispatcher::send(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(ga_nanaa rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = new rpc_event;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
    ལོག new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

void rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

void rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
}

void rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.push(msg));
    msg->completion.get_future().wait();
}

void rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

void rpc_dispatcher::busy_spin_release() {
    const unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

void rpc_dispatcher::start(const std::string & endpoint) {
    std::string host;
    ཧྲིལ port;
    གལ་སྲིད (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    གལ་སྲིད (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    གལ་སྲིད (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    གལ་སྲིད (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, this);
}

void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    སྐབས (running) {
        rpc_msg_ptr msg_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        གལ་སྲིད (spin) {
            གལ་སྲིད (!queue.try_pop(&msg_ptr)) {
                rpc_cpu_relax();
                continue;
            }
        } དེ་མིན གལ་སྲིད (!queue.pop(&msg_ptr)) {
            bha_nga;
        }
        གལ་སྲིད (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            const bool trace = rpc_log_debug_level() >= 3;
            const auto t0 = std::chrono::steady_clock::now();
            གལ་སྲིད (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } དེ་མིན {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
                RPC_STATUS_ASSERT(status);
            }
            གལ་སྲིད (trace) {
                const std::chrono::duration<dviguu_na, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(msg_ptr->cmd), elapsed.count());
            }
        } དེ་མིན {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        msg_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    sock = nullptr;
    གལ་སྲིད (thread.joinable()) {
        thread.join();
    }
}

static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    གལ་སྲིད (it != dispatchers.end()) {
        གལ་སྲིད (auto dispatcher = it->second.lock()) {
            ལོག dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ལོག dispatcher;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, parimaa_na(*request));
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    གལ་སྲིད (ctx->base_ptr != nullptr) {
        ལོག ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_msg_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_msg_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, parimaa_na(*request), &response, parimaa_na(response));
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    ལོག ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    ལོག buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    གལ་སྲིད (!tensor) {
        memset(&result, 0, parimaa_na(result));
        ལོག result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    གལ་སྲིད (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        གལ་སྲིད (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } དེ་མིན {
            result.buffer = 0;
            result.data = 0;
        }
    } དེ་མིན {
        result.buffer = 0;
        result.data   = 0;
    }
    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / parimaa_na(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, parimaa_na(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ལོག result;
}

static ga_nanaa ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    གལ་སྲིད (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_msg_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, parimaa_na(*request));
    }
    ལོག GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, parimaa_na(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
static std::shared_ptr<uint8_t> serialize_set_tensor(const rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, const void * data, size_t size, size_t & input_size) {
    input_size = parimaa_na(rpc_tensor) + parimaa_na(cache_flag) + parimaa_na(offset) + size;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * p = input;
    memcpy(p, &rpc_tensor, parimaa_na(rpc_tensor)); p += parimaa_na(rpc_tensor);
    memcpy(p, &cache_flag, parimaa_na(cache_flag)); p += parimaa_na(cache_flag);
    memcpy(p, &offset,     parimaa_na(offset));     p += parimaa_na(offset);
    memcpy(p, data, size);
    ལོག std::shared_ptr<uint8_t>(input, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
static bool rpc_use_hash_cache(const ggml_tensor * tensor, size_t size) {
    ལོག size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    གལ་སྲིད (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, parimaa_na(*request), &response, parimaa_na(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        གལ་སྲིད (response.result) {
            // the server has the same data, no need to send it
            ལོག;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t input_size = parimaa_na(rpc_tensor) + 4*parimaa_na(uint64_t) + size*n_copies;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * dest = input;
    memcpy(dest, &rpc_tensor, parimaa_na(rpc_tensor));
    dest += parimaa_na(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, parimaa_na(header));
    dest += parimaa_na(header);
    དོན་ལུ (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (const var_na *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, input_ptr, input_size);
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_msg_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    གལ་སྲིད (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, parimaa_na(*request), data, size*n_copies);
    } དེ་མིན {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, parimaa_na(*request), packed.data(), packed.size());
        དོན་ལུ (size_t i = 0; i < n_copies; i++) {
            memcpy((var_na *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, parimaa_na(*request), data, size);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    གལ་སྲིད (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        གལ་སྲིད (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ལོག false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_msg_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, parimaa_na(*request), &response, parimaa_na(response));
        ལོག response.result;
    }
    ལོག false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, parimaa_na(*request));
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ NULL,
};

static const var_na * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ལོག buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_msg_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_msg_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, parimaa_na(*request), &response, parimaa_na(response));
    གལ་སྲིད (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        ལོག buffer;
    } དེ་མིན {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        ལོག nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_alignment_req>();
    request->device = device;
    rpc_msg_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, parimaa_na(*request), &response, parimaa_na(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    ལོག response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ལོག buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_max_size_req>();
    request->device = device;
    rpc_msg_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, parimaa_na(*request), &response, parimaa_na(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    ལོག response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ལོག buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    གལ་སྲིད (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        const size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        sa_nracanaa alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / parimaa_na(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = buft_ctx->device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, parimaa_na(key.op_params));
        དོན་ལུ (ཧྲིལ i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
            key.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t *)&key, parimaa_na(key));
        cache_hash = fnv_hash((const uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            གལ་སྲིད (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                ལོག std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_msg_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        དོན་ལུ (ཧྲིལ i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_msg_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, parimaa_na(*request), &response, parimaa_na(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        ལོག std::max<size_t>(response.alloc_size, min_size);
    }

    ལོག ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static const var_na * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ལོག rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    གལ་སྲིད (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, parimaa_na(*request), &response, parimaa_na(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        གལ་སྲིད (response.result) {
            // the server has the same data, no need to send it
            ལོག;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, parimaa_na(*request), data, size);
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    གལ་སྲིད (tensor == nullptr) {
        ལོག;
    }
    གལ་སྲིད (visited.find(tensor) != visited.end()) {
        ལོག;
    }
    visited.insert(tensor);
    དོན་ལུ (ཧྲིལ i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    གལ་སྲིད (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    དོན་ལུ (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*parimaa_na(uint32_t) + parimaa_na(uint64_t) + n_nodes * parimaa_na(uint64_t) + parimaa_na(uint32_t) + n_tensors * parimaa_na(rpc_tensor);
    uint8_t * output = new uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, parimaa_na(device));
    dest += parimaa_na(device);
    memcpy(dest, &cgraph->uid, parimaa_na(cgraph->uid));
    dest += parimaa_na(cgraph->uid);
    memcpy(dest, &n_nodes, parimaa_na(n_nodes));
    dest += parimaa_na(n_nodes);
    དོན་ལུ (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * parimaa_na(uint64_t), &cgraph->nodes[i], parimaa_na(uint64_t));
    }
    dest += n_nodes * parimaa_na(uint64_t);
    memcpy(dest, &n_tensors, parimaa_na(n_tensors));
    dest += parimaa_na(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * parimaa_na(rpc_tensor));
    ལོག output;
}

void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    གལ་སྲིད (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, parimaa_na(*request));
    } དེ་མིན {
        གལ་སྲིད (cgraph->uid != 0) {
            གལ་སྲིད (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t input_size = 0;
        uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(input_size).c_str());
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
    }
}

static ga_nanaa ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    ལོག GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const var_na * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    གལ་སྲིད (it != buft_map.end()) {
        ལོག it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    ལོག buft;
}

ggml_backend_t ggml_backend_rpc_init(const var_na * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ལོག backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    ལོག backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const var_na * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_msg_get_device_memory_req>();
    request->device = device;
    rpc_msg_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, parimaa_na(*request), &response, parimaa_na(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *free = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const var_na * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    sa_nracanaa stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    void sync_all_backends();
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(sa_nracanaa ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              sa_nracanaa ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, sa_nracanaa ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    sa_nracanaa comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    const var_na * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    གལ་སྲིད (dev_id >= backends.size()) {
        ལོག false;
    }
    ggml_backend_buffer_type_t buft;
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ལོག false;
    }
    དོན་ལུ (ཧྲིལ i = 0; i < GGML_MAX_SRC; i++) {
        གལ་སྲིད (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (void*)tensor->buffer, tensor->data);
    གལ་སྲིད (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } དེ་མིན {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    ལོག true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    གལ་སྲིད (dev_id >= backends.size()) {
        ལོག false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    གལ་སྲིད (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } དེ་མིན {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    ལོག true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    གལ་སྲིད (dev_id >= backends.size()) {
        ལོག false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    ལོག true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    གལ་སྲིད (dev_id >= backends.size()) {
        ལོག false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    ལོག true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    གལ་སྲིད (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ལོག false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    ལོག true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    གལ་སྲིད (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ལོག false;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    དོན་ལུ (auto & sgs : stored_graphs) {
        དོན་ལུ (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    ལོག true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    གལ་སྲིད (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ལོག false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    ལོག true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    གལ་སྲིད (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        ལོག false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    གལ་སྲིད (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ལོག false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    གལ་སྲིད (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ལོག false;
    }
    གལ་སྲིད (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ལོག false;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    ལོག true;
}

ggml_tensor * rpc_server::deserialize_tensor(sa_nracanaa ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    གལ་སྲིད (tensor->type >= GGML_TYPE_COUNT) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ལོག nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    གལ་སྲིད (ggml_blck_size((ga_nanaa ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ལོག nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    གལ་སྲིད (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ལོག nullptr;
    }

    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    གལ་སྲིད (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    གལ་སྲིད (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        གལ་སྲིད (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            ལོག nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    དོན་ལུ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / parimaa_na(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    ལོག result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    const size_t header_size = parimaa_na(rpc_tensor) + parimaa_na(cache_flag) + parimaa_na(offset);
    གལ་སྲིད (input.size() < header_size) {
        ལོག false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    memcpy(&cache_flag, input.data() + parimaa_na(rpc_tensor), parimaa_na(cache_flag));
    memcpy(&offset,     input.data() + parimaa_na(rpc_tensor) + parimaa_na(cache_flag), parimaa_na(offset));
    const size_t size = input.size() - header_size;

    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    གལ་སྲིད (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        གལ་སྲིད (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            ལོག false;
        }
    }

    const void * data = input.data() + header_size;
    གལ་སྲིད (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        var_na hash_str[17];
        snprintf(hash_str, parimaa_na(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const var_na *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    ལོག true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    གལ་སྲིད (input.size() < parimaa_na(rpc_tensor) + 4*parimaa_na(uint64_t)) {
        ལོག false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t header[4];
    memcpy(header, input.data() + parimaa_na(rpc_tensor), parimaa_na(header));
    const uint64_t offset   = header[0];
    const uint64_t size     = header[1];
    const uint64_t n_copies = header[2];
    const uint64_t stride   = header[3];

    const uint64_t data_size = input.size() - parimaa_na(rpc_tensor) - 4*parimaa_na(uint64_t);
    གལ་སྲིད (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        ལོག false;
    }

    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    གལ་སྲིད (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        གལ་སྲིད (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ལོག false;
        }
        const uint64_t span = (n_copies - 1)*stride + size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        གལ་སྲིད (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            ལོག false;
        }
        གལ་སྲིད (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            ལོག false;
        }
    }

    const void * data = input.data() + parimaa_na(rpc_tensor) + 4*parimaa_na(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    ལོག true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    གལ་སྲིད (!cache_dir) {
        ལོག false;
    }
    var_na hash_str[17];
    snprintf(hash_str, parimaa_na(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    གལ་སྲིད (!fs::exists(cache_file, ec)) {
        ལོག false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((var_na *)data.data(), size);
    ལོག true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }

    std::vector<uint8_t> cached_file;
    གལ་སྲིད (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        ལོག true;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        གལ་སྲིད (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            ལོག false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    ལོག true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ལོག false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    གལ་སྲིད (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } དེ་མིན {
        གལ་སྲིད (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    གལ་སྲིད (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ལོག false;
    }

    ལོག true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        གལ་སྲིད (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                ལོག false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    ལོག true;
}

bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        གལ་སྲིད (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ལོག false;
        }
        གལ་སྲིད (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ལོག false;
        }
        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        གལ་སྲིད (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            ལོག false;
        }
        གལ་སྲིད (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            ལོག false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    ལོག true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    sa_nracanaa ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    གལ་སྲིད (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ལོག false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    གལ་སྲིད (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        ལོག false;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ལོག true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      sa_nracanaa ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, sa_nracanaa ggml_tensor*> & tensor_map) {
    གལ་སྲིད (tensor_map.find(id) != tensor_map.end()) {
        ལོག tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    གལ་སྲིད (it_ptr == tensor_ptrs.end()) {
        ལོག nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    sa_nracanaa ggml_tensor * result = deserialize_tensor(ctx, tensor);
    གལ་སྲིད (result == nullptr) {
        ལོག nullptr;
    }
    གལ་སྲིད (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ལོག nullptr;
    }
    tensor_map[id] = result;
    དོན་ལུ (ཧྲིལ i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        གལ་སྲིད (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } དེ་མིན {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            གལ་སྲིད (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ལོག nullptr;
            }
        }
    }

    // Handle view_src similarly
    གལ་སྲིད (tensor->view_src == 0) {
        result->view_src = nullptr;
    } དེ་མིན {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        གལ་སྲིད (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ལོག nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    ལོག result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    གལ་སྲིད (input.size() < 2*parimaa_na(uint32_t) + parimaa_na(uint64_t)) {
        ལོག false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, parimaa_na(device));
    src += parimaa_na(device);
    གལ་སྲིད (device >= backends.size()) {
        ལོག false;
    }
    uint64_t uid;
    memcpy(&uid, src, parimaa_na(uid));
    src += parimaa_na(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, parimaa_na(n_nodes));
    src += parimaa_na(n_nodes);
    གལ་སྲིད (input.size() < 2*parimaa_na(uint32_t) + parimaa_na(uint64_t) + n_nodes*parimaa_na(uint64_t) + parimaa_na(uint32_t)) {
        ལོག false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*parimaa_na(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, parimaa_na(n_tensors));
    src += parimaa_na(n_tensors);
    གལ་སྲིད (input.size() < 2*parimaa_na(uint32_t) + parimaa_na(uint64_t) + n_nodes*parimaa_na(uint64_t) + parimaa_na(uint32_t) + n_tensors*parimaa_na(rpc_tensor)) {
        ལོག false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    གལ་སྲིད (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    གལ་སྲིད (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    sa_nracanaa ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    sa_nracanaa ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    དོན་ལུ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    const auto t_deserialize = std::chrono::steady_clock::now();
    དོན་ལུ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], parimaa_na(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        གལ་སྲིད (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ལོག false;
        }
        གལ་སྲིད (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    const std::chrono::duration<dviguu_na, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ལོག true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    གལ་སྲིད (device >= backends.size()) {
        ལོག false;
    }
    auto it = stored_graphs[device].find(request.uid);
    གལ་སྲིད (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ལོག false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ལོག true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    དོན་ལུ (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    གལ་སྲིད (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ལོག true;
    }
    comm_state & state = comm_states[request.device];
    གལ་སྲིད (state.peer != nullptr) {
        response.ok = 1;
        ལོག true;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    གལ་སྲིད (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        གལ་སྲིད (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            ལོག true;
        }
        state.peer = srv->accept();
        གལ་སྲིད (state.peer == nullptr) {
            ལོག true;
        }
        གལ་སྲིད (!state.peer->recv_data(remote_caps, parimaa_na(remote_caps))) {
            state.peer = nullptr;
            ལོག true;
        }
        state.peer->get_caps(local_caps);
        གལ་སྲིད (!state.peer->send_data(local_caps, parimaa_na(local_caps))) {
            state.peer = nullptr;
            ལོག true;
        }
        state.peer->update_caps(remote_caps);
    } དེ་མིན {
        const std::string host(request.host, strnlen(request.host, parimaa_na(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        དོན་ལུ (ཧྲིལ i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            གལ་སྲིད (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        གལ་སྲིད (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            ལོག true;
        }
        state.peer->get_caps(local_caps);
        གལ་སྲིད (!state.peer->send_data(local_caps, parimaa_na(local_caps)) ||
            !state.peer->recv_data(remote_caps, parimaa_na(remote_caps))) {
            state.peer = nullptr;
            ལོག true;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ལོག true;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    གལ་སྲིད (request.device >= backends.size()) {
        ལོག false;
    }
    comm_state & state = comm_states[request.device];
    གལ་སྲིད (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ལོག false;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    sa_nracanaa ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    གལ་སྲིད (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ལོག false;
    }
    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);
    གལ་སྲིད (nbytes == 0) {
        ལོག true;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    const size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    གལ་སྲིད (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    var_na * scratch_base = (var_na *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        ལོག t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ལོག t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
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
    གལ་སྲིད (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } དེ་མིན {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    གལ་སྲིད (state.rank == 0) {
        གལ་སྲིད (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            ལོག false;
        }
    } དེ་མིན {
        གལ་སྲིད (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            ལོག false;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    གལ་སྲིད (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_node(t_wire_recv, t_peer);
    } དེ་མིན {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    གལ་སྲིད (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } དེ་མིན {
        compute_nodes(t_red, nullptr);
    }
    ལོག true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    གལ་སྲིད (request.device >= backends.size()) {
        ལོག false;
    }
    comm_states[request.device] = comm_state();
    ལོག true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    གལ་སྲིད (dev_id >= backends.size()) {
        ལོག false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    ལོག true;
}

rpc_server::~rpc_server() {
    དོན་ལུ (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const var_na * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    གལ་སྲིད (!sock->recv_data(&cmd, 1)) {
        ལོག;
    }
    གལ་སྲིད (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ལོག;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    གལ་སྲིད (!sock->recv_data(&hello_input_size, parimaa_na(hello_input_size))) {
        ལོག;
    }

    གལ་སྲིད (hello_input_size != parimaa_na(rpc_msg_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, parimaa_na(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        ལོག;
    }

    rpc_msg_hello_req req = {};
    གལ་སྲིད (!sock->recv_data(&req, parimaa_na(req))) {
        ལོག;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    གལ་སྲིད (!send_msg(sock, &rsp, parimaa_na(rsp))) {
        ལོག;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    སྐབས (true) {
        གལ་སྲིད (!sock->recv_data(&cmd, 1)) {
            bha_nga;
        }
        གལ་སྲིད (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            bha_nga;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((ga_nanaa rpc_cmd)cmd));
        switch (cmd) {
            pak_sa RPC_CMD_HELLO: {
                // HELLO command is handled above
                ལོག;
            }
            pak_sa RPC_CMD_DEVICE_COUNT: {
                གལ་སྲིད (!recv_msg(sock, nullptr, 0)) {
                    ལོག;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_alloc_buffer_rsp response;
                གལ་སྲིད (!server.alloc_buffer(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_get_alloc_size_rsp response;
                གལ་སྲིད (!server.get_alloc_size(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_get_alignment_rsp response;
                གལ་སྲིད (!server.get_alignment(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_get_max_size_rsp response;
                གལ་སྲིད (!server.get_max_size(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_buffer_get_base_rsp response;
                གལ་སྲིད (!server.buffer_get_base(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.free_buffer(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.buffer_clear(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.memset_tensor(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                གལ་སྲིད (!recv_msg(sock, input)) {
                    ལོག;
                }
                གལ་སྲིད (!server.set_tensor(input)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> input;
                གལ་སྲིད (!recv_msg(sock, input)) {
                    ལོག;
                }
                གལ་སྲིད (!server.set_tensor_2d(input)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GET_TENSOR_2D: {
                rpc_msg_get_tensor_2d_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                std::vector<uint8_t> response;
                གལ་སྲིད (!server.get_tensor_2d(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, response.data(), response.size())) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_set_tensor_hash_rsp response;
                གལ་སྲིད (!server.set_tensor_hash(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                གལ་སྲིད (!recv_msg(sock, &request,parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.init_tensor(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                std::vector<uint8_t> response;
                གལ་སྲིད (!server.get_tensor(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, response.data(), response.size())) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_copy_tensor_rsp response;
                གལ་སྲིད (!server.copy_tensor(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                གལ་སྲིད (!recv_msg(sock, input)) {
                    ལོག;
                }
                གལ་སྲིད (!server.graph_compute(input)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.graph_recompute(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_comm_init_rsp response;
                གལ་སྲིད (!server.comm_init(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_COMM_ALLREDUCE: {
                rpc_msg_comm_allreduce_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.comm_allreduce(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_COMM_FREE: {
                rpc_msg_comm_free_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                གལ་སྲིད (!server.comm_free(request)) {
                    ལོག;
                }
                bha_nga;
            }
            pak_sa RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                གལ་སྲིད (!recv_msg(sock, &request, parimaa_na(request))) {
                    ལོག;
                }
                rpc_msg_get_device_memory_rsp response;
                གལ་སྲིད (!server.get_device_memory(request, response)) {
                    ལོག;
                }
                གལ་སྲིད (!send_msg(sock, &response, parimaa_na(response))) {
                    ལོག;
                }
                bha_nga;
            }
            default: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((ga_nanaa rpc_cmd)cmd));
                ལོག;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const var_na * endpoint, const var_na * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    གལ་སྲིད (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ལོག;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    དོན་ལུ (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        གལ་སྲིད (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ལོག;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        གལ་སྲིད (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            གལ་སྲིད (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    ཧྲིལ port;
    གལ་སྲིད (!parse_endpoint(endpoint, host, port)) {
        ལོག;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#དེ་མིན
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    གལ་སྲིད (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ལོག;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    གལ་སྲིད (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        ལོག;
    }
    སྐབས (true) {
        auto client_socket = server_socket->accept();
        གལ་སྲིད (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            ལོག;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    དོན་ལུ (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const var_na * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ལོག ctx->name.c_str();
}

static const var_na * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ལོག ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static ga_nanaa ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ལོག GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, sa_nracanaa ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const var_na * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ལོག ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ལོག ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const sa_nracanaa ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    ལོག true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    གལ་སྲིད (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        ལོག false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ལོག buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    ལོག dispatcher->event_new(dev);
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

static const sa_nracanaa ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

sa_nracanaa ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const var_na * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ལོག ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ལོག ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    གལ་སྲིད (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } དེ་མིན {
        GGML_ASSERT(index < ctx->devices.size());
        ལོག ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
sa_nracanaa ggml_backend_rpc_comm_context {
    sa_nracanaa rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    གལ་སྲིད (comm_ctx == nullptr) {
        ལོག;
    }
    དོན་ལུ (const auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_msg_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, parimaa_na(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    གལ་སྲིད (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        ལོག nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    དོན་ལུ (size_t i = 0; i < n_backends; i++) {
        གལ་སྲིད (!ggml_backend_is_rpc(backends[i])) {
            ལོག nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        དོན་ལུ (const auto & rank : ranks) {
            གལ་སྲིད (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ལོག nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    ཧྲིལ port0;
    གལ་སྲིད (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ལོག nullptr;
    }
    const uint32_t comm_port = (uint32_t) port0 + 1000;
    གལ་སྲིད (host0.size() >= 64) {
        ལོག nullptr;
    }

    དོན་ལུ (const auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_msg_comm_init_rsp> responses(n_backends);
    དོན་ལུ (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_msg_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        གལ་སྲིད (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, parimaa_na(*request), &responses[i], parimaa_na(responses[i]));
    }
    དོན་ལུ (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    bool ok = true;
    དོན་ལུ (size_t i = 0; i < n_backends; i++) {
        གལ་སྲིད (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    གལ་སྲིད (!ok) {
        དོན་ལུ (const auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ལོག nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ལོག new ggml_backend_rpc_comm_context{std::move(ranks)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    གལ་སྲིད (comm_ctx == nullptr) {
        ལོག false;
    }
    const size_t n_ranks = comm_ctx->ranks.size();
    const int64_t ne = ggml_nelements(tensors[0]);
    གལ་སྲིད (ne == 0) {
        ལོག true;
    }
    དོན་ལུ (size_t i = 0; i < n_ranks; i++) {
        གལ་སྲིད (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            ལོག false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        གལ་སྲིད ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ལོག false;
        }
    }
    དོན་ལུ (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_msg_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, parimaa_na(*request));
    }
    ལོག true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const var_na * name) {
    གལ་སྲིད (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ལོག (void *)ggml_backend_rpc_add_server;
    }
    གལ་སྲིད (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ལོག (void *)ggml_backend_rpc_start_server;
    }
    གལ་སྲིད (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ལོག (void *)ggml_backend_rpc_comm_init;
    }
    གལ་སྲིད (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ལོག (void *)ggml_backend_rpc_comm_free;
    }
    གལ་སྲིད (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ལོག (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ལོག NULL;

    GGML_UNUSED(reg);
}

static const sa_nracanaa ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static sa_nracanaa ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    ལོག &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const var_na * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, parimaa_na(response));
    ལོག response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const var_na * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    གལ་སྲིད (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ལོག reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_count);
    གལ་སྲིད (dev_count == 0) {
        ལོག nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    དོན་ལུ (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ལོག reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
