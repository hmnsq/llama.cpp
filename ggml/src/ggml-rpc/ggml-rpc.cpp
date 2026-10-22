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

نامخلا fs = std::filesystem;

// macro for nicer error messages on server crash
#تعریف_کرو RPC_حالت_ASSERT(x) اگر (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#ہدایت pack(ڈالو, 1)
// ggml_tensor is serialized into rpc_tensor
ساخت rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t بفر;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t ڈیٹا;
    حرف name[GGML_MAX_NAME];

    int32_t use_گنتی;
};

static_assert(حجم(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
شمار rpc_cmd {
    RPC_CMD_ALLOC_بفر = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_سائز,
    RPC_CMD_بفر_GET_بنیاد,
    RPC_CMD_FREE_بفر,
    RPC_CMD_بفر_CLEAR,
    RPC_CMD_سیٹ_TENSOR,
    RPC_CMD_سیٹ_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_گراف_COMPUTE,
    RPC_CMD_GET_DEVICE_حافظہ,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_سائز,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_گنتی,
    RPC_CMD_گراف_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_سیٹ_TENSOR_2_D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_گنتی,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ساکن ثابت حرف * rpc_cmd_name(شمار rpc_cmd cmd) {
    انتخاب (cmd) {
        حالت RPC_CMD_ALLOC_بفر:      واپس "ALLOC_BUFFER";
        حالت RPC_CMD_GET_ALIGNMENT:     واپس "GET_ALIGNMENT";
        حالت RPC_CMD_GET_MAX_سائز:      واپس "GET_MAX_SIZE";
        حالت RPC_CMD_بفر_GET_بنیاد:   واپس "BUFFER_GET_BASE";
        حالت RPC_CMD_FREE_بفر:       واپس "FREE_BUFFER";
        حالت RPC_CMD_بفر_CLEAR:      واپس "BUFFER_CLEAR";
        حالت RPC_CMD_سیٹ_TENSOR:        واپس "SET_TENSOR";
        حالت RPC_CMD_سیٹ_TENSOR_HASH:   واپس "SET_TENSOR_HASH";
        حالت RPC_CMD_GET_TENSOR:        واپس "GET_TENSOR";
        حالت RPC_CMD_COPY_TENSOR:       واپس "COPY_TENSOR";
        حالت RPC_CMD_گراف_COMPUTE:     واپس "GRAPH_COMPUTE";
        حالت RPC_CMD_GET_DEVICE_حافظہ: واپس "GET_DEVICE_MEMORY";
        حالت RPC_CMD_INIT_TENSOR:       واپس "INIT_TENSOR";
        حالت RPC_CMD_GET_ALLOC_سائز:    واپس "GET_ALLOC_SIZE";
        حالت RPC_CMD_HELLO:             واپس "HELLO";
        حالت RPC_CMD_DEVICE_گنتی:      واپس "DEVICE_COUNT";
        حالت RPC_CMD_گراف_RECOMPUTE:   واپس "GRAPH_RECOMPUTE";
        حالت RPC_CMD_MEMSET_TENSOR:     واپس "MEMSET_TENSOR";
        حالت RPC_CMD_سیٹ_TENSOR_2_D:     واپس "SET_TENSOR_2D";
        حالت RPC_CMD_GET_TENSOR_2D:     واپس "GET_TENSOR_2D";
        حالت RPC_CMD_COMM_INIT:         واپس "COMM_INIT";
        حالت RPC_CMD_COMM_ALLREDUCE:    واپس "COMM_ALLREDUCE";
        حالت RPC_CMD_COMM_FREE:         واپس "COMM_FREE";
        حالت RPC_CMD_NONE:              واپس "NONE";
        طےشدہ:                        واپس "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ثابت size_t HASH_حد = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ثابت size_t گراف_CACHE_MAX = 1024;

ساخت rpc_پیغام_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_سائز];
};

ساخت rpc_پیغام_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_سائز];
};

ساخت rpc_پیغام_device_گنتی_rsp {
    uint32_t device_گنتی;
};

ساخت rpc_پیغام_get_alloc_سائز_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ساخت rpc_پیغام_get_alloc_سائز_rsp {
    uint64_t alloc_سائز;
};

ساخت rpc_پیغام_init_tensor_req {
    rpc_tensor tensor;
};

ساخت rpc_پیغام_alloc_بفر_req {
    uint32_t device;
    uint64_t size;
};

ساخت rpc_پیغام_alloc_بفر_rsp {
    uint64_t remote_ptr;
    uint64_t remote_سائز;
};

ساخت rpc_پیغام_get_alignment_req {
    uint32_t device;
};

ساخت rpc_پیغام_get_alignment_rsp {
    uint64_t alignment;
};

ساخت rpc_پیغام_get_max_سائز_req {
    uint32_t device;
};

ساخت rpc_پیغام_get_max_سائز_rsp {
    uint64_t max_size;
};

ساخت rpc_پیغام_بفر_get_بنیاد_req {
    uint64_t remote_ptr;
};

ساخت rpc_پیغام_بفر_get_بنیاد_rsp {
    uint64_t بنیاد_ptr;
};

ساخت rpc_پیغام_free_بفر_req {
    uint64_t remote_ptr;
};

ساخت rpc_پیغام_بفر_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ساخت rpc_پیغام_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t انحراف;
    uint64_t size;
    uint8_t value;
};

ساخت rpc_پیغام_سیٹ_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t انحراف;
    uint64_t hash;
};

ساخت rpc_پیغام_سیٹ_tensor_hash_rsp {
    uint8_t نتیجہ;
};

ساخت rpc_پیغام_get_tensor_req {
    rpc_tensor tensor;
    uint64_t انحراف;
    uint64_t size;
};

ساخت rpc_پیغام_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ساخت rpc_پیغام_copy_tensor_rsp {
    uint8_t نتیجہ;
};

ساخت rpc_پیغام_get_device_حافظہ_req {
    uint32_t device;
};

ساخت rpc_پیغام_get_device_حافظہ_rsp {
    uint64_t free_mem;
    uint64_t کل_mem;
};

ساخت rpc_پیغام_گراف_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ساخت rpc_پیغام_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t انحراف;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ساخت rpc_پیغام_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t پورٹ;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    حرف     host[64];  // rank > 0: rank 0's host
};

ساخت rpc_پیغام_comm_init_rsp {
    uint8_t ok;
};

ساخت rpc_پیغام_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ساخت rpc_پیغام_comm_free_req {
    uint32_t device;
};

#ہدایت pack(نکالو)

// RPC data structures

ساکن ggml_guid_t ggml_backend_rpc_guid() {
    ساکن ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    واپس &guid;
}

ساخت ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ساخت ggml_backend_rpc_بفر_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

جماعت rpc_dispatcher;
ساخت ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ساخت ggml_backend_rpc_بفر_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    خالی                            * بنیاد_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ساکن uint64_t fnv_hash(ثابت uint8_t * ڈیٹا, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ثابت uint64_t fnv_مفرد = 0x100000001b3ULL;

    برائے (size_t i = 0; i < len; ++i) {
        hash ^= ڈیٹا[i];
        hash *= fnv_مفرد;
    }
    واپس hash;
}

ساکن منطقی send_پیغام(socket_ptr sock, ثابت خالی * پیغام_2, size_t پیغام_سائز) {
    اگر (!sock->send_ڈیٹا(&پیغام_سائز, حجم(پیغام_سائز))) {
        واپس جھوٹ;
    }
    اگر (!sock->send_ڈیٹا(پیغام_2, پیغام_سائز)) {
        واپس جھوٹ;
    }
    واپس sock->flush();
}

ساکن منطقی recv_پیغام(socket_ptr sock, خالی * پیغام_2, size_t پیغام_سائز) {
    uint64_t size;
    اگر (!sock->recv_ڈیٹا(&size, حجم(size))) {
        واپس جھوٹ;
    }
    اگر (size != پیغام_سائز) {
        واپس جھوٹ;
    }
    واپس sock->recv_ڈیٹا(پیغام_2, پیغام_سائز);
}

ساکن منطقی recv_پیغام(socket_ptr sock, std::vector<uint8_t> & داخلہ) {
    uint64_t size;
    اگر (!sock->recv_ڈیٹا(&size, حجم(size))) {
        واپس جھوٹ;
    }
    کوشش {
        داخلہ.resize(size);
    } پکڑو (ثابت std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        واپس جھوٹ;
    }
    واپس sock->recv_ڈیٹا(داخلہ.ڈیٹا(), size);
}

ساکن منطقی parse_endpoint(ثابت std::string & endpoint, std::string & host, صحیح & پورٹ) {
    size_t pos = endpoint.find(':');
    اگر (pos == std::string::npos) {
        واپس جھوٹ;
    }
    host = endpoint.substr(0, pos);
    کوشش {
        پورٹ = std::stoi(endpoint.substr(pos + 1));
    } پکڑو (...) {
        واپس جھوٹ;
    }
    واپس سچ;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ساکن منطقی send_rpc_cmd(socket_ptr sock, شمار rpc_cmd cmd, ثابت خالی * داخلہ, size_t داخلہ_سائز) {
    uint8_t cmd_byte = cmd;
    اگر (!sock->send_ڈیٹا(&cmd_byte, حجم(cmd_byte))) {
        واپس جھوٹ;
    }
    اگر (!sock->send_ڈیٹا(&داخلہ_سائز, حجم(داخلہ_سائز))) {
        واپس جھوٹ;
    }
    اگر (!sock->send_ڈیٹا(داخلہ, داخلہ_سائز)) {
        واپس جھوٹ;
    }
    واپس sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ساکن منطقی send_rpc_cmd(socket_ptr sock, شمار rpc_cmd cmd, ثابت خالی * داخلہ, size_t داخلہ_سائز, خالی * output, size_t output_سائز) {
    اگر (!send_rpc_cmd(sock, cmd, داخلہ, داخلہ_سائز)) {
        واپس جھوٹ;
    }
    uint64_t برآمد_سائز;
    اگر (!sock->recv_ڈیٹا(&برآمد_سائز, حجم(برآمد_سائز))) {
        واپس جھوٹ;
    }
    اگر (برآمد_سائز != output_سائز) {
        واپس جھوٹ;
    }
    اگر (!sock->recv_ڈیٹا(output, output_سائز)) {
        واپس جھوٹ;
    }
    واپس سچ;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ساکن constexpr خودکار RPC_BUSY_SPIN_IDLE_وقت = std::chrono::milliseconds(100);

ساکن اندرونی خالی rpc_cpu_relax() {
#اگر defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ متلون("yield" ::: "memory");
#ورنہ
    std::this_thread::yield();
#اگر_ختم
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ساکن منطقی negotiate_hello(ثابت std::shared_ptr<socket_t> & sock) {
    rpc_پیغام_hello_req request = {};
    rpc_پیغام_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    منطقی حالت_2 = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, حجم(request), &response, حجم(response));
    RPC_حالت_ASSERT(حالت_2);

    اگر (response.major != RPC_PROTO_MAJOR_نسخہ || response.minor > RPC_PROTO_MINOR_نسخہ) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        واپس جھوٹ;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    واپس سچ;
}

سانچہ <قسم_نام T>
جماعت پیغام_قطار {
عوامی:
    پیغام_قطار() {}

    منطقی ڈالو(ثابت T &value) {
        std::unique_lock<std::mutex> تالا(mutex);
        اگر (interrupted) {
            واپس جھوٹ;
        }
        queue.ڈالو(value);
        cvar.notify_all();
        واپس سچ;
    }

    منطقی نکالو(T* برآمد) {
        std::unique_lock<std::mutex> تالا(mutex);
        cvar.wait(تالا, [یہ] { واپس !queue.خالی_خانے() || interrupted; });
        اگر (interrupted) {
            واپس جھوٹ;
        }
        *برآمد = queue.front();
        queue.نکالو();
        واپس سچ;
    }

    منطقی try_نکالو(T* برآمد) {
        std::unique_lock<std::mutex> تالا(mutex);
        اگر (interrupted || queue.خالی_خانے()) {
            واپس جھوٹ;
        }
        *برآمد = queue.front();
        queue.نکالو();
        واپس سچ;
    }

    خالی مداخلت() {
        std::unique_lock<std::mutex> تالا(mutex);
        interrupted = سچ;
        تالا.unlock();
        cvar.notify_all();
    }

نجی:
    منطقی interrupted = جھوٹ;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

جماعت rpc_dispatcher {
عوامی:
    rpc_dispatcher() {
    }

    خالی send(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز);
    خالی send(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز, خالی * output, size_t output_سائز);
    خالی send_async(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز);
    خالی send_async(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز, خالی * output, size_t output_سائز);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    خالی event_free(ggml_backend_event_t event);
    خالی event_synchronize(ggml_backend_event_t event);
    خالی event_ریکارڈ(ggml_backend_event_t event);
    خالی synchronize();
    خالی busy_spin_acquire();
    خالی busy_spin_release();
    خالی گراف_compute(uint32_t device, ثابت ggml_cgraph * cgraph);

    خالی start(ثابت std::string & endpoint);
    خالی work();

    ~rpc_dispatcher();

نجی:
    ساخت rpc_پیغام {
        rpc_cmd                       cmd;
        std::shared_ptr<ثابت خالی>   داخلہ;
        size_t                        داخلہ_سائز;
        خالی                        * output;
        size_t                        output_سائز;
        std::promise<خالی>            completion;
    };
    استعمال rpc_پیغام_ptr   = std::shared_ptr<rpc_پیغام>;
    استعمال rpc_پیغام_قطار = پیغام_قطار<rpc_پیغام_ptr>;
    ساخت rpc_event {
        rpc_پیغام_ptr              پیغام_2;
        std::shared_future<خالی> sf;
    };
    std::mutex گراف_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> گراف_uids;
    rpc_پیغام_قطار    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ساکن خالی rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

خالی rpc_dispatcher::send(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز) {
    خودکار پیغام_2 = std::make_shared<rpc_پیغام>();
    پیغام_2->cmd = cmd;
    پیغام_2->داخلہ = داخلہ;
    پیغام_2->داخلہ_سائز = داخلہ_سائز;
    پیغام_2->output = خالی_اشاریہ;
    پیغام_2->output_سائز = 0;
    GGML_ASSERT(queue.ڈالو(پیغام_2));
    خودکار future = پیغام_2->completion.get_future();
    future.wait();
}

خالی rpc_dispatcher::send_async(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز) {
    خودکار پیغام_2 = std::make_shared<rpc_پیغام>();
    پیغام_2->cmd = cmd;
    پیغام_2->داخلہ = داخلہ;
    پیغام_2->داخلہ_سائز = داخلہ_سائز;
    پیغام_2->output = خالی_اشاریہ;
    پیغام_2->output_سائز = 0;
    GGML_ASSERT(queue.ڈالو(پیغام_2));
}

خالی rpc_dispatcher::send(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز, خالی * output, size_t output_سائز) {
    خودکار پیغام_2 = std::make_shared<rpc_پیغام>();
    پیغام_2->cmd = cmd;
    پیغام_2->داخلہ = داخلہ;
    پیغام_2->داخلہ_سائز = داخلہ_سائز;
    پیغام_2->output = output;
    پیغام_2->output_سائز = output_سائز;
    GGML_ASSERT(queue.ڈالو(پیغام_2));
    خودکار future = پیغام_2->completion.get_future();
    future.wait();
}

خالی rpc_dispatcher::send_async(شمار rpc_cmd cmd, std::shared_ptr<ثابت خالی> داخلہ, size_t داخلہ_سائز, خالی * output, size_t output_سائز) {
    خودکار پیغام_2 = std::make_shared<rpc_پیغام>();
    پیغام_2->cmd = cmd;
    پیغام_2->داخلہ = داخلہ;
    پیغام_2->داخلہ_سائز = داخلہ_سائز;
    پیغام_2->output = output;
    پیغام_2->output_سائز = output_سائز;
    GGML_ASSERT(queue.ڈالو(پیغام_2));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = نیا rpc_event;
    ev->پیغام_2 = std::make_shared<rpc_پیغام>();
    ev->پیغام_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->پیغام_2->completion.get_future().share();
    GGML_ASSERT(queue.ڈالو(ev->پیغام_2));
    واپس نیا ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

خالی rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    حذف ev;
}

خالی rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

خالی rpc_dispatcher::event_ریکارڈ(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->پیغام_2 = std::make_shared<rpc_پیغام>();
    ev->پیغام_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->پیغام_2->completion.get_future().share();
    GGML_ASSERT(queue.ڈالو(ev->پیغام_2));
}

خالی rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    خودکار پیغام_2 = std::make_shared<rpc_پیغام>();
    پیغام_2->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ڈالو(پیغام_2));
    پیغام_2->completion.get_future().wait();
}

خالی rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

خالی rpc_dispatcher::busy_spin_release() {
    ثابت غیرعلامتی پچھلا_2 = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(پچھلا_2 > 0);
}

خالی rpc_dispatcher::start(ثابت std::string & endpoint) {
    std::string host;
    صحیح پورٹ;
    اگر (!parse_endpoint(endpoint, host, پورٹ)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    اگر (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), پورٹ);
    اگر (sock == خالی_اشاریہ) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    اگر (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = سچ;
    thread = std::thread(rpc_dispatcher_trampoline, یہ);
}

خالی rpc_dispatcher::work() {
    خودکار آخری_cmd = std::chrono::steady_clock::now();
    جب (running) {
        rpc_پیغام_ptr پیغام_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ثابت منطقی spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - آخری_cmd < RPC_BUSY_SPIN_IDLE_وقت;
        اگر (spin) {
            اگر (!queue.try_نکالو(&پیغام_ptr)) {
                rpc_cpu_relax();
                جاری;
            }
        } ورنہ اگر (!queue.نکالو(&پیغام_ptr)) {
            توڑو;
        }
        اگر (پیغام_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(پیغام_ptr->cmd),
                     rpc_log_format_سائز(پیغام_ptr->داخلہ_سائز).c_str(), rpc_log_format_سائز(پیغام_ptr->output_سائز).c_str());
            ثابت منطقی trace = rpc_log_debug_level() >= 3;
            ثابت خودکار t0 = std::chrono::steady_clock::now();
            اگر (پیغام_ptr->output) {
                منطقی حالت_2 = send_rpc_cmd(sock, پیغام_ptr->cmd, پیغام_ptr->داخلہ.get(), پیغام_ptr->داخلہ_سائز, پیغام_ptr->output, پیغام_ptr->output_سائز);
                RPC_حالت_ASSERT(حالت_2);
            } ورنہ {
                منطقی حالت_2 = send_rpc_cmd(sock, پیغام_ptr->cmd, پیغام_ptr->داخلہ.get(), پیغام_ptr->داخلہ_سائز);
                RPC_حالت_ASSERT(حالت_2);
            }
            اگر (trace) {
                ثابت std::chrono::duration<دوہرا, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(پیغام_ptr->cmd), elapsed.count());
            }
        } ورنہ {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        پیغام_ptr->completion.set_value();
        آخری_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = جھوٹ;
    queue.مداخلت();
    sock = خالی_اشاریہ;
    اگر (thread.joinable()) {
        thread.join();
    }
}

ساکن std::shared_ptr<rpc_dispatcher> get_dispatcher(ثابت std::string & endpoint) {
    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);
    ساکن std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    خودکار it = dispatchers.find(endpoint);
    اگر (it != dispatchers.end()) {
        اگر (خودکار dispatcher = it->second.تالا()) {
            واپس dispatcher;
        }
    }

    خودکار dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    واپس dispatcher;
}

ساکن خالی ggml_backend_rpc_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    خودکار request = std::make_shared<rpc_پیغام_free_بفر_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_بفر, request, حجم(*request));
    حذف ctx;
}

ساکن خالی * ggml_backend_rpc_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    اگر (ctx->بنیاد_ptr != خالی_اشاریہ) {
        واپس ctx->بنیاد_ptr;
    }
    خودکار request = std::make_shared<rpc_پیغام_بفر_get_بنیاد_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_پیغام_بفر_get_بنیاد_rsp response;
    ctx->dispatcher->send(RPC_CMD_بفر_GET_بنیاد, request, حجم(*request), &response, حجم(response));
    ctx->بنیاد_ptr = reinterpret_cast<خالی *>(response.بنیاد_ptr);
    واپس ctx->بنیاد_ptr;
}

ساکن منطقی ggml_backend_بفر_is_rpc(ggml_backend_بفر_t بفر) {
    واپس بفر->iface.free_بفر == ggml_backend_rpc_بفر_free_بفر;
}

ساکن rpc_tensor serialize_tensor(ثابت ggml_tensor * tensor, ثابت rpc_dispatcher * dispatcher = خالی_اشاریہ) {
    rpc_tensor نتیجہ;
    اگر (!tensor) {
        memset(&نتیجہ, 0, حجم(نتیجہ));
        واپس نتیجہ;
    }

    نتیجہ.id = reinterpret_cast<uint64_t>(tensor);
    نتیجہ.type = tensor->type;
    اگر (tensor->بفر && ggml_backend_بفر_is_rpc(tensor->بفر)) {
        ggml_backend_بفر_t بفر = tensor->بفر;
        ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        اگر (ctx != خالی_اشاریہ && (dispatcher == خالی_اشاریہ || ctx->dispatcher.get() == dispatcher)) {
            نتیجہ.بفر = ctx->remote_ptr;
            نتیجہ.ڈیٹا = reinterpret_cast<uint64_t>(tensor->ڈیٹا);
        } ورنہ {
            نتیجہ.بفر = 0;
            نتیجہ.ڈیٹا = 0;
        }
    } ورنہ {
        نتیجہ.بفر = 0;
        نتیجہ.ڈیٹا   = 0;
    }
    برائے (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ.ne[i] = tensor->ne[i];
        نتیجہ.nb[i] = tensor->nb[i];
    }
    نتیجہ.op = tensor->op;
    برائے (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجہ.op_params[i] = tensor->op_params[i];
    }
    نتیجہ.flags = tensor->flags;
    برائے (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        نتیجہ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    نتیجہ.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    نتیجہ.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(نتیجہ.name, 0, حجم(نتیجہ.name));
    نتیجہ.use_گنتی = 0;

    snprintf(نتیجہ.name, GGML_MAX_NAME, "%s", tensor->name);
    واپس نتیجہ;
}

ساکن شمار ggml_حالت ggml_backend_rpc_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    اگر (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == خالی_اشاریہ)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        خودکار request = std::make_shared<rpc_پیغام_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, حجم(*request));
    }
    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_rpc_بفر_memset_tensor(
        ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t انحراف, size_t size) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    خودکار request = std::make_shared<rpc_پیغام_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->انحراف = انحراف;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), انحراف,
             rpc_log_format_سائز(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, حجم(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ساکن std::shared_ptr<uint8_t> serialize_سیٹ_tensor(ثابت rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t انحراف, ثابت خالی * ڈیٹا, size_t size, size_t & داخلہ_سائز) {
    داخلہ_سائز = حجم(rpc_tensor) + حجم(cache_flag) + حجم(انحراف) + size;
    uint8_t * داخلہ = نیا uint8_t[داخلہ_سائز]();
    uint8_t * p = داخلہ;
    حافظہ_نقل(p, &rpc_tensor, حجم(rpc_tensor)); p += حجم(rpc_tensor);
    حافظہ_نقل(p, &cache_flag, حجم(cache_flag)); p += حجم(cache_flag);
    حافظہ_نقل(p, &انحراف,     حجم(انحراف));     p += حجم(انحراف);
    حافظہ_نقل(p, ڈیٹا, size);
    واپس std::shared_ptr<uint8_t>(داخلہ, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ساکن منطقی rpc_use_hash_cache(ثابت ggml_tensor * tensor, size_t size) {
    واپس size > HASH_حد && tensor->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS;
}

ساکن خالی ggml_backend_rpc_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), انحراف,
             rpc_log_format_سائز(size).c_str());
    اگر (rpc_use_hash_cache(tensor, size)) {
        خودکار request = std::make_shared<rpc_پیغام_سیٹ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->انحراف = انحراف;
        request->hash = fnv_hash((ثابت uint8_t*)ڈیٹا, size);
        rpc_پیغام_سیٹ_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_سیٹ_TENSOR_HASH, request, حجم(*request), &response, حجم(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.نتیجہ ? "hit" : "miss");
        اگر (response.نتیجہ) {
            // the server has the same data, no need to send it
            واپس;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t داخلہ_سائز;
    خودکار داخلہ = serialize_سیٹ_tensor(rpc_tensor, cache_flag, انحراف, ڈیٹا, size, داخلہ_سائز);
    ctx->dispatcher->send(RPC_CMD_سیٹ_TENSOR, داخلہ, داخلہ_سائز);
}

ساکن خالی ggml_backend_rpc_بفر_سیٹ_tensor_2d(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت خالی * ڈیٹا,
        size_t انحراف, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), انحراف, rpc_log_format_سائز(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t داخلہ_سائز = حجم(rpc_tensor) + 4*حجم(uint64_t) + size*n_copies;
    uint8_t * داخلہ = نیا uint8_t[داخلہ_سائز]();
    uint8_t * dest = داخلہ;
    حافظہ_نقل(dest, &rpc_tensor, حجم(rpc_tensor));
    dest += حجم(rpc_tensor);
    uint64_t سرخی[4] = { انحراف, size, n_copies, stride_tensor };
    حافظہ_نقل(dest, سرخی, حجم(سرخی));
    dest += حجم(سرخی);
    برائے (size_t i = 0; i < n_copies; i++) {
        حافظہ_نقل(dest + i*size, (ثابت حرف *)ڈیٹا + i*stride_ڈیٹا, size);
    }
    std::shared_ptr<uint8_t> داخلہ_ptr(داخلہ, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_سیٹ_TENSOR_2_D, داخلہ_ptr, داخلہ_سائز);
}

ساکن خالی ggml_backend_rpc_بفر_get_tensor_2d(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا,
        size_t انحراف, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), انحراف, rpc_log_format_سائز(size).c_str(), n_copies, stride_tensor);
    خودکار request = std::make_shared<rpc_پیغام_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->انحراف   = انحراف;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    اگر (stride_ڈیٹا == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, حجم(*request), ڈیٹا, size*n_copies);
    } ورنہ {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, حجم(*request), packed.ڈیٹا(), packed.size());
        برائے (size_t i = 0; i < n_copies; i++) {
            حافظہ_نقل((حرف *)ڈیٹا + i*stride_ڈیٹا, packed.ڈیٹا() + i*size, size);
        }
    }
}

ساکن خالی ggml_backend_rpc_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), انحراف,
             rpc_log_format_سائز(size).c_str());
    خودکار request = std::make_shared<rpc_پیغام_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->انحراف = انحراف;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, حجم(*request), ڈیٹا, size);
}

ساکن منطقی ggml_backend_rpc_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (ggml_backend_بفر_is_rpc(src->بفر)) {
        // check if src and dst are on the same server
        ggml_backend_بفر_t src_بفر = src->بفر;
        ggml_backend_rpc_بفر_context * src_ctx = (ggml_backend_rpc_بفر_context *)src_بفر->context;
        ggml_backend_بفر_t dst_بفر = dst->بفر;
        ggml_backend_rpc_بفر_context * dst_ctx = (ggml_backend_rpc_بفر_context *)dst_بفر->context;
        اگر (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            واپس جھوٹ;
        }
        ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        خودکار request = std::make_shared<rpc_پیغام_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_پیغام_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, حجم(*request), &response, حجم(response));
        واپس response.نتیجہ;
    }
    واپس جھوٹ;
}

ساکن خالی ggml_backend_rpc_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    خودکار request = std::make_shared<rpc_پیغام_بفر_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_بفر_CLEAR, request, حجم(*request));
}

ساکن ggml_backend_بفر_i ggml_backend_rpc_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_rpc_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_rpc_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_بفر_سیٹ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_بفر_clear,
    /* .reset           = */ تہی,
};

ساکن ثابت حرف * ggml_backend_rpc_بفر_type_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    واپس buft_ctx->name.c_str();
}

ساکن ggml_backend_بفر_t ggml_backend_rpc_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    خودکار request = std::make_shared<rpc_پیغام_alloc_بفر_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_پیغام_alloc_بفر_rsp response;

    خودکار dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_بفر, request, حجم(*request), &response, حجم(response));
    اگر (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_سائز(request->size).c_str(),
                response.remote_ptr, rpc_log_format_سائز(response.remote_سائز).c_str());
        ggml_backend_بفر_t بفر = ggml_backend_بفر_init(buft,
            ggml_backend_rpc_بفر_interface,
            نیا ggml_backend_rpc_بفر_context{dispatcher, خالی_اشاریہ, response.remote_ptr},
            response.remote_سائز);
        واپس بفر;
    } ورنہ {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_سائز(request->size).c_str());
        واپس خالی_اشاریہ;
    }
}

ساکن size_t get_alignment(ثابت std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    خودکار request = std::make_shared<rpc_پیغام_get_alignment_req>();
    request->device = device;
    rpc_پیغام_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, حجم(*request), &response, حجم(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_سائز(response.alignment).c_str());
    واپس response.alignment;
}

ساکن size_t ggml_backend_rpc_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    واپس buft_ctx->alignment;
}

ساکن size_t get_max_سائز(ثابت std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    خودکار request = std::make_shared<rpc_پیغام_get_max_سائز_req>();
    request->device = device;
    rpc_پیغام_get_max_سائز_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_سائز, request, حجم(*request), &response, حجم(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_سائز(response.max_size).c_str());
    واپس response.max_size;
}

ساکن size_t ggml_backend_rpc_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    واپس buft_ctx->max_size;
}

ساکن size_t ggml_backend_rpc_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    منطقی rpc_get = جھوٹ;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == خالی_اشاریہ);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_سائز_may_expand(tensor->op);

    اگر (rpc_get) {
        ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ثابت size_t min_سائز = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ساخت alloc_سائز_cache_کلید {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_سائز_cache_کلید کلید = {};
        کلید.device = buft_ctx->device;
        کلید.type = tensor->type;
        کلید.op = tensor->op;
        حافظہ_نقل(کلید.op_params, tensor->op_params, حجم(کلید.op_params));
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
            کلید.ne[i] = (uint32_t)tensor->ne[i];
            کلید.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ثابت uint8_t *)&کلید, حجم(کلید));
        cache_hash = fnv_hash((ثابت uint8_t *)buft_ctx->endpoint.ڈیٹا(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ساکن std::mutex cache_mutex;
        ساکن std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> تالا(cache_mutex);
            خودکار it = cache.find(cache_hash);
            اگر (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_سائز(it->second).c_str());
                واپس std::max<size_t>(it->second, min_سائز);
            }
        }

        خودکار request = std::make_shared<rpc_پیغام_get_alloc_سائز_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_پیغام_get_alloc_سائز_rsp response;
        خودکار dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_سائز, request, حجم(*request), &response, حجم(response));

        {
            std::lock_guard<std::mutex> تالا(cache_mutex);
            cache[cache_hash] = response.alloc_سائز;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_سائز(response.alloc_سائز).c_str());

        واپس std::max<size_t>(response.alloc_سائز, min_سائز);
    }

    واپس ggml_nbytes(tensor);
}

ساکن ggml_backend_بفر_type_i ggml_backend_rpc_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_rpc_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_rpc_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ تہی,
};

ساکن ثابت حرف * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    واپس rpc_ctx->name.c_str();
}

ساکن خالی ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    حذف rpc_ctx;
    حذف backend;
}

ساکن خالی ggml_backend_rpc_سیٹ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), انحراف,
             rpc_log_format_سائز(size).c_str());
    اگر (rpc_use_hash_cache(tensor, size)) {
        خودکار request = std::make_shared<rpc_پیغام_سیٹ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->انحراف = انحراف;
        request->hash = fnv_hash((ثابت uint8_t*)ڈیٹا, size);
        rpc_پیغام_سیٹ_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_سیٹ_TENSOR_HASH, request, حجم(*request), &response, حجم(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.نتیجہ ? "hit" : "miss");
        اگر (response.نتیجہ) {
            // the server has the same data, no need to send it
            واپس;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t داخلہ_سائز;
    خودکار داخلہ = serialize_سیٹ_tensor(rpc_tensor, cache_flag, انحراف, ڈیٹا, size, داخلہ_سائز);
    ctx->dispatcher->send_async(RPC_CMD_سیٹ_TENSOR, داخلہ, داخلہ_سائز);
}

ساکن خالی ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, خالی * ڈیٹا, size_t انحراف, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), انحراف,
             rpc_log_format_سائز(size).c_str());
    خودکار request = std::make_shared<rpc_پیغام_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->انحراف = انحراف;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, حجم(*request), ڈیٹا, size);
}

ساکن خالی ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ساکن خالی جوڑو_tensor(ggml_tensor * tensor, ثابت ggml_cgraph * cgraph, ثابت rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & دیکھا) {
    اگر (tensor == خالی_اشاریہ) {
        واپس;
    }
    اگر (دیکھا.find(tensor) != دیکھا.end()) {
        واپس;
    }
    دیکھا.insert(tensor);
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        جوڑو_tensor(tensor->src[i], cgraph, dispatcher, tensors, دیکھا);
    }
    جوڑو_tensor(tensor->view_src, cgraph, dispatcher, tensors, دیکھا);
    rpc_tensor نتیجہ = serialize_tensor(tensor, dispatcher);
    ثابت size_t hash_pos = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, tensor);
    اگر (hash_pos != GGML_HASHSET_بھرا && ggml_bitset_get(cgraph->دیکھا_hash_سیٹ.مستعمل, hash_pos)) {
        نتیجہ.use_گنتی = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(نتیجہ);
}

ساکن uint8_t * serialize_گراف(uint32_t device, ثابت ggml_cgraph * cgraph, ثابت rpc_dispatcher * dispatcher, size_t * output_سائز) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> دیکھا;
    برائے (uint32_t i = 0; i < n_nodes; i++) {
        جوڑو_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, دیکھا);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_سائز = 2*حجم(uint32_t) + حجم(uint64_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(rpc_tensor);
    uint8_t * output = نیا uint8_t[*output_سائز]();
    uint8_t * dest = output;
    حافظہ_نقل(dest, &device, حجم(device));
    dest += حجم(device);
    حافظہ_نقل(dest, &cgraph->uid, حجم(cgraph->uid));
    dest += حجم(cgraph->uid);
    حافظہ_نقل(dest, &n_nodes, حجم(n_nodes));
    dest += حجم(n_nodes);
    برائے (uint32_t i = 0; i < n_nodes; i++) {
        حافظہ_نقل(dest + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    dest += n_nodes * حجم(uint64_t);
    حافظہ_نقل(dest, &n_tensors, حجم(n_tensors));
    dest += حجم(n_tensors);
    rpc_tensor * برآمد_tensors = (rpc_tensor *)dest;
    حافظہ_نقل(برآمد_tensors, tensors.ڈیٹا(), n_tensors * حجم(rpc_tensor));
    واپس output;
}

خالی rpc_dispatcher::گراف_compute(uint32_t device, ثابت ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> تالا(گراف_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    خودکار & device_گراف_uids = گراف_uids[device];
    منطقی reuse = cgraph->uid != 0 && device_گراف_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    اگر (reuse) {
        خودکار request = std::make_shared<rpc_پیغام_گراف_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_گراف_RECOMPUTE, request, حجم(*request));
    } ورنہ {
        اگر (cgraph->uid != 0) {
            اگر (device_گراف_uids.size() >= گراف_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_گراف_uids.size());
                device_گراف_uids.clear();
            }
            device_گراف_uids.insert(cgraph->uid);
        }
        size_t داخلہ_سائز = 0;
        uint8_t * داخلہ = serialize_گراف(device, cgraph, یہ, &داخلہ_سائز);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_سائز(داخلہ_سائز).c_str());
        std::shared_ptr<uint8_t> داخلہ_ptr(داخلہ, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_گراف_COMPUTE, داخلہ_ptr, داخلہ_سائز);
    }
}

ساکن شمار ggml_حالت ggml_backend_rpc_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->گراف_compute(rpc_ctx->device, cgraph);
    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_rpc_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_ریکارڈ(event);
}

ساکن خالی ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_غیر_مستعمل(backend);
    GGML_غیر_مستعمل(event);
}

ساکن خالی ggml_backend_rpc_سیٹ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت خالی * ڈیٹا,
        size_t انحراف, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    ggml_backend_tensor_سیٹ_2d(tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
    GGML_غیر_مستعمل(backend);
}

ساکن خالی ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, خالی * ڈیٹا,
        size_t انحراف, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ڈیٹا) {
    ggml_backend_tensor_get_2d(tensor, ڈیٹا, انحراف, size, n_copies, stride_tensor, stride_ڈیٹا);
    GGML_غیر_مستعمل(backend);
}

ساکن ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_سیٹ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ تہی,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_rpc_گراف_compute,
    /* .event_record            = */ ggml_backend_rpc_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ تہی,
};

ggml_backend_بفر_type_t ggml_backend_rpc_بفر_type(ثابت حرف * endpoint, uint32_t device) {
    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ساکن std::unordered_map<std::string, ggml_backend_بفر_type_t> buft_map;
    خودکار it = buft_map.find(buft_name);
    اگر (it != buft_map.end()) {
        واپس it->second;
    }
    خودکار dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_سائز(dispatcher, device);
    ggml_backend_rpc_بفر_type_context * buft_ctx = نیا ggml_backend_rpc_بفر_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    خودکار reg = ggml_backend_rpc_جوڑو_server(endpoint);
    ggml_backend_بفر_type_t buft = نیا ggml_backend_بفر_type {
        /* .iface   = */ ggml_backend_rpc_بفر_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_سائز(alignment).c_str(), rpc_log_format_سائز(max_size).c_str());
    واپس buft;
}

ggml_backend_t ggml_backend_rpc_init(ثابت حرف * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    خودکار dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = نیا ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    خودکار reg = ggml_backend_rpc_جوڑو_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = نیا ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    واپس backend;
}

منطقی ggml_backend_is_rpc(ggml_backend_t backend) {
    واپس backend != تہی && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

خالی ggml_backend_rpc_get_device_حافظہ(ثابت حرف * endpoint, uint32_t device, size_t * آزاد, size_t * کل) {
    خودکار dispatcher = get_dispatcher(endpoint);
    خودکار request = std::make_shared<rpc_پیغام_get_device_حافظہ_req>();
    request->device = device;
    rpc_پیغام_get_device_حافظہ_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_حافظہ, request, حجم(*request), &response, حجم(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_سائز(response.free_mem).c_str(), rpc_log_format_سائز(response.کل_mem).c_str());
    *آزاد = response.free_mem;
    *کل = response.کل_mem;
}

// RPC server-side implementation

جماعت rpc_server {
عوامی:
    rpc_server(std::vector<ggml_backend_t> all_backends, ثابت حرف * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_حالتیں.resize(backends.size());
    }
    ~rpc_server();

    خالی hello(rpc_پیغام_hello_rsp & response);
    منطقی alloc_بفر(ثابت rpc_پیغام_alloc_بفر_req & request, rpc_پیغام_alloc_بفر_rsp & response);
    منطقی get_alignment(ثابت rpc_پیغام_get_alignment_req & request, rpc_پیغام_get_alignment_rsp & response);
    منطقی get_max_سائز(ثابت rpc_پیغام_get_max_سائز_req & request, rpc_پیغام_get_max_سائز_rsp & response);
    منطقی بفر_get_بنیاد(ثابت rpc_پیغام_بفر_get_بنیاد_req & request, rpc_پیغام_بفر_get_بنیاد_rsp & response);
    منطقی free_بفر(ثابت rpc_پیغام_free_بفر_req & request);
    منطقی بفر_clear(ثابت rpc_پیغام_بفر_clear_req & request);
    منطقی memset_tensor(ثابت rpc_پیغام_memset_tensor_req & request);
    منطقی سیٹ_tensor(ثابت std::vector<uint8_t> & داخلہ);
    منطقی سیٹ_tensor_2d(ثابت std::vector<uint8_t> & داخلہ);
    منطقی سیٹ_tensor_hash(ثابت rpc_پیغام_سیٹ_tensor_hash_req & request, rpc_پیغام_سیٹ_tensor_hash_rsp & response);
    منطقی get_tensor(ثابت rpc_پیغام_get_tensor_req & request, std::vector<uint8_t> & response);
    منطقی get_tensor_2d(ثابت rpc_پیغام_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    منطقی copy_tensor(ثابت rpc_پیغام_copy_tensor_req & request, rpc_پیغام_copy_tensor_rsp & response);
    منطقی گراف_compute(ثابت std::vector<uint8_t> & داخلہ);
    منطقی گراف_recompute(ثابت rpc_پیغام_گراف_recompute_req & request);
    منطقی comm_init(ثابت rpc_پیغام_comm_init_req & request, rpc_پیغام_comm_init_rsp & response);
    منطقی comm_allreduce(ثابت rpc_پیغام_comm_allreduce_req & request);
    منطقی comm_free(ثابت rpc_پیغام_comm_free_req & request);
    منطقی init_tensor(ثابت rpc_پیغام_init_tensor_req & request);
    منطقی get_alloc_سائز(ثابت rpc_پیغام_get_alloc_سائز_req & request, rpc_پیغام_get_alloc_سائز_rsp & response);
    منطقی get_device_حافظہ(ثابت rpc_پیغام_get_device_حافظہ_req & request, rpc_پیغام_get_device_حافظہ_rsp & response);

    ساخت stored_گراف {
        std::vector<uint8_t>   بفر;
        ggml_cgraph          * graph;
    };

نجی:
    خالی sync_all_backends();
    منطقی get_cached_file(uint64_t hash, std::vector<uint8_t> & ڈیٹا);
    ggml_tensor * deserialize_tensor(ساخت ggml_context * ctx, ثابت rpc_tensor * tensor);
    ggml_tensor * create_گرہ(uint64_t id,
                              ساخت ggml_context * ctx,
                              ثابت std::unordered_map<uint64_t, ثابت rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ساخت ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ساخت comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_بفر_ptr scratch;
        size_t                  scratch_سائز = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ثابت حرف * cache_dir;
    std::unordered_set<ggml_backend_بفر_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_گراف>> stored_graphs;
    std::vector<comm_state> comm_حالتیں;
};

خالی rpc_server::hello(rpc_پیغام_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_نسخہ;
    response.minor = RPC_PROTO_MINOR_نسخہ;
    response.patch = RPC_PROTO_PATCH_نسخہ;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

منطقی rpc_server::get_alloc_سائز(ثابت rpc_پیغام_get_alloc_سائز_req & request, rpc_پیغام_get_alloc_سائز_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        واپس جھوٹ;
    }
    ggml_backend_بفر_type_t buft;
    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == خالی_اشاریہ) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        واپس جھوٹ;
    }
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (خالی*)tensor->بفر, tensor->ڈیٹا);
    اگر (tensor->بفر == خالی_اشاریہ) {
        //No buffer allocated.
        buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    } ورنہ {
        buft = tensor->بفر->buft;
    }

    response.alloc_سائز = ggml_backend_buft_get_alloc_سائز(buft, tensor);

    واپس سچ;
}

منطقی rpc_server::alloc_بفر(ثابت rpc_پیغام_alloc_بفر_req & request, rpc_پیغام_alloc_بفر_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        واپس جھوٹ;
    }
    ggml_backend_بفر_type_t buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, request.size);
    response.remote_ptr = 0;
    response.remote_سائز = 0;
    اگر (بفر != خالی_اشاریہ) {
        response.remote_ptr = reinterpret_cast<uint64_t>(بفر);
        response.remote_سائز = بفر->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_سائز(request.size).c_str(), response.remote_ptr,
                rpc_log_format_سائز(response.remote_سائز).c_str());
        buffers.insert(بفر);
    } ورنہ {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_سائز(request.size).c_str());
    }
    واپس سچ;
}

منطقی rpc_server::get_alignment(ثابت rpc_پیغام_get_alignment_req & request, rpc_پیغام_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        واپس جھوٹ;
    }
    ggml_backend_بفر_type_t buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_سائز(alignment).c_str());
    response.alignment = alignment;
    واپس سچ;
}

منطقی rpc_server::get_max_سائز(ثابت rpc_پیغام_get_max_سائز_req & request, rpc_پیغام_get_max_سائز_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        واپس جھوٹ;
    }
    ggml_backend_بفر_type_t buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_سائز(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_سائز(max_size).c_str());
    response.max_size = max_size;
    واپس سچ;
}

منطقی rpc_server::بفر_get_بنیاد(ثابت rpc_پیغام_بفر_get_بنیاد_req & request, rpc_پیغام_بفر_get_بنیاد_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_بفر_t بفر = reinterpret_cast<ggml_backend_بفر_t>(request.remote_ptr);
    اگر (buffers.find(بفر) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        واپس جھوٹ;
    }
    خالی * بنیاد = ggml_backend_بفر_get_بنیاد(بفر);
    response.بنیاد_ptr = reinterpret_cast<uint64_t>(بنیاد);
    واپس سچ;
}

منطقی rpc_server::free_بفر(ثابت rpc_پیغام_free_بفر_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_بفر_t بفر = reinterpret_cast<ggml_backend_بفر_t>(request.remote_ptr);
    اگر (buffers.find(بفر) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        واپس جھوٹ;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    برائے (خودکار & sgs : stored_graphs) {
        برائے (خودکار & sg : sgs) {
            sg.second.graph = خالی_اشاریہ;
        }
    }
    ggml_backend_بفر_free(بفر);
    buffers.erase(بفر);
    واپس سچ;
}

منطقی rpc_server::بفر_clear(ثابت rpc_پیغام_بفر_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_بفر_t بفر = reinterpret_cast<ggml_backend_بفر_t>(request.remote_ptr);
    اگر (buffers.find(بفر) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        واپس جھوٹ;
    }
    ggml_backend_بفر_clear(بفر, request.value);
    واپس سچ;
}

منطقی rpc_server::memset_tensor(ثابت rpc_پیغام_memset_tensor_req & request) {
    sync_all_backends();
    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == خالی_اشاریہ || tensor->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }

    ثابت uint64_t tensor_سائز = ggml_nbytes(tensor);
    اگر (request.انحراف > tensor_سائز || request.size > tensor_سائز - request.انحراف) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.انحراف, request.size, tensor_سائز);
        واپس جھوٹ;
    }

    ثابت uint64_t بفر_آغاز = (uint64_t) ggml_backend_بفر_get_بنیاد(tensor->بفر);
    ثابت uint64_t بفر_سائز = ggml_backend_بفر_get_سائز(tensor->بفر);
    اگر (request.tensor.ڈیٹا < بفر_آغاز) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        واپس جھوٹ;
    }
    ثابت uint64_t ڈیٹا_انحراف = request.tensor.ڈیٹا - بفر_آغاز;
    اگر (ڈیٹا_انحراف > بفر_سائز ||
        request.انحراف > بفر_سائز - ڈیٹا_انحراف ||
        request.size > بفر_سائز - ڈیٹا_انحراف - request.انحراف) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        واپس جھوٹ;
    }
    اگر (tensor->بفر->iface.memset_tensor == خالی_اشاریہ) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        واپس جھوٹ;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (خالی *) tensor->بفر, tensor->ڈیٹا, request.انحراف,
            rpc_log_format_سائز(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.انحراف, request.size);
    واپس سچ;
}

ggml_tensor * rpc_server::deserialize_tensor(ساخت ggml_context * ctx, ثابت rpc_tensor * tensor) {
    // Validate tensor type before using it
    اگر (tensor->type >= GGML_TYPE_گنتی) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        واپس خالی_اشاریہ;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    اگر (ggml_blck_سائز((شمار ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        واپس خالی_اشاریہ;
    }

    ggml_tensor * نتیجہ = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    اگر (نتیجہ == خالی_اشاریہ) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        واپس خالی_اشاریہ;
    }

    برائے (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتیجہ->nb[i] = tensor->nb[i];
    }
    نتیجہ->بفر = reinterpret_cast<ggml_backend_بفر_t>(tensor->بفر);
    اگر (نتیجہ->بفر && buffers.find(نتیجہ->بفر) == buffers.end()) {
        نتیجہ->بفر = خالی_اشاریہ;
    }

    اگر (نتیجہ->بفر && ggml_nelements(نتیجہ) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_سائز = (uint64_t) ggml_nbytes(نتیجہ);
        uint64_t بفر_آغاز = (uint64_t) ggml_backend_بفر_get_بنیاد(نتیجہ->بفر);
        uint64_t بفر_سائز = (uint64_t) ggml_backend_بفر_get_سائز(نتیجہ->بفر);
        اگر (tensor->ڈیٹا + tensor_سائز < tensor->ڈیٹا ||
            tensor->ڈیٹا < بفر_آغاز || tensor->ڈیٹا + tensor_سائز > بفر_آغاز + بفر_سائز) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(نتیجہ->type),
                      نتیجہ->ne[0], نتیجہ->ne[1], نتیجہ->ne[2], نتیجہ->ne[3],
                      tensor->ڈیٹا, tensor->ڈیٹا + tensor_سائز, بفر_آغاز, بفر_آغاز + بفر_سائز);
            واپس خالی_اشاریہ;
        }
    }

    نتیجہ->op = (ggml_op) tensor->op;
    برائے (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتیجہ->op_params[i] = tensor->op_params[i];
    }
    نتیجہ->flags = tensor->flags;
    نتیجہ->ڈیٹا = reinterpret_cast<خالی *>(tensor->ڈیٹا);
    ggml_سیٹ_name(نتیجہ, tensor->name);
    واپس نتیجہ;
}


منطقی rpc_server::سیٹ_tensor(ثابت std::vector<uint8_t> & داخلہ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t انحراف;
    ثابت size_t سرخی_سائز = حجم(rpc_tensor) + حجم(cache_flag) + حجم(انحراف);
    اگر (داخلہ.size() < سرخی_سائز) {
        واپس جھوٹ;
    }
    ثابت rpc_tensor * اندر_tensor = (ثابت rpc_tensor *)داخلہ.ڈیٹا();
    حافظہ_نقل(&cache_flag, داخلہ.ڈیٹا() + حجم(rpc_tensor), حجم(cache_flag));
    حافظہ_نقل(&انحراف,     داخلہ.ڈیٹا() + حجم(rpc_tensor) + حجم(cache_flag), حجم(انحراف));
    ثابت size_t size = داخلہ.size() - سرخی_سائز;

    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, اندر_tensor);
    اگر (tensor == خالی_اشاریہ || tensor->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (خالی*)tensor->بفر, tensor->ڈیٹا, انحراف,
            rpc_log_format_سائز(size).c_str());

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_بفر_get_بنیاد(tensor->بفر);
        ثابت size_t p1 = p0 + ggml_backend_بفر_get_سائز(tensor->بفر);

        اگر (اندر_tensor->ڈیٹا + انحراف < p0 || اندر_tensor->ڈیٹا + انحراف >= p1 || size > (p1 - اندر_tensor->ڈیٹا - انحراف)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, اندر_tensor->ڈیٹا, انحراف, size, p0, p1);
            واپس جھوٹ;
        }
    }

    ثابت خالی * ڈیٹا = داخلہ.ڈیٹا() + سرخی_سائز;
    اگر (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ثابت uint8_t*)ڈیٹا, size);
        حرف hash_str[17];
        snprintf(hash_str, حجم(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.اكتب((ثابت حرف *)ڈیٹا, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_سیٹ(tensor, ڈیٹا, انحراف, size);
    واپس سچ;
}

منطقی rpc_server::سیٹ_tensor_2d(ثابت std::vector<uint8_t> & داخلہ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    اگر (داخلہ.size() < حجم(rpc_tensor) + 4*حجم(uint64_t)) {
        واپس جھوٹ;
    }
    ثابت rpc_tensor * اندر_tensor = (ثابت rpc_tensor *)داخلہ.ڈیٹا();
    uint64_t سرخی[4];
    حافظہ_نقل(سرخی, داخلہ.ڈیٹا() + حجم(rpc_tensor), حجم(سرخی));
    ثابت uint64_t انحراف   = سرخی[0];
    ثابت uint64_t size     = سرخی[1];
    ثابت uint64_t n_copies = سرخی[2];
    ثابت uint64_t stride   = سرخی[3];

    ثابت uint64_t ڈیٹا_سائز = داخلہ.size() - حجم(rpc_tensor) - 4*حجم(uint64_t);
    اگر (n_copies == 0 || size == 0 || size > ڈیٹا_سائز / n_copies || size * n_copies != ڈیٹا_سائز) {
        واپس جھوٹ;
    }

    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, اندر_tensor);
    اگر (tensor == خالی_اشاریہ || tensor->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (خالی*)tensor->بفر, tensor->ڈیٹا, انحراف,
            rpc_log_format_سائز(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        اگر (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            واپس جھوٹ;
        }
        ثابت uint64_t وسعت = (n_copies - 1)*stride + size;
        ثابت uint64_t p0 = (uint64_t) ggml_backend_بفر_get_بنیاد(tensor->بفر);
        ثابت uint64_t p1 = p0 + ggml_backend_بفر_get_سائز(tensor->بفر);

        اگر (اندر_tensor->ڈیٹا < p0 || اندر_tensor->ڈیٹا > p1 || انحراف > p1 - اندر_tensor->ڈیٹا || وسعت > p1 - اندر_tensor->ڈیٹا - انحراف) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, اندر_tensor->ڈیٹا, انحراف, وسعت, p0, p1);
            واپس جھوٹ;
        }
        اگر (انحراف > ggml_nbytes(tensor) || وسعت > ggml_nbytes(tensor) - انحراف) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, انحراف, وسعت, ggml_nbytes(tensor));
            واپس جھوٹ;
        }
    }

    ثابت خالی * ڈیٹا = داخلہ.ڈیٹا() + حجم(rpc_tensor) + 4*حجم(uint64_t);
    ggml_backend_tensor_سیٹ_2d(tensor, ڈیٹا, انحراف, size, n_copies, stride, size);
    واپس سچ;
}

منطقی rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & ڈیٹا) {
    اگر (!cache_dir) {
        واپس جھوٹ;
    }
    حرف hash_str[17];
    snprintf(hash_str, حجم(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    اگر (!fs::exists(cache_file, ec)) {
        واپس جھوٹ;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    ڈیٹا.resize(size);
    ifs.اقرأ((حرف *)ڈیٹا.ڈیٹا(), size);
    واپس سچ;
}

منطقی rpc_server::سیٹ_tensor_hash(ثابت rpc_پیغام_سیٹ_tensor_hash_req & request, rpc_پیغام_سیٹ_tensor_hash_rsp & response)
{
    sync_all_backends();
    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == خالی_اشاریہ || tensor->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }

    std::vector<uint8_t> cached_file;
    اگر (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.انحراف, request.hash);
        response.نتیجہ = 0;
        واپس سچ;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (خالی*)tensor->بفر, tensor->ڈیٹا, request.انحراف,
            rpc_log_format_سائز(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_بفر_get_بنیاد(tensor->بفر);
        ثابت size_t p1 = p0 + ggml_backend_بفر_get_سائز(tensor->بفر);

        اگر (request.tensor.ڈیٹا + request.انحراف < p0
         || request.tensor.ڈیٹا + request.انحراف >= p1
         || size > (p1 - request.tensor.ڈیٹا - request.انحراف)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.ڈیٹا, request.انحراف, size, request.hash, p0, p1);
            واپس جھوٹ;
        }
    }
    ggml_backend_tensor_سیٹ(tensor, cached_file.ڈیٹا(), request.انحراف, size);
    response.نتیجہ = 1;
    واپس سچ;
}

منطقی rpc_server::init_tensor(ثابت rpc_پیغام_init_tensor_req & request) {
    sync_all_backends();
    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == خالی_اشاریہ) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        واپس جھوٹ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (خالی*)tensor->بفر, tensor->ڈیٹا);
    // Call the backend's buffer_init_tensor function
    ggml_backend_بفر_t بفر = tensor->بفر;
    اگر (بفر && بفر->iface.init_tensor) {
        بفر->iface.init_tensor(بفر, tensor);
    } ورنہ {
        اگر (!بفر) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    اگر (tensor->extra != خالی_اشاریہ) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        واپس جھوٹ;
    }

    واپس سچ;
}

منطقی rpc_server::get_tensor(ثابت rpc_پیغام_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == خالی_اشاریہ || tensor->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (خالی*)tensor->بفر, tensor->ڈیٹا, request.انحراف,
             rpc_log_format_سائز(request.size).c_str());

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_بفر_get_بنیاد(tensor->بفر);
        ثابت size_t p1 = p0 + ggml_backend_بفر_get_سائز(tensor->بفر);

        اگر (request.tensor.ڈیٹا + request.انحراف < p0 ||
            request.tensor.ڈیٹا + request.انحراف >= p1 ||
            request.size > (p1 - request.tensor.ڈیٹا - request.انحراف)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.ڈیٹا, request.انحراف, request.size, p0, p1);
                واپس جھوٹ;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.ڈیٹا(), request.انحراف, request.size);
    واپس سچ;
}

منطقی rpc_server::get_tensor_2d(ثابت rpc_پیغام_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ساخت ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    اگر (tensor == خالی_اشاریہ || tensor->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (خالی*)tensor->بفر, tensor->ڈیٹا, request.انحراف,
            rpc_log_format_سائز(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        اگر (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            واپس جھوٹ;
        }
        اگر (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            واپس جھوٹ;
        }
        ثابت uint64_t وسعت = (request.n_copies - 1)*request.stride + request.size;
        ثابت uint64_t p0 = (uint64_t) ggml_backend_بفر_get_بنیاد(tensor->بفر);
        ثابت uint64_t p1 = p0 + ggml_backend_بفر_get_سائز(tensor->بفر);

        اگر (request.tensor.ڈیٹا < p0 || request.tensor.ڈیٹا > p1 || request.انحراف > p1 - request.tensor.ڈیٹا ||
                وسعت > p1 - request.tensor.ڈیٹا - request.انحراف) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.ڈیٹا, request.انحراف, وسعت, p0, p1);
            واپس جھوٹ;
        }
        اگر (request.انحراف > ggml_nbytes(tensor) || وسعت > ggml_nbytes(tensor) - request.انحراف) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.انحراف, وسعت, ggml_nbytes(tensor));
            واپس جھوٹ;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.ڈیٹا(), request.انحراف, request.size, request.n_copies, request.stride, request.size);
    واپس سچ;
}

منطقی rpc_server::copy_tensor(ثابت rpc_پیغام_copy_tensor_req & request, rpc_پیغام_copy_tensor_rsp & response) {
    sync_all_backends();
    ساخت ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    اگر (src == خالی_اشاریہ || dst == خالی_اشاریہ || src->بفر == خالی_اشاریہ || dst->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        واپس جھوٹ;
    }

    uint64_t src_سائز   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_ڈیٹا   = (uint64_t) dst->ڈیٹا;
    uint64_t dst_بنیاد   = (uint64_t) ggml_backend_بفر_get_بنیاد(dst->بفر);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_بفر_get_سائز(dst->بفر);

    اگر (dst_ڈیٹا + src_سائز > dst_بنیاد + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_ڈیٹا,
                         dst_ڈیٹا + src_سائز,
                         dst_بنیاد,
                         dst_بنیاد + dst_buf_sz);
        واپس جھوٹ;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.نتیجہ = ggml_backend_بفر_copy_tensor(src, dst);
    واپس سچ;
}

ggml_tensor * rpc_server::create_گرہ(uint64_t id,
                                      ساخت ggml_context * ctx,
                                      ثابت std::unordered_map<uint64_t, ثابت rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ساخت ggml_tensor*> & tensor_map) {
    اگر (tensor_map.find(id) != tensor_map.end()) {
        واپس tensor_map[id];
    }
    // Safely find the tensor pointer
    خودکار it_ptr = tensor_ptrs.find(id);
    اگر (it_ptr == tensor_ptrs.end()) {
        واپس خالی_اشاریہ;
    }
    ثابت rpc_tensor * tensor = it_ptr->second;

    ساخت ggml_tensor * نتیجہ = deserialize_tensor(ctx, tensor);
    اگر (نتیجہ == خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }
    اگر (نتیجہ->بفر == خالی_اشاریہ && نتیجہ->ڈیٹا != خالی_اشاریہ) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        واپس خالی_اشاریہ;
    }
    tensor_map[id] = نتیجہ;
    برائے (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        اگر (tensor->src[i] == 0) {
            نتیجہ->src[i] = خالی_اشاریہ;
        } ورنہ {
            نتیجہ->src[i] = create_گرہ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            اگر (نتیجہ->src[i] == خالی_اشاریہ) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                واپس خالی_اشاریہ;
            }
        }
    }

    // Handle view_src similarly
    اگر (tensor->view_src == 0) {
        نتیجہ->view_src = خالی_اشاریہ;
    } ورنہ {
        نتیجہ->view_src = create_گرہ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        اگر (نتیجہ->view_src == خالی_اشاریہ) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            واپس خالی_اشاریہ;
        }
    }
    نتیجہ->view_offs = tensor->view_offs;
    واپس نتیجہ;
}

منطقی rpc_server::گراف_compute(ثابت std::vector<uint8_t> & داخلہ) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    اگر (داخلہ.size() < 2*حجم(uint32_t) + حجم(uint64_t)) {
        واپس جھوٹ;
    }
    ثابت uint8_t * src = داخلہ.ڈیٹا();
    uint32_t device;
    حافظہ_نقل(&device, src, حجم(device));
    src += حجم(device);
    اگر (device >= backends.size()) {
        واپس جھوٹ;
    }
    uint64_t uid;
    حافظہ_نقل(&uid, src, حجم(uid));
    src += حجم(uid);
    uint32_t n_nodes;
    حافظہ_نقل(&n_nodes, src, حجم(n_nodes));
    src += حجم(n_nodes);
    اگر (داخلہ.size() < 2*حجم(uint32_t) + حجم(uint64_t) + n_nodes*حجم(uint64_t) + حجم(uint32_t)) {
        واپس جھوٹ;
    }
    ثابت uint64_t * nodes = (ثابت uint64_t *)src;
    src += n_nodes*حجم(uint64_t);
    uint32_t n_tensors;
    حافظہ_نقل(&n_tensors, src, حجم(n_tensors));
    src += حجم(n_tensors);
    اگر (داخلہ.size() < 2*حجم(uint32_t) + حجم(uint64_t) + n_nodes*حجم(uint64_t) + حجم(uint32_t) + n_tensors*حجم(rpc_tensor)) {
        واپس جھوٹ;
    }
    ثابت rpc_tensor * tensors = (ثابت rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    اگر (uid != 0 && stored_graphs[device].size() >= گراف_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_گراف sg_tmp;
    stored_گراف & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_سائز = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_گراف_overhead_custom(n_nodes, جھوٹ);
    اگر (sg.بفر.size() < buf_سائز) {
        sg.بفر.resize(buf_سائز);
    }
    ساخت ggml_init_params params = {
        /*.mem_size   =*/ buf_سائز,
        /*.mem_buffer =*/ sg.بفر.ڈیٹا(),
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ساخت ggml_cgraph * graph = ggml_new_گراف_custom(ctx, n_nodes, جھوٹ);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ثابت rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    برائے (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ثابت خودکار t_deserialize = std::chrono::steady_clock::now();
    برائے (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        حافظہ_نقل(&id, &nodes[i], حجم(id));
        graph->nodes[i] = create_گرہ(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        اگر (graph->nodes[i] == خالی_اشاریہ && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            واپس جھوٹ;
        }
        اگر (graph->nodes[i] != خالی_اشاریہ) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ثابت size_t hash_pos = ggml_hash_داخل_کرو(&graph->دیکھا_hash_سیٹ, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_گنتی;
        }
    }
    ثابت std::chrono::duration<دوہرا, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(backends[device], graph);
    GGML_ASSERT(حالت_2 == GGML_حالت_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    واپس سچ;
}

منطقی rpc_server::گراف_recompute(ثابت rpc_پیغام_گراف_recompute_req & request) {
    uint32_t device = request.device;
    اگر (device >= backends.size()) {
        واپس جھوٹ;
    }
    خودکار it = stored_graphs[device].find(request.uid);
    اگر (it == stored_graphs[device].end() || it->second.graph == خالی_اشاریہ) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        واپس جھوٹ;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(backends[device], graph);
    GGML_ASSERT(حالت_2 == GGML_حالت_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    واپس سچ;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
خالی rpc_server::sync_all_backends() {
    برائے (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
منطقی rpc_server::comm_init(ثابت rpc_پیغام_comm_init_req & request, rpc_پیغام_comm_init_rsp & response) {
    response.ok = 0;
    اگر (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        واپس سچ;
    }
    comm_state & state = comm_حالتیں[request.device];
    اگر (state.peer != خالی_اشاریہ) {
        response.ok = 1;
        واپس سچ;
    }
    uint8_t local_caps[RPC_CONN_CAPS_سائز] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_سائز] = {};
    اگر (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.پورٹ);
        اگر (srv == خالی_اشاریہ) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.پورٹ);
            واپس سچ;
        }
        state.peer = srv->accept();
        اگر (state.peer == خالی_اشاریہ) {
            واپس سچ;
        }
        اگر (!state.peer->recv_ڈیٹا(remote_caps, حجم(remote_caps))) {
            state.peer = خالی_اشاریہ;
            واپس سچ;
        }
        state.peer->get_caps(local_caps);
        اگر (!state.peer->send_ڈیٹا(local_caps, حجم(local_caps))) {
            state.peer = خالی_اشاریہ;
            واپس سچ;
        }
        state.peer->update_caps(remote_caps);
    } ورنہ {
        ثابت std::string host(request.host, strnlen(request.host, حجم(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        برائے (صحیح i = 0; i < 100 && state.peer == خالی_اشاریہ; i++) {
            state.peer = socket_t::connect(host.c_str(), request.پورٹ);
            اگر (state.peer == خالی_اشاریہ) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        اگر (state.peer == خالی_اشاریہ) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.پورٹ);
            واپس سچ;
        }
        state.peer->get_caps(local_caps);
        اگر (!state.peer->send_ڈیٹا(local_caps, حجم(local_caps)) ||
            !state.peer->recv_ڈیٹا(remote_caps, حجم(remote_caps))) {
            state.peer = خالی_اشاریہ;
            واپس سچ;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    واپس سچ;
}

منطقی rpc_server::comm_allreduce(ثابت rpc_پیغام_comm_allreduce_req & request) {
    اگر (request.device >= backends.size()) {
        واپس جھوٹ;
    }
    comm_state & state = comm_حالتیں[request.device];
    اگر (state.peer == خالی_اشاریہ) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        واپس جھوٹ;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_سائز = 16*ggml_tensor_overhead() + 2*ggml_گراف_overhead_custom(8, جھوٹ);
    ساخت ggml_init_params params = {
        /*.mem_size   =*/ ctx_سائز,
        /*.mem_buffer =*/ تہی,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالی_اشاریہ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    اگر (t_dst == خالی_اشاریہ || t_dst->بفر == خالی_اشاریہ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        واپس جھوٹ;
    }
    ثابت size_t  nbytes = ggml_nbytes(t_dst);
    ثابت int64_t ne     = ggml_nelements(t_dst);
    اگر (nbytes == 0) {
        واپس سچ;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ثابت منطقی   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ثابت size_t wire_بائٹیں = wire_bf16 ? (size_t) ne*2 : nbytes;
    ثابت size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    اگر (state.scratch_سائز < need) {
        state.scratch.reset(ggml_backend_alloc_بفر(backend, need));
        state.scratch_سائز = need;
    }
    حرف * scratch_بنیاد = (حرف *) ggml_backend_بفر_get_بنیاد(state.scratch.get());
    state.send_buf.resize(wire_بائٹیں);
    state.recv_buf.resize(wire_بائٹیں);

    خودکار new_scratch_tensor = [&](ggml_type type, size_t انحراف) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->بفر = state.scratch.get();
        t->ڈیٹا   = scratch_بنیاد + انحراف;
        واپس t;
    };
    خودکار new_cpy_گرہ = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->بفر = dst->بفر;
        t->ڈیٹا   = dst->ڈیٹا;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        واپس t;
    };
    خودکار compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_گراف_custom(ctx, 2, جھوٹ);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != خالی_اشاریہ ? 2 : 1;
        ggml_حالت حالت_2 = ggml_backend_گراف_compute_async(backend, graph);
        GGML_ASSERT(حالت_2 == GGML_حالت_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = خالی_اشاریہ;
    ggml_tensor * t_wire_recv = خالی_اشاریہ;
    اگر (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_گرہ(t_dst, t_wire_send), خالی_اشاریہ);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.ڈیٹا(), 0, wire_بائٹیں);
    } ورنہ {
        ggml_backend_tensor_get(t_dst, state.send_buf.ڈیٹا(), 0, wire_بائٹیں);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    اگر (state.rank == 0) {
        اگر (!state.peer->send_ڈیٹا(state.send_buf.ڈیٹا(), wire_بائٹیں) || !state.peer->flush() ||
            !state.peer->recv_ڈیٹا(state.recv_buf.ڈیٹا(), wire_بائٹیں)) {
            واپس جھوٹ;
        }
    } ورنہ {
        اگر (!state.peer->recv_ڈیٹا(state.recv_buf.ڈیٹا(), wire_بائٹیں) ||
            !state.peer->send_ڈیٹا(state.send_buf.ڈیٹا(), wire_بائٹیں) || !state.peer->flush()) {
            واپس جھوٹ;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = خالی_اشاریہ;
    اگر (wire_bf16) {
        ggml_backend_tensor_سیٹ(t_wire_recv, state.recv_buf.ڈیٹا(), 0, wire_بائٹیں);
        t_cast = new_cpy_گرہ(t_wire_recv, t_peer);
    } ورنہ {
        ggml_backend_tensor_سیٹ(t_peer, state.recv_buf.ڈیٹا(), 0, wire_بائٹیں);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_جوڑو;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->بفر = t_dst->بفر;
    t_red->ڈیٹا   = t_dst->ڈیٹا;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    اگر (t_cast != خالی_اشاریہ) {
        compute_nodes(t_cast, t_red);
    } ورنہ {
        compute_nodes(t_red, خالی_اشاریہ);
    }
    واپس سچ;
}

منطقی rpc_server::comm_free(ثابت rpc_پیغام_comm_free_req & request) {
    اگر (request.device >= backends.size()) {
        واپس جھوٹ;
    }
    comm_حالتیں[request.device] = comm_state();
    واپس سچ;
}

منطقی rpc_server::get_device_حافظہ(ثابت rpc_پیغام_get_device_حافظہ_req & request, rpc_پیغام_get_device_حافظہ_rsp & response) {
    uint32_t dev_id = request.device;
    اگر (dev_id >= backends.size()) {
        واپس جھوٹ;
    }
    size_t آزاد, کل;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_حافظہ(dev, &آزاد, &کل);
    response.free_mem = آزاد;
    response.کل_mem = کل;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_سائز(response.free_mem).c_str(), rpc_log_format_سائز(response.کل_mem).c_str());
    واپس سچ;
}

rpc_server::~rpc_server() {
    برائے (خودکار بفر : buffers) {
        ggml_backend_بفر_free(بفر);
    }
}

ساکن خالی rpc_serve_client(ثابت std::vector<ggml_backend_t> & backends, ثابت حرف * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    اگر (!sock->recv_ڈیٹا(&cmd, 1)) {
        واپس;
    }
    اگر (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        واپس;
    }

    // Read input_size and validate protocol version
    uint64_t hello_داخلہ_سائز;
    اگر (!sock->recv_ڈیٹا(&hello_داخلہ_سائز, حجم(hello_داخلہ_سائز))) {
        واپس;
    }

    اگر (hello_داخلہ_سائز != حجم(rpc_پیغام_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_داخلہ_سائز, حجم(rpc_پیغام_hello_req), RPC_PROTO_MAJOR_نسخہ);
        واپس;
    }

    rpc_پیغام_hello_req req = {};
    اگر (!sock->recv_ڈیٹا(&req, حجم(req))) {
        واپس;
    }

    rpc_پیغام_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    اگر (!send_پیغام(sock, &rsp, حجم(rsp))) {
        واپس;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    جب (سچ) {
        اگر (!sock->recv_ڈیٹا(&cmd, 1)) {
            توڑو;
        }
        اگر (cmd >= RPC_CMD_گنتی) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            توڑو;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((شمار rpc_cmd)cmd));
        انتخاب (cmd) {
            حالت RPC_CMD_HELLO: {
                // HELLO command is handled above
                واپس;
            }
            حالت RPC_CMD_DEVICE_گنتی: {
                اگر (!recv_پیغام(sock, خالی_اشاریہ, 0)) {
                    واپس;
                }
                rpc_پیغام_device_گنتی_rsp response;
                response.device_گنتی = backends.size();
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_ALLOC_بفر: {
                rpc_پیغام_alloc_بفر_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_alloc_بفر_rsp response;
                اگر (!server.alloc_بفر(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_GET_ALLOC_سائز: {
                rpc_پیغام_get_alloc_سائز_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_get_alloc_سائز_rsp response;
                اگر (!server.get_alloc_سائز(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_GET_ALIGNMENT: {
                rpc_پیغام_get_alignment_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_get_alignment_rsp response;
                اگر (!server.get_alignment(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_GET_MAX_سائز: {
                rpc_پیغام_get_max_سائز_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_get_max_سائز_rsp response;
                اگر (!server.get_max_سائز(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_بفر_GET_بنیاد: {
                rpc_پیغام_بفر_get_بنیاد_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_بفر_get_بنیاد_rsp response;
                اگر (!server.بفر_get_بنیاد(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_FREE_بفر: {
                rpc_پیغام_free_بفر_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                اگر (!server.free_بفر(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_بفر_CLEAR: {
                rpc_پیغام_بفر_clear_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                اگر (!server.بفر_clear(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_MEMSET_TENSOR: {
                rpc_پیغام_memset_tensor_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                اگر (!server.memset_tensor(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_سیٹ_TENSOR: {
                std::vector<uint8_t> داخلہ;
                اگر (!recv_پیغام(sock, داخلہ)) {
                    واپس;
                }
                اگر (!server.سیٹ_tensor(داخلہ)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_سیٹ_TENSOR_2_D: {
                std::vector<uint8_t> داخلہ;
                اگر (!recv_پیغام(sock, داخلہ)) {
                    واپس;
                }
                اگر (!server.سیٹ_tensor_2d(داخلہ)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_GET_TENSOR_2D: {
                rpc_پیغام_get_tensor_2d_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                std::vector<uint8_t> response;
                اگر (!server.get_tensor_2d(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, response.ڈیٹا(), response.size())) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_سیٹ_TENSOR_HASH: {
                rpc_پیغام_سیٹ_tensor_hash_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_سیٹ_tensor_hash_rsp response;
                اگر (!server.سیٹ_tensor_hash(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_INIT_TENSOR: {
                rpc_پیغام_init_tensor_req request;
                اگر (!recv_پیغام(sock, &request,حجم(request))) {
                    واپس;
                }
                اگر (!server.init_tensor(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_GET_TENSOR: {
                rpc_پیغام_get_tensor_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                std::vector<uint8_t> response;
                اگر (!server.get_tensor(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, response.ڈیٹا(), response.size())) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_COPY_TENSOR: {
                rpc_پیغام_copy_tensor_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_copy_tensor_rsp response;
                اگر (!server.copy_tensor(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_گراف_COMPUTE: {
                std::vector<uint8_t> داخلہ;
                اگر (!recv_پیغام(sock, داخلہ)) {
                    واپس;
                }
                اگر (!server.گراف_compute(داخلہ)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_گراف_RECOMPUTE: {
                rpc_پیغام_گراف_recompute_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                اگر (!server.گراف_recompute(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_COMM_INIT: {
                rpc_پیغام_comm_init_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_comm_init_rsp response;
                اگر (!server.comm_init(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_COMM_ALLREDUCE: {
                rpc_پیغام_comm_allreduce_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                اگر (!server.comm_allreduce(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_COMM_FREE: {
                rpc_پیغام_comm_free_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                اگر (!server.comm_free(request)) {
                    واپس;
                }
                توڑو;
            }
            حالت RPC_CMD_GET_DEVICE_حافظہ: {
                rpc_پیغام_get_device_حافظہ_req request;
                اگر (!recv_پیغام(sock, &request, حجم(request))) {
                    واپس;
                }
                rpc_پیغام_get_device_حافظہ_rsp response;
                اگر (!server.get_device_حافظہ(request, response)) {
                    واپس;
                }
                اگر (!send_پیغام(sock, &response, حجم(response))) {
                    واپس;
                }
                توڑو;
            }
            طےشدہ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((شمار rpc_cmd)cmd));
                واپس;
            }
        }
    }
}

خالی ggml_backend_rpc_آغاز_server(ثابت حرف * endpoint, ثابت حرف * cache_dir,
                                   size_t n_دھاگے, size_t n_devices, ggml_backend_dev_t * devices) {
    اگر (n_devices == 0 || devices == خالی_اشاریہ) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        واپس;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_نسخہ,
        RPC_PROTO_MINOR_نسخہ,
        RPC_PROTO_PATCH_نسخہ);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    برائے (size_t i = 0; i < n_devices; i++) {
        خودکار dev = devices[i];
        size_t آزاد, کل;
        ggml_backend_dev_حافظہ(dev, &آزاد, &کل);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 کل / 1024 / 1024, آزاد / 1024 / 1024);
        خودکار backend = ggml_backend_dev_init(dev, خالی_اشاریہ);
        اگر (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            واپس;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : خالی_اشاریہ;
        اگر (reg) {
            خودکار ggml_backend_سیٹ_n_دھاگے_fn = (ggml_backend_سیٹ_n_دھاگے_t) ggml_backend_reg_get_proc_پتہ(reg, "ggml_backend_set_n_threads");
            اگر (ggml_backend_سیٹ_n_دھاگے_fn) {
                ggml_backend_سیٹ_n_دھاگے_fn(backend, n_دھاگے);
            }
        }
    }

    std::string host;
    صحیح پورٹ;
    اگر (!parse_endpoint(endpoint, host, پورٹ)) {
        واپس;
    }

#اگر_تعریف_شدہ GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#ورنہ
    LOG_INFO("  transport      : TCP\n");
#اگر_ختم // GGML_RPC_RDMA
    اگر (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        واپس;
    }
    خودکار server_socket = socket_t::create_server(host.c_str(), پورٹ);
    اگر (server_socket == خالی_اشاریہ) {
        LOG_ERROR("Failed to create server socket\n");
        واپس;
    }
    جب (سچ) {
        خودکار client_socket = server_socket->accept();
        اگر (client_socket == خالی_اشاریہ) {
            LOG_ERROR("Failed to accept client connection\n");
            واپس;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    برائے (خودکار backend : backends) {
        ggml_backend_free(backend);
    }
}

ساکن ثابت حرف * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    واپس ctx->name.c_str();
}

ساکن ثابت حرف * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    واپس ctx->description.c_str();
}

ساکن خالی ggml_backend_rpc_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_حافظہ(ctx->endpoint.c_str(), ctx->device, آزاد, کل);
}

ساکن شمار ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    واپس GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_غیر_مستعمل(dev);
}

ساکن خالی ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ جھوٹ,
        /* .buffer_from_host_ptr  = */ جھوٹ,
        /* .events                = */ سچ,
        /* .mmap_support          = */ سچ,
    };
}

ساکن ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    واپس ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_غیر_مستعمل(params);
}

ساکن ggml_backend_بفر_type_t ggml_backend_rpc_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    واپس ggml_backend_rpc_بفر_type(ctx->endpoint.c_str(), ctx->device);

    GGML_غیر_مستعمل(dev);
}

ساکن منطقی ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op) {
    GGML_غیر_مستعمل(dev);
    GGML_غیر_مستعمل(op);
    //TODO: call the remote backend and cache the results
    واپس سچ;
}

ساکن منطقی ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    اگر (!buft || buft->iface.get_name != ggml_backend_rpc_بفر_type_name) {
        واپس جھوٹ;
    }
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    واپس buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ساکن ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    خودکار dispatcher = get_dispatcher(ctx->endpoint);
    واپس dispatcher->event_new(dev);
}

ساکن خالی ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    خودکار dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ساکن خالی ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    خودکار dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ساکن ثابت ساخت ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_بفر_type,
    /* .get_host_buffer_type = */ تہی,
    /* .buffer_from_host_ptr = */ تہی,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ تہی,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ساخت ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ساکن ثابت حرف * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    واپس ctx ? ctx->name.c_str() : "RPC";
}

ساکن size_t ggml_backend_rpc_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    واپس ctx ? ctx->devices.size() : 0;
}

ساکن ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    اگر (ctx == خالی_اشاریہ) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } ورنہ {
        GGML_ASSERT(اشاریہ < ctx->devices.size());
        واپس ctx->devices[اشاریہ];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ساخت ggml_backend_rpc_comm_context {
    ساخت rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ساکن خالی ggml_backend_rpc_comm_free(خالی * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    اگر (comm_ctx == خالی_اشاریہ) {
        واپس;
    }
    برائے (ثابت خودکار & rank : comm_ctx->ranks) {
        خودکار request = std::make_shared<rpc_پیغام_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, حجم(*request));
        rank.dispatcher->busy_spin_release();
    }
    حذف comm_ctx;
}

ساکن خالی * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    اگر (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != خالی_اشاریہ) {
        واپس خالی_اشاریہ;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    برائے (size_t i = 0; i < n_backends; i++) {
        اگر (!ggml_backend_is_rpc(backends[i])) {
            واپس خالی_اشاریہ;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        برائے (ثابت خودکار & rank : ranks) {
            اگر (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                واپس خالی_اشاریہ;
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
        واپس خالی_اشاریہ;
    }
    ثابت uint32_t comm_پورٹ = (uint32_t) port0 + 1000;
    اگر (host0.size() >= 64) {
        واپس خالی_اشاریہ;
    }

    برائے (ثابت خودکار & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_پیغام_comm_init_rsp> responses(n_backends);
    برائے (size_t i = 0; i < n_backends; i++) {
        خودکار request = std::make_shared<rpc_پیغام_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->پورٹ   = comm_پورٹ;
        اگر (i > 0) {
            حافظہ_نقل(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, حجم(*request), &responses[i], حجم(responses[i]));
    }
    برائے (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    منطقی ok = سچ;
    برائے (size_t i = 0; i < n_backends; i++) {
        اگر (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = جھوٹ;
        }
    }
    اگر (!ok) {
        برائے (ثابت خودکار & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        واپس خالی_اشاریہ;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    واپس نیا ggml_backend_rpc_comm_context{std::move(ranks)};
}

ساکن منطقی ggml_backend_rpc_comm_allreduce_tensor(خالی * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    اگر (comm_ctx == خالی_اشاریہ) {
        واپس جھوٹ;
    }
    ثابت size_t n_ranks = comm_ctx->ranks.size();
    ثابت int64_t ne = ggml_nelements(tensors[0]);
    اگر (ne == 0) {
        واپس سچ;
    }
    برائے (size_t i = 0; i < n_ranks; i++) {
        اگر (tensors[i] == خالی_اشاریہ || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->بفر == خالی_اشاریہ || !ggml_backend_بفر_is_rpc(tensors[i]->بفر)) {
            واپس جھوٹ;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        اگر ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            واپس جھوٹ;
        }
    }
    برائے (size_t i = 0; i < n_ranks; i++) {
        خودکار request = std::make_shared<rpc_پیغام_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, حجم(*request));
    }
    واپس سچ;
}

ساکن خالی * ggml_backend_rpc_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    اگر (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        واپس (خالی *)ggml_backend_rpc_جوڑو_server;
    }
    اگر (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        واپس (خالی *)ggml_backend_rpc_آغاز_server;
    }
    اگر (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        واپس (خالی *)ggml_backend_rpc_comm_init;
    }
    اگر (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        واپس (خالی *)ggml_backend_rpc_comm_free;
    }
    اگر (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        واپس (خالی *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    واپس تہی;

    GGML_غیر_مستعمل(reg);
}

ساکن ثابت ساخت ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_rpc_reg(خالی) {
    ساکن ساخت ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ تہی,
    };

    واپس &ggml_backend_rpc_reg;
}

ساکن uint32_t ggml_backend_rpc_get_device_گنتی(ثابت حرف * endpoint) {
    خودکار dispatcher = get_dispatcher(endpoint);
    rpc_پیغام_device_گنتی_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_گنتی, خالی_اشاریہ, 0, &response, حجم(response));
    واپس response.device_گنتی;
}

ساکن ثابت ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_گنتی,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_rpc_جوڑو_server(ثابت حرف * endpoint) {
    ساکن std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ساکن std::mutex mutex;
    ساکن uint32_t dev_id = 0;
    std::lock_guard<std::mutex> تالا(mutex);
    اگر (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        واپس reg_map[endpoint];
    }
    uint32_t dev_گنتی = ggml_backend_rpc_get_device_گنتی(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_گنتی);
    اگر (dev_گنتی == 0) {
        واپس خالی_اشاریہ;
    }
    ggml_backend_rpc_reg_context * ctx = نیا ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    برائے (uint32_t ind = 0; ind < dev_گنتی; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = نیا ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = نیا ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = نیا ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_نسخہ,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    واپس reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
