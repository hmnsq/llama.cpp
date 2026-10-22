#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>

ਬਣਤਰ socket_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::shared_ptr<socket_t> socket_ptr;

ਸਥਿਰ constexpr size_t MAX_CHUNK_ਮਾਪ = 1024ull * 1024ull * 1024ull; // 1 GiB
ਸਥਿਰ constexpr size_t RPC_CONN_CAPS_ਮਾਪ = 24;

ਬਣਤਰ socket_t {
    ~socket_t();

    bool send_ਅੰਕੜੇ(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);
    bool recv_ਅੰਕੜੇ(ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    ਖਾਲੀ get_caps(uint8_t * local_caps);
    ਖਾਲੀ update_caps(ਅਚਲ uint8_t * remote_caps);
    ਅਚਲ ਅੱਖਰ * transport_name() ਅਚਲ;

    ਸਥਿਰ socket_ptr create_server(ਅਚਲ ਅੱਖਰ * host, ਪੂਰਨ ਪੋਰਟ);
    ਸਥਿਰ socket_ptr connect(ਅਚਲ ਅੱਖਰ * host, ਪੂਰਨ ਪੋਰਟ);

ਨਿਜੀ:
    ਬਣਤਰ impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
ਖਾਲੀ rpc_transport_shutdown();
