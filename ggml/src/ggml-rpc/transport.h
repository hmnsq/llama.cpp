#ہدایت once

#شامل <cstddef>
#شامل <cstdint>
#شامل <memory>

ساخت socket_t;
قسمتعریف std::shared_ptr<socket_t> socket_ptr;

ساکن constexpr size_t MAX_CHUNK_سائز = 1024ull * 1024ull * 1024ull; // 1 GiB
ساکن constexpr size_t RPC_CONN_CAPS_سائز = 24;

ساخت socket_t {
    ~socket_t();

    bool send_ڈیٹا(ثابت خالی * ڈیٹا, size_t سائز);
    bool recv_ڈیٹا(خالی * ڈیٹا, size_t سائز);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    خالی get_caps(uint8_t * local_caps);
    خالی update_caps(ثابت uint8_t * remote_caps);
    ثابت حرف * transport_name() ثابت;

    ساکن socket_ptr create_server(ثابت حرف * host, صحیح پورٹ);
    ساکن socket_ptr connect(ثابت حرف * host, صحیح پورٹ);

نجی:
    ساخت impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
خالی rpc_transport_shutdown();
