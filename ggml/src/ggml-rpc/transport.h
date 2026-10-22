#ہدایت once

#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <memory>

بنية socket_t;
تعريفنوع std::shared_ptr<socket_t> socket_ptr;

ساكن constexpr size_t MAX_CHUNK_ناپ = 1024ull * 1024ull * 1024ull; // 1 GiB
ساكن constexpr size_t RPC_CONN_CAPS_ناپ = 24;

بنية socket_t {
    ~socket_t();

    bool send_انکڑے(ثابت فراغ * انکڑے, size_t ناپ);
    bool recv_انکڑے(فراغ * انکڑے, size_t ناپ);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    فراغ get_caps(uint8_t * local_caps);
    فراغ update_caps(ثابت uint8_t * remote_caps);
    ثابت حرف * transport_name() ثابت;

    ساكن socket_ptr create_server(ثابت حرف * host, صحيح پورٹ);
    ساكن socket_ptr connect(ثابت حرف * host, صحيح پورٹ);

خاص:
    بنية impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
فراغ rpc_transport_shutdown();
