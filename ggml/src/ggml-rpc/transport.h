#pragma once

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <memory>

কাঠামো socket_t;
প্রকারলেখ std::shared_ptr<socket_t> socket_ptr;

স্থির constexpr size_t MAX_CHUNK_আকার = 1024ull * 1024ull * 1024ull; // 1 GiB
স্থির constexpr size_t RPC_CONN_CAPS_আকার = 24;

কাঠামো socket_t {
    ~socket_t();

    bool send_তথ্য(ধ্রুবক শূন্য * তথ্য, size_t আকার_2);
    bool recv_তথ্য(শূন্য * তথ্য, size_t আকার_2);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    শূন্য get_caps(uint8_t * local_caps);
    শূন্য update_caps(ধ্রুবক uint8_t * remote_caps);
    ধ্রুবক অক্ষর * transport_name() ধ্রুবক;

    স্থির socket_ptr create_server(ধ্রুবক অক্ষর * host, পূর্ণসংখ্যা port);
    স্থির socket_ptr connect(ধ্রুবক অক্ষর * host, পূর্ণসংখ্যা port);

ব্যক্তিগত:
    কাঠামো impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
শূন্য rpc_transport_shutdown();
