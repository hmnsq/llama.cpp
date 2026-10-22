#pragma once

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <memory>

সংরচনা socket_t;
প্রকারনির্দেশ std::shared_ptr<socket_t> socket_ptr;

স্থির constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
স্থির constexpr size_t RPC_CONN_CAPS_SIZE = 24;

সংরচনা socket_t {
    ~socket_t();

    bool send_data(ধ্রুব শূন্য * data, size_t size);
    bool recv_data(শূন্য * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    শূন্য get_caps(uint8_t * local_caps);
    শূন্য update_caps(ধ্রুব uint8_t * remote_caps);
    ধ্রুব বর্ণ * transport_name() ধ্রুব;

    স্থির socket_ptr create_server(ধ্রুব বর্ণ * host, পূর্ণাঙ্ক port);
    স্থির socket_ptr connect(ধ্রুব বর্ণ * host, পূর্ণাঙ্ক port);

গুপ্ত:
    সংরচনা impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
শূন্য rpc_transport_shutdown();
