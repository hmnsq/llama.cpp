#pragma once

#شامل <cstddef>
#شامل <cstdint>
#شامل <memory>

ساختار socket_t;
تعریفنوع std::shared_ptr<socket_t> socket_ptr;

ایستا constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
ایستا constexpr size_t RPC_CONN_CAPS_SIZE = 24;

ساختار socket_t {
    ~socket_t();

    bool send_data(ثابت تهی * data, size_t size);
    bool recv_data(تهی * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    تهی get_caps(uint8_t * local_caps);
    تهی update_caps(ثابت uint8_t * remote_caps);
    ثابت نویسه * transport_name() ثابت;

    ایستا socket_ptr create_server(ثابت نویسه * host, صحیح port);
    ایستا socket_ptr connect(ثابت نویسه * host, صحیح port);

خصوصی:
    ساختار impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
تهی rpc_transport_shutdown();
