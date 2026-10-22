#pragma once

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <memory>

கட்டமைப்பு socket_t;
வகைவரை std::shared_ptr<socket_t> socket_ptr;

நிலையான constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
நிலையான constexpr size_t RPC_CONN_CAPS_SIZE = 24;

கட்டமைப்பு socket_t {
    ~socket_t();

    bool send_data(மாறிலி வெறுமை * data, size_t size);
    bool recv_data(வெறுமை * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    வெறுமை get_caps(uint8_t * local_caps);
    வெறுமை update_caps(மாறிலி uint8_t * remote_caps);
    மாறிலி எழுத்து * transport_name() மாறிலி;

    நிலையான socket_ptr create_server(மாறிலி எழுத்து * host, முழு port);
    நிலையான socket_ptr connect(மாறிலி எழுத்து * host, முழு port);

தனிப்பட்ட:
    கட்டமைப்பு impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
வெறுமை rpc_transport_shutdown();
