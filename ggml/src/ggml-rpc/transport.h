#pragma once

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <memory>

ഘടന socket_t;
തരനിർവചനം std::shared_ptr<socket_t> socket_ptr;

സ്ഥിതം constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
സ്ഥിതം constexpr size_t RPC_CONN_CAPS_SIZE = 24;

ഘടന socket_t {
    ~socket_t();

    bool send_data(സ്ഥിരം ശൂന്യം * data, size_t size);
    bool recv_data(ശൂന്യം * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    ശൂന്യം get_caps(uint8_t * local_caps);
    ശൂന്യം update_caps(സ്ഥിരം uint8_t * remote_caps);
    സ്ഥിരം അക്ഷരം * transport_name() സ്ഥിരം;

    സ്ഥിതം socket_ptr create_server(സ്ഥിരം അക്ഷരം * host, പൂർണ്ണസംഖ്യ port);
    സ്ഥിതം socket_ptr connect(സ്ഥിരം അക്ഷരം * host, പൂർണ്ണസംഖ്യ port);

സ്വകാര്യ:
    ഘടന impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
ശൂന്യം rpc_transport_shutdown();
