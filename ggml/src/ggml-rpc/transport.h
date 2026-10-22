#pragma once

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <memory>

સંરચના socket_t;
પ્રકારવ્યાખ્યા std::shared_ptr<socket_t> socket_ptr;

સ્થિર constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
સ્થિર constexpr size_t RPC_CONN_CAPS_SIZE = 24;

સંરચના socket_t {
    ~socket_t();

    bool send_data(અચલ શૂન્ય * data, size_t size);
    bool recv_data(શૂન્ય * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    શૂન્ય get_caps(uint8_t * local_caps);
    શૂન્ય update_caps(અચલ uint8_t * remote_caps);
    અચલ અક્ષર * transport_name() અચલ;

    સ્થિર socket_ptr create_server(અચલ અક્ષર * host, પૂર્ણાંક port);
    સ્થિર socket_ptr connect(અચલ અક્ષર * host, પૂર્ણાંક port);

ખાનગી:
    સંરચના impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
શૂન્ય rpc_transport_shutdown();
