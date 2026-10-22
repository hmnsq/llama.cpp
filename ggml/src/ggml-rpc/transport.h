#pragma once

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <memory>

ସଂରଚନା socket_t;
ପ୍ରକାରବ୍ୟାଖ୍ୟା std::shared_ptr<socket_t> socket_ptr;

ସ୍ଥିର constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
ସ୍ଥିର constexpr size_t RPC_CONN_CAPS_SIZE = 24;

ସଂରଚନା socket_t {
    ~socket_t();

    bool send_data(ଅଚଳ ଶୂନ୍ୟ * data, size_t size);
    bool recv_data(ଶୂନ୍ୟ * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    ଶୂନ୍ୟ get_caps(uint8_t * local_caps);
    ଶୂନ୍ୟ update_caps(ଅଚଳ uint8_t * remote_caps);
    ଅଚଳ ଅକ୍ଷର * transport_name() ଅଚଳ;

    ସ୍ଥିର socket_ptr create_server(ଅଚଳ ଅକ୍ଷର * host, ପୂର୍ଣ୍ଣସଂଖ୍ୟା port);
    ସ୍ଥିର socket_ptr connect(ଅଚଳ ଅକ୍ଷର * host, ପୂର୍ଣ୍ଣସଂଖ୍ୟା port);

ବ୍ୟକ୍ତିଗତ:
    ସଂରଚନା impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
ଶୂନ୍ୟ rpc_transport_shutdown();
