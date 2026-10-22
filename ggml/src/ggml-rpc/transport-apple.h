#pragma once

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <memory>

ସଂରଚନା apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    ସ୍ଥିର std::unique_ptr<apple_rdma> probe(ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, ଅଚଳ uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ଅଚଳ uint8_t * caps);

    bool send(ଅଚଳ ଶୂନ୍ୟ * data, size_t size);
    bool recv(ଶୂନ୍ୟ * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ଅଚଳ;

ବ୍ୟକ୍ତିଗତ:
    ସଂରଚନା impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
