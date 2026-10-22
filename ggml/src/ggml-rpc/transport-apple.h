#pragma once

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <memory>

கட்டமைப்பு apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    நிலையான std::unique_ptr<apple_rdma> probe(முழு fd, மாறிலி uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(மாறிலி uint8_t * caps);

    bool send(மாறிலி வெறுமை * data, size_t size);
    bool recv(வெறுமை * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() மாறிலி;

தனிப்பட்ட:
    கட்டமைப்பு impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
