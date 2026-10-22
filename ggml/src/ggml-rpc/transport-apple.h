#pragma once

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>

संरचना apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    स्थिर std::unique_ptr<apple_rdma> probe(पूर्णाङ्क fd, ध्रुव uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ध्रुव uint8_t * caps);

    bool send(ध्रुव शून्य * data, माप_प्रकार size);
    bool recv(शून्य * data, माप_प्रकार size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ध्रुव;

गुप्त:
    संरचना impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
