#आशय once

#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <memory>

काष्ठा apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    अटल std::unique_ptr<apple_rdma> probe(पूर्णांक fd, स्थिर uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(स्थिर uint8_t * caps);

    bool send(स्थिर व्योम * तथ्य, माप_प्रकार आकार);
    bool recv(व्योम * तथ्य, माप_प्रकार आकार);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() स्थिर;

निजी:
    काष्ठा impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
