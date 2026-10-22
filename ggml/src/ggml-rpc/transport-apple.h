#pragma once

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <memory>

ഘടന apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    സ്ഥിതം std::unique_ptr<apple_rdma> probe(പൂർണ്ണസംഖ്യ fd, സ്ഥിരം uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(സ്ഥിരം uint8_t * caps);

    bool send(സ്ഥിരം ശൂന്യം * data, size_t size);
    bool recv(ശൂന്യം * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() സ്ഥിരം;

സ്വകാര്യ:
    ഘടന impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
