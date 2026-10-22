#आशय once

#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <memory>

काष्ठा socket_t;
प्रकार std::shared_ptr<socket_t> socket_ptr;

अटल constexpr माप_प्रकार MAX_CHUNK_आकार = 1024ull * 1024ull * 1024ull; // 1 GiB
अटल constexpr माप_प्रकार RPC_CONN_CAPS_आकार = 24;

काष्ठा socket_t {
    ~socket_t();

    bool send_तथ्य(स्थिर व्योम * तथ्य, माप_प्रकार आकार);
    bool recv_तथ्य(व्योम * तथ्य, माप_प्रकार आकार);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    व्योम get_caps(uint8_t * local_caps);
    व्योम update_caps(स्थिर uint8_t * remote_caps);
    स्थिर अक्षर * transport_name() स्थिर;

    अटल socket_ptr create_server(स्थिर अक्षर * host, पूर्णांक port);
    अटल socket_ptr connect(स्थिर अक्षर * host, पूर्णांक port);

निजी:
    काष्ठा impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
व्योम rpc_transport_shutdown();
