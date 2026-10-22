#समावेश "transport-apple.h"
#समावेश "transport.h"
#समावेश "log.h"

#समावेश <infiniband/verbs.h>

#समावेश <cerrno>
#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <string>
#समावेश <dlfcn.h>
#समावेश <poll.h>
#समावेश <sys/socket.h>
#समावेश <unistd.h>

// Apple RDMA-over-Thunderbolt (see Apple TN3205).
//
// Apple's RDMA is quite different from what's supported in Linux - deserving of its own transport implementation.
// see https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt for details
// at a high level the main differences are:
// UC(unreliable connection) on Apple vs RC(reliable connection) QP transport types on Linux (though in practice UC on Apple is still lossless)
// fixed 128KiB stride on Apple vs variable chunk size on Linux
// relying on Apple's hardware credit based flow control vs RNR NAKs + retries on Linux
//
// on Apple a SEND and its corresponding RECV must cover the same number of 4 KiB Thunderbolt frames,
// so every SEND posts a whole 128KiB stride over the wire, even when partially filled.
// (In testing 128KiB was the best performing among 32, 64, 128, 256)

//TODO: add mechanism similar to https://github.com/ggml-org/llama.cpp/pull/29440 to prevent idle CPU from spinning
अटल constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
अटल constexpr पूर्णांक      RDMA_NBUF        = 16;          // ring depth (frames per direction)
अटल constexpr माप_प्रकार   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
अटल constexpr माप_प्रकार   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
अटल constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
अटल constexpr माप_प्रकार   RDMA_GID_आकार    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
अटल constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
अटल constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
अटल constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
अटल constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

काष्ठा rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
अटल constexpr माप_प्रकार RDMA_PAYLOAD = RDMA_STRIDE - माप(rdma_seg_hdr);

काष्ठा apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_आकार];
};

static_assert(माप(apple_rdma_caps) == RPC_CONN_CAPS_आकार, "apple_rdma_caps must match conn_caps size");

काष्ठा apple_rdma::impl {
    पूर्णांक fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    काष्ठा ibv_context * ctx = nullptr;
    काष्ठा ibv_pd * pd = nullptr;
    काष्ठा ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    काष्ठा ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    काष्ठा ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    काष्ठा ibv_mr * recv_mr  = nullptr;

    पूर्णांक      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    काष्ठा { पूर्णांक buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    पूर्णांक      inq_शीर्ष = 0;
    पूर्णांक      inq_गिनती = 0;
    पूर्णांक      pend_buf = -1;
    uint32_t pend_len = 0;
    bool     broken = false;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    पूर्णांक          gid_idx = 0;
    क्रमागत ibv_mtu path_mtu = IBV_MTU_1024;

    पूर्णांक  progress();
    bool acquire_pending();
    bool post_pending();

    bool post_recv(पूर्णांक i) {
        काष्ठा ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (माप_प्रकार)i * RDMA_STRIDE);
        sge.लंबाई = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        काष्ठा ibv_recv_wr wr = {}, * बुरा = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;
        वापस ibv_post_recv(qp, &wr, &बुरा) == 0;
    }

    bool post_send(पूर्णांक i, माप_प्रकार len) {
        काष्ठा ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (माप_प्रकार)i * RDMA_STRIDE);
        sge.लंबाई = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        काष्ठा ibv_send_wr wr = {}, * बुरा = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        वापस ibv_post_send(qp, &wr, &बुरा) == 0;
    }

    ~impl() {
        broken = true;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        अगर (qp) ibv_destroy_qp(qp);
        अगर (send_mr) ibv_dereg_mr(send_mr);
        अगर (recv_mr) ibv_dereg_mr(recv_mr);
        मुक्त(send_mem);
        मुक्त(recv_mem);
        अगर (cq)  ibv_destroy_cq(cq);
        अगर (pd)  ibv_dealloc_pd(pd);
        अगर (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = शेष;

bool apple_rdma::broken() स्थिर {
    वापस pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
अटल bool tcp_send_byte(पूर्णांक fd, uint8_t b) {
    ssize_t n;
    करो { n = ::send(fd, &b, माप(b), 0); } जबतक (n < 0 && त्रुटि_सं == EINTR);
    वापस n == माप(b);
}

अटल bool tcp_recv_byte(पूर्णांक fd, uint8_t * b) {
    ssize_t n;
    करो { n = ::recv(fd, b, माप(*b), 0); } जबतक (n < 0 && त्रुटि_सं == EINTR);
    वापस n == (ssize_t)माप(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
अटल पूर्णांक rdma_match_gid(काष्ठा ibv_context * ctx, uint8_t port, पूर्णांक gid_tbl_len,
                          स्थिर uint8_t * target, जोड़ ibv_gid * out) {
    क्रम (पूर्णांक i = 0; i < gid_tbl_len; i++) {
        जोड़ ibv_gid g;
        अगर (ibv_query_gid(ctx, port, i, &g) != 0) जारी;
        अगर (स_भेद(g.raw, target, RDMA_GID_आकार) != 0) जारी;
        अगर (out) *out = g;
        वापस i;
    }
    वापस -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
अटल uint8_t rdma_first_active_port(काष्ठा ibv_context * ctx, काष्ठा ibv_port_attr * out) {
    काष्ठा ibv_device_attr da;
    अगर (ibv_query_device(ctx, &da) != 0) वापस 0;
    क्रम (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        काष्ठा ibv_port_attr pa;
        अगर (ibv_query_port(ctx, p, &pa) != 0) जारी;
        अगर (pa.state == IBV_PORT_ACTIVE) { अगर (out) *out = pa; वापस p; }
    }
    वापस 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
अटल bool rdma_library_present() {
    अटल स्थिर bool present = [] {
        व्योम * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        अगर (handle == nullptr) {
            वापस false;
        }
        dlclose(handle);
        वापस true;
    }();
    वापस present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(पूर्णांक fd, स्थिर uint8_t * target_gid, uint8_t * caps) {
    अगर (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        वापस nullptr;
    }
    पूर्णांक ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    अगर (!devs) वापस nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    काष्ठा ibv_port_attr pa = {};
    जोड़ ibv_gid gid = {};
    पूर्णांक gid_idx = -1;
    std::string matched;
    क्रम (पूर्णांक d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        अगर (!c) जारी;
        काष्ठा ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        पूर्णांक gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        अगर (gi < 0) { ibv_close_device(c); जारी; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        स्थिर अक्षर * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        अवरोध;
    }
    ibv_free_device_list(devs);
    अगर (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        वापस nullptr;
    }

    std::unique_ptr<impl> c(नया impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    अगर (!c->pd) वापस nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    अगर (!c->cq) वापस nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    अगर (!c->qp) वापस nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_सूचकांक = 0;
        a.port_अंक = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        अगर (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_सूचकांक | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            वापस nullptr;
        }
    }

    दीर्घ page = sysconf(_SC_PAGESIZE);
    अगर (page <= 0) page = 4096;
    स्थिर माप_प्रकार ring_bytes = (माप_प्रकार)RDMA_NBUF * RDMA_STRIDE;
    अगर (posix_memalign((व्योम **)&c->send_mem, (माप_प्रकार)page, ring_bytes) != 0) c->send_mem = nullptr;
    अगर (posix_memalign((व्योम **)&c->recv_mem, (माप_प्रकार)page, ring_bytes) != 0) c->recv_mem = nullptr;
    अगर (!c->send_mem || !c->recv_mem) वापस nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    स्थिर पूर्णांक mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    अगर (!c->send_mr || !c->recv_mr) वापस nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_अंक;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    स_नकल(rc.gid, gid.raw, RDMA_GID_आकार);
    स_नकल(caps, &rc, माप(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (अचिन्हित)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    वापस std::unique_ptr<apple_rdma>(नया apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
bool apple_rdma::activate(स्थिर uint8_t * caps) {
    impl * c = pimpl.लो();

    apple_rdma_caps rc = {};
    स_नकल(&rc, caps, माप(rc));

    bool ok = true;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_अंक = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_अंक      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_सूचकांक = (uint8_t)c->gid_idx;
        स_नकल(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_आकार);
        अगर (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", म_त्रुटि(त्रुटि_सं));
            ok = false;
        }
    }
    अगर (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        अगर (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", म_त्रुटि(त्रुटि_सं));
            ok = false;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    क्रम (पूर्णांक i = 0; ok && i < RDMA_NBUF; i++) {
        अगर (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = false;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    अगर (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        वापस false;
    }
    अगर (!ok || peer_ready != RDMA_SYNC_READY) {
        वापस false;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    वापस true;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
पूर्णांक apple_rdma::impl::progress() {
    काष्ठा ibv_wc wc[RDMA_NBUF * 2];
    पूर्णांक n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    अगर (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = true; वापस -1; }
    क्रम (पूर्णांक j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        bool is_recv = (id & RDMA_RECV_WR) != 0;
        अगर (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = true;
            वापस -1;
        }
        अगर (is_recv) {
            पूर्णांक b = (पूर्णांक)(id & RDMA_WR_IDX_MASK);
            स्थिर rdma_seg_hdr * h = (स्थिर rdma_seg_hdr *)(recv_mem + (माप_प्रकार)b * RDMA_STRIDE);
            अगर (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = true; वापस -1; }
            अगर (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = true; वापस -1; }
            पूर्णांक slot = (inq_शीर्ष + inq_गिनती) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_गिनती++;
        } अन्यथा {
            send_busy[(पूर्णांक)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    वापस n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
bool apple_rdma::impl::acquire_pending() {
    अगर (pend_buf >= 0) वापस true;
    क्रम (;;) {
        अगर (broken) वापस false;
        क्रम (पूर्णांक k = 0; k < RDMA_NBUF; k++) अगर (!send_busy[k]) { pend_buf = k; pend_len = 0; वापस true; }
        अगर (progress() < 0) वापस false;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
bool apple_rdma::impl::post_pending() {
    अगर (pend_buf < 0) वापस true;
    पूर्णांक i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (माप_प्रकार)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    अगर (!post_send(i, RDMA_STRIDE)) { broken = true; वापस false; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    वापस true;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
bool apple_rdma::send(स्थिर व्योम * तथ्य, माप_प्रकार size) {
    impl * c = pimpl.लो();
    स्थिर uint8_t * p = (स्थिर uint8_t *)तथ्य;
    जबतक (size > 0) {
        अगर (c->broken) वापस false;
        अगर (!c->acquire_pending()) वापस false;
        uint8_t * sb = c->send_mem + (माप_प्रकार)c->pend_buf * RDMA_STRIDE;
        माप_प्रकार space = RDMA_PAYLOAD - c->pend_len;
        माप_प्रकार chunk = size < space ? size : space;
        स_नकल(sb + माप(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        अगर (c->pend_len == RDMA_PAYLOAD) { अगर (!c->post_pending()) वापस false; }
    }
    वापस true;
}

bool apple_rdma::recv(व्योम * तथ्य, माप_प्रकार size) {
    impl * c = pimpl.लो();
    uint8_t * p = (uint8_t *)तथ्य;
    अगर (!c->post_pending()) वापस false;   // turnaround: flush the coalesced request
    अचिन्हित idle = 0;
    जबतक (size > 0) {
        अगर (c->inq_गिनती == 0) {
            अगर (c->broken) वापस false;
            पूर्णांक n = c->progress();
            अगर (n < 0) वापस false;
            अगर (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                अगर ((++idle & 0xFFFFF) == 0) {
                    काष्ठा pollfd pfd = { c->fd, POLLIN, 0 };
                    अगर (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        वापस false;
                    }
                }
            } अन्यथा {
                idle = 0;
            }
            जारी;
        }
        idle = 0;
        पूर्णांक slot = c->inq_शीर्ष;
        पूर्णांक b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (माप_प्रकार)avail) ? (uint32_t)size : avail;
        स_नकल(p, c->recv_mem + (माप_प्रकार)b * RDMA_STRIDE + माप(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        अगर (c->inq[slot].off == c->inq[slot].len) {
            अगर (!c->post_recv(b)) { c->broken = true; वापस false; }
            c->inq_शीर्ष = (c->inq_शीर्ष + 1) % RDMA_NBUF;
            c->inq_गिनती--;
        }
    }
    वापस true;
}

bool apple_rdma::सफाई() {
    वापस pimpl->post_pending();
}
