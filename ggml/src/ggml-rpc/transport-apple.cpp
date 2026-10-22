#അന്തര്ഭാവ "transport-apple.h"
#അന്തര്ഭാവ "transport.h"
#അന്തര്ഭാവ "log.h"

#അന്തര്ഭാവ <infiniband/verbs.h>

#അന്തര്ഭാവ <cerrno>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <dlfcn.h>
#അന്തര്ഭാവ <poll.h>
#അന്തര്ഭാവ <sys/socket.h>
#അന്തര്ഭാവ <unistd.h>

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
സ്ഥിതം constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ      RDMA_NBUF        = 16;          // ring depth (frames per direction)
സ്ഥിതം constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
സ്ഥിതം constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
സ്ഥിതം constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
സ്ഥിതം constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
സ്ഥിതം constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
സ്ഥിതം constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
സ്ഥിതം constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
സ്ഥിതം constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

ഘടന rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
സ്ഥിതം constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - വലുപ്പം(rdma_seg_hdr);

ഘടന apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(വലുപ്പം(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

ഘടന apple_rdma::impl {
    പൂർണ്ണസംഖ്യ fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    ഘടന ibv_context * ctx = nullptr;
    ഘടന ibv_pd * pd = nullptr;
    ഘടന ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    ഘടന ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    ഘടന ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    ഘടന ibv_mr * recv_mr  = nullptr;

    പൂർണ്ണസംഖ്യ      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    ഘടന { പൂർണ്ണസംഖ്യ buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    പൂർണ്ണസംഖ്യ      inq_head = 0;
    പൂർണ്ണസംഖ്യ      inq_എണ്ണം = 0;
    പൂർണ്ണസംഖ്യ      pend_buf = -1;
    uint32_t pend_len = 0;
    തർക്കം     broken = അസത്യം;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    പൂർണ്ണസംഖ്യ          gid_idx = 0;
    എണ്ണൽ ibv_mtu path_mtu = IBV_MTU_1024;

    പൂർണ്ണസംഖ്യ  progress();
    തർക്കം acquire_pending();
    തർക്കം post_pending();

    തർക്കം post_recv(പൂർണ്ണസംഖ്യ i) {
        ഘടന ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        ഘടന ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.അക്കം_sge = 1;
        തിരികെ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    തർക്കം post_send(പൂർണ്ണസംഖ്യ i, size_t len) {
        ഘടന ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        ഘടന ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.അക്കം_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        തിരികെ ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = സത്യം;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        എങ്കിൽ (qp) ibv_destroy_qp(qp);
        എങ്കിൽ (send_mr) ibv_dereg_mr(send_mr);
        എങ്കിൽ (recv_mr) ibv_dereg_mr(recv_mr);
        സ്വതന്ത്രം(send_mem);
        സ്വതന്ത്രം(recv_mem);
        എങ്കിൽ (cq)  ibv_destroy_cq(cq);
        എങ്കിൽ (pd)  ibv_dealloc_pd(pd);
        എങ്കിൽ (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = സ്ഥിരസ്ഥിതി;

തർക്കം apple_rdma::broken() സ്ഥിരം {
    തിരികെ pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
സ്ഥിതം തർക്കം tcp_send_byte(പൂർണ്ണസംഖ്യ fd, uint8_t b) {
    ssize_t n;
    ചെയ്യുക { n = ::send(fd, &b, വലുപ്പം(b), 0); } ആകുമ്പോൾ (n < 0 && errno == EINTR);
    തിരികെ n == വലുപ്പം(b);
}

സ്ഥിതം തർക്കം tcp_recv_byte(പൂർണ്ണസംഖ്യ fd, uint8_t * b) {
    ssize_t n;
    ചെയ്യുക { n = ::recv(fd, b, വലുപ്പം(*b), 0); } ആകുമ്പോൾ (n < 0 && errno == EINTR);
    തിരികെ n == (ssize_t)വലുപ്പം(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
സ്ഥിതം പൂർണ്ണസംഖ്യ rdma_match_gid(ഘടന ibv_context * ctx, uint8_t port, പൂർണ്ണസംഖ്യ gid_tbl_len,
                          സ്ഥിരം uint8_t * target, സംഘം ibv_gid * out) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gid_tbl_len; i++) {
        സംഘം ibv_gid g;
        എങ്കിൽ (ibv_query_gid(ctx, port, i, &g) != 0) തുടരുക;
        എങ്കിൽ (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) തുടരുക;
        എങ്കിൽ (out) *out = g;
        തിരികെ i;
    }
    തിരികെ -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
സ്ഥിതം uint8_t rdma_first_active_port(ഘടന ibv_context * ctx, ഘടന ibv_port_attr * out) {
    ഘടന ibv_device_attr da;
    എങ്കിൽ (ibv_query_device(ctx, &da) != 0) തിരികെ 0;
    വേണ്ടി (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        ഘടന ibv_port_attr pa;
        എങ്കിൽ (ibv_query_port(ctx, p, &pa) != 0) തുടരുക;
        എങ്കിൽ (pa.state == IBV_PORT_ACTIVE) { എങ്കിൽ (out) *out = pa; തിരികെ p; }
    }
    തിരികെ 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
സ്ഥിതം തർക്കം rdma_library_present() {
    സ്ഥിതം സ്ഥിരം തർക്കം present = [] {
        ശൂന്യം * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        എങ്കിൽ (handle == nullptr) {
            തിരികെ അസത്യം;
        }
        dlclose(handle);
        തിരികെ സത്യം;
    }();
    തിരികെ present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(പൂർണ്ണസംഖ്യ fd, സ്ഥിരം uint8_t * target_gid, uint8_t * caps) {
    എങ്കിൽ (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        തിരികെ nullptr;
    }
    പൂർണ്ണസംഖ്യ ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    എങ്കിൽ (!devs) തിരികെ nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    ഘടന ibv_port_attr pa = {};
    സംഘം ibv_gid gid = {};
    പൂർണ്ണസംഖ്യ gid_idx = -1;
    std::string matched;
    വേണ്ടി (പൂർണ്ണസംഖ്യ d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        എങ്കിൽ (!c) തുടരുക;
        ഘടന ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        പൂർണ്ണസംഖ്യ gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        എങ്കിൽ (gi < 0) { ibv_close_device(c); തുടരുക; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        സ്ഥിരം അക്ഷരം * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        നിർത്തുക;
    }
    ibv_free_device_list(devs);
    എങ്കിൽ (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        തിരികെ nullptr;
    }

    std::unique_ptr<impl> c(പുതിയ impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    എങ്കിൽ (!c->pd) തിരികെ nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    എങ്കിൽ (!c->cq) തിരികെ nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    എങ്കിൽ (!c->qp) തിരികെ nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_അക്കം = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        എങ്കിൽ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            തിരികെ nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    എങ്കിൽ (page <= 0) page = 4096;
    സ്ഥിരം size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    എങ്കിൽ (posix_memalign((ശൂന്യം **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    എങ്കിൽ (posix_memalign((ശൂന്യം **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    എങ്കിൽ (!c->send_mem || !c->recv_mem) തിരികെ nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    സ്ഥിരം പൂർണ്ണസംഖ്യ mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    എങ്കിൽ (!c->send_mr || !c->recv_mr) തിരികെ nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_അക്കം;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, വലുപ്പം(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    തിരികെ std::unique_ptr<apple_rdma>(പുതിയ apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
തർക്കം apple_rdma::activate(സ്ഥിരം uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, വലുപ്പം(rc));

    തർക്കം ok = സത്യം;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_അക്കം = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_അക്കം      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        എങ്കിൽ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = അസത്യം;
        }
    }
    എങ്കിൽ (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        എങ്കിൽ (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = അസത്യം;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; ok && i < RDMA_NBUF; i++) {
        എങ്കിൽ (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = അസത്യം;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    എങ്കിൽ (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ok || peer_ready != RDMA_SYNC_READY) {
        തിരികെ അസത്യം;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    തിരികെ സത്യം;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
പൂർണ്ണസംഖ്യ apple_rdma::impl::progress() {
    ഘടന ibv_wc wc[RDMA_NBUF * 2];
    പൂർണ്ണസംഖ്യ n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    എങ്കിൽ (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = സത്യം; തിരികെ -1; }
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        തർക്കം is_recv = (id & RDMA_RECV_WR) != 0;
        എങ്കിൽ (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = സത്യം;
            തിരികെ -1;
        }
        എങ്കിൽ (is_recv) {
            പൂർണ്ണസംഖ്യ b = (പൂർണ്ണസംഖ്യ)(id & RDMA_WR_IDX_MASK);
            സ്ഥിരം rdma_seg_hdr * h = (സ്ഥിരം rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            എങ്കിൽ (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = സത്യം; തിരികെ -1; }
            എങ്കിൽ (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = സത്യം; തിരികെ -1; }
            പൂർണ്ണസംഖ്യ slot = (inq_head + inq_എണ്ണം) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_എണ്ണം++;
        } അല്ലെങ്കിൽ {
            send_busy[(പൂർണ്ണസംഖ്യ)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    തിരികെ n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
തർക്കം apple_rdma::impl::acquire_pending() {
    എങ്കിൽ (pend_buf >= 0) തിരികെ സത്യം;
    വേണ്ടി (;;) {
        എങ്കിൽ (broken) തിരികെ അസത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < RDMA_NBUF; k++) എങ്കിൽ (!send_busy[k]) { pend_buf = k; pend_len = 0; തിരികെ സത്യം; }
        എങ്കിൽ (progress() < 0) തിരികെ അസത്യം;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
തർക്കം apple_rdma::impl::post_pending() {
    എങ്കിൽ (pend_buf < 0) തിരികെ സത്യം;
    പൂർണ്ണസംഖ്യ i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    എങ്കിൽ (!post_send(i, RDMA_STRIDE)) { broken = സത്യം; തിരികെ അസത്യം; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    തിരികെ സത്യം;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
തർക്കം apple_rdma::send(സ്ഥിരം ശൂന്യം * data, size_t size) {
    impl * c = pimpl.get();
    സ്ഥിരം uint8_t * p = (സ്ഥിരം uint8_t *)data;
    ആകുമ്പോൾ (size > 0) {
        എങ്കിൽ (c->broken) തിരികെ അസത്യം;
        എങ്കിൽ (!c->acquire_pending()) തിരികെ അസത്യം;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + വലുപ്പം(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        എങ്കിൽ (c->pend_len == RDMA_PAYLOAD) { എങ്കിൽ (!c->post_pending()) തിരികെ അസത്യം; }
    }
    തിരികെ സത്യം;
}

തർക്കം apple_rdma::recv(ശൂന്യം * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    എങ്കിൽ (!c->post_pending()) തിരികെ അസത്യം;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    ആകുമ്പോൾ (size > 0) {
        എങ്കിൽ (c->inq_എണ്ണം == 0) {
            എങ്കിൽ (c->broken) തിരികെ അസത്യം;
            പൂർണ്ണസംഖ്യ n = c->progress();
            എങ്കിൽ (n < 0) തിരികെ അസത്യം;
            എങ്കിൽ (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                എങ്കിൽ ((++idle & 0xFFFFF) == 0) {
                    ഘടന pollfd pfd = { c->fd, POLLIN, 0 };
                    എങ്കിൽ (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        തിരികെ അസത്യം;
                    }
                }
            } അല്ലെങ്കിൽ {
                idle = 0;
            }
            തുടരുക;
        }
        idle = 0;
        പൂർണ്ണസംഖ്യ slot = c->inq_head;
        പൂർണ്ണസംഖ്യ b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + വലുപ്പം(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        എങ്കിൽ (c->inq[slot].off == c->inq[slot].len) {
            എങ്കിൽ (!c->post_recv(b)) { c->broken = സത്യം; തിരികെ അസത്യം; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_എണ്ണം--;
        }
    }
    തിരികെ സത്യം;
}

തർക്കം apple_rdma::flush() {
    തിരികെ pimpl->post_pending();
}
