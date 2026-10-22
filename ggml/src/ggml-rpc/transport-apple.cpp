#ଅନ୍ତର୍ଭାଵ "transport-apple.h"
#ଅନ୍ତର୍ଭାଵ "transport.h"
#ଅନ୍ତର୍ଭାଵ "log.h"

#ଅନ୍ତର୍ଭାଵ <infiniband/verbs.h>

#ଅନ୍ତର୍ଭାଵ <cerrno>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <dlfcn.h>
#ଅନ୍ତର୍ଭାଵ <poll.h>
#ଅନ୍ତର୍ଭାଵ <sys/socket.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

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
ସ୍ଥିର constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା      RDMA_NBUF        = 16;          // ring depth (frames per direction)
ସ୍ଥିର constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
ସ୍ଥିର constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
ସ୍ଥିର constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
ସ୍ଥିର constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
ସ୍ଥିର constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
ସ୍ଥିର constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
ସ୍ଥିର constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
ସ୍ଥିର constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

ସଂରଚନା rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
ସ୍ଥିର constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - ଆକାର(rdma_seg_hdr);

ସଂରଚନା apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(ଆକାର(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

ସଂରଚନା apple_rdma::impl {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    ସଂରଚନା ibv_context * ctx = nullptr;
    ସଂରଚନା ibv_pd * pd = nullptr;
    ସଂରଚନା ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    ସଂରଚନା ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    ସଂରଚନା ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    ସଂରଚନା ibv_mr * recv_mr  = nullptr;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    ସଂରଚନା { ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      inq_head = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      inq_ଗଣନା = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      pend_buf = -1;
    uint32_t pend_len = 0;
    ତାର୍କିକ     broken = ମିଥ୍ୟା;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା          gid_idx = 0;
    ଗଣନା ibv_mtu path_mtu = IBV_MTU_1024;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  progress();
    ତାର୍କିକ acquire_pending();
    ତାର୍କିକ post_pending();

    ତାର୍କିକ post_recv(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
        ସଂରଚନା ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        ସଂରଚନା ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.ଅଙ୍କ_sge = 1;
        ଫେରନ୍ତୁ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ତାର୍କିକ post_send(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, size_t len) {
        ସଂରଚନା ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        ସଂରଚନା ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ଅଙ୍କ_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ଫେରନ୍ତୁ ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = ସତ୍ୟ;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        ଯଦି (qp) ibv_destroy_qp(qp);
        ଯଦି (send_mr) ibv_dereg_mr(send_mr);
        ଯଦି (recv_mr) ibv_dereg_mr(recv_mr);
        ମୁକ୍ତ(send_mem);
        ମୁକ୍ତ(recv_mem);
        ଯଦି (cq)  ibv_destroy_cq(cq);
        ଯଦି (pd)  ibv_dealloc_pd(pd);
        ଯଦି (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

ତାର୍କିକ apple_rdma::broken() ଅଚଳ {
    ଫେରନ୍ତୁ pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ସ୍ଥିର ତାର୍କିକ tcp_send_byte(ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, uint8_t b) {
    ssize_t n;
    କର { n = ::send(fd, &b, ଆକାର(b), 0); } ଯେତେବେଳେ (n < 0 && errno == EINTR);
    ଫେରନ୍ତୁ n == ଆକାର(b);
}

ସ୍ଥିର ତାର୍କିକ tcp_recv_byte(ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, uint8_t * b) {
    ssize_t n;
    କର { n = ::recv(fd, b, ଆକାର(*b), 0); } ଯେତେବେଳେ (n < 0 && errno == EINTR);
    ଫେରନ୍ତୁ n == (ssize_t)ଆକାର(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା rdma_match_gid(ସଂରଚନା ibv_context * ctx, uint8_t port, ପୂର୍ଣ୍ଣସଂଖ୍ୟା gid_tbl_len,
                          ଅଚଳ uint8_t * target, ସଂଘ ibv_gid * out) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gid_tbl_len; i++) {
        ସଂଘ ibv_gid g;
        ଯଦି (ibv_query_gid(ctx, port, i, &g) != 0) ଜାରି;
        ଯଦି (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) ଜାରି;
        ଯଦି (out) *out = g;
        ଫେରନ୍ତୁ i;
    }
    ଫେରନ୍ତୁ -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ସ୍ଥିର uint8_t rdma_first_active_port(ସଂରଚନା ibv_context * ctx, ସଂରଚନା ibv_port_attr * out) {
    ସଂରଚନା ibv_device_attr da;
    ଯଦି (ibv_query_device(ctx, &da) != 0) ଫେରନ୍ତୁ 0;
    ପାଇଁ (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        ସଂରଚନା ibv_port_attr pa;
        ଯଦି (ibv_query_port(ctx, p, &pa) != 0) ଜାରି;
        ଯଦି (pa.state == IBV_PORT_ACTIVE) { ଯଦି (out) *out = pa; ଫେରନ୍ତୁ p; }
    }
    ଫେରନ୍ତୁ 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ସ୍ଥିର ତାର୍କିକ rdma_library_present() {
    ସ୍ଥିର ଅଚଳ ତାର୍କିକ present = [] {
        ଶୂନ୍ୟ * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        ଯଦି (handle == nullptr) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        dlclose(handle);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }();
    ଫେରନ୍ତୁ present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, ଅଚଳ uint8_t * target_gid, uint8_t * caps) {
    ଯଦି (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        ଫେରନ୍ତୁ nullptr;
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    ଯଦି (!devs) ଫେରନ୍ତୁ nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    ସଂରଚନା ibv_port_attr pa = {};
    ସଂଘ ibv_gid gid = {};
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା gid_idx = -1;
    std::string matched;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        ଯଦି (!c) ଜାରି;
        ସଂରଚନା ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        ଯଦି (gi < 0) { ibv_close_device(c); ଜାରି; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        ଅଚଳ ଅକ୍ଷର * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        ଭାଙ୍ଗ;
    }
    ibv_free_device_list(devs);
    ଯଦି (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        ଫେରନ୍ତୁ nullptr;
    }

    std::unique_ptr<impl> c(ନୂତନ impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    ଯଦି (!c->pd) ଫେରନ୍ତୁ nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    ଯଦି (!c->cq) ଫେରନ୍ତୁ nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    ଯଦି (!c->qp) ଫେରନ୍ତୁ nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_ଅଙ୍କ = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ଯଦି (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ଫେରନ୍ତୁ nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    ଯଦି (page <= 0) page = 4096;
    ଅଚଳ size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    ଯଦି (posix_memalign((ଶୂନ୍ୟ **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    ଯଦି (posix_memalign((ଶୂନ୍ୟ **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    ଯଦି (!c->send_mem || !c->recv_mem) ଫେରନ୍ତୁ nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    ଯଦି (!c->send_mr || !c->recv_mr) ଫେରନ୍ତୁ nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_ଅଙ୍କ;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, ଆକାର(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    ଫେରନ୍ତୁ std::unique_ptr<apple_rdma>(ନୂତନ apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
ତାର୍କିକ apple_rdma::activate(ଅଚଳ uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, ଆକାର(rc));

    ତାର୍କିକ ok = ସତ୍ୟ;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_ଅଙ୍କ = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_ଅଙ୍କ      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        ଯଦି (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = ମିଥ୍ୟା;
        }
    }
    ଯଦି (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        ଯଦି (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = ମିଥ୍ୟା;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; ok && i < RDMA_NBUF; i++) {
        ଯଦି (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = ମିଥ୍ୟା;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    ଯଦି (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ok || peer_ready != RDMA_SYNC_READY) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
ପୂର୍ଣ୍ଣସଂଖ୍ୟା apple_rdma::impl::progress() {
    ସଂରଚନା ibv_wc wc[RDMA_NBUF * 2];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    ଯଦି (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = ସତ୍ୟ; ଫେରନ୍ତୁ -1; }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        ତାର୍କିକ is_recv = (id & RDMA_RECV_WR) != 0;
        ଯଦି (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = ସତ୍ୟ;
            ଫେରନ୍ତୁ -1;
        }
        ଯଦି (is_recv) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(id & RDMA_WR_IDX_MASK);
            ଅଚଳ rdma_seg_hdr * h = (ଅଚଳ rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            ଯଦି (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = ସତ୍ୟ; ଫେରନ୍ତୁ -1; }
            ଯଦି (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = ସତ୍ୟ; ଫେରନ୍ତୁ -1; }
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = (inq_head + inq_ଗଣନା) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_ଗଣନା++;
        } ଅନ୍ୟଥା {
            send_busy[(ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    ଫେରନ୍ତୁ n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
ତାର୍କିକ apple_rdma::impl::acquire_pending() {
    ଯଦି (pend_buf >= 0) ଫେରନ୍ତୁ ସତ୍ୟ;
    ପାଇଁ (;;) {
        ଯଦି (broken) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < RDMA_NBUF; k++) ଯଦି (!send_busy[k]) { pend_buf = k; pend_len = 0; ଫେରନ୍ତୁ ସତ୍ୟ; }
        ଯଦି (progress() < 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
ତାର୍କିକ apple_rdma::impl::post_pending() {
    ଯଦି (pend_buf < 0) ଫେରନ୍ତୁ ସତ୍ୟ;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    ଯଦି (!post_send(i, RDMA_STRIDE)) { broken = ସତ୍ୟ; ଫେରନ୍ତୁ ମିଥ୍ୟା; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
ତାର୍କିକ apple_rdma::send(ଅଚଳ ଶୂନ୍ୟ * data, size_t size) {
    impl * c = pimpl.get();
    ଅଚଳ uint8_t * p = (ଅଚଳ uint8_t *)data;
    ଯେତେବେଳେ (size > 0) {
        ଯଦି (c->broken) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (!c->acquire_pending()) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + ଆକାର(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        ଯଦି (c->pend_len == RDMA_PAYLOAD) { ଯଦି (!c->post_pending()) ଫେରନ୍ତୁ ମିଥ୍ୟା; }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ apple_rdma::recv(ଶୂନ୍ୟ * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    ଯଦି (!c->post_pending()) ଫେରନ୍ତୁ ମିଥ୍ୟା;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    ଯେତେବେଳେ (size > 0) {
        ଯଦି (c->inq_ଗଣନା == 0) {
            ଯଦି (c->broken) ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = c->progress();
            ଯଦି (n < 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ଯଦି (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                ଯଦି ((++idle & 0xFFFFF) == 0) {
                    ସଂରଚନା pollfd pfd = { c->fd, POLLIN, 0 };
                    ଯଦି (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }
            } ଅନ୍ୟଥା {
                idle = 0;
            }
            ଜାରି;
        }
        idle = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = c->inq_head;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + ଆକାର(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        ଯଦି (c->inq[slot].off == c->inq[slot].len) {
            ଯଦି (!c->post_recv(b)) { c->broken = ସତ୍ୟ; ଫେରନ୍ତୁ ମିଥ୍ୟା; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_ଗଣନା--;
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ apple_rdma::flush() {
    ଫେରନ୍ତୁ pimpl->post_pending();
}
