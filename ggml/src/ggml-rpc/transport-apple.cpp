#અન્તર્ભાવ "transport-apple.h"
#અન્તર્ભાવ "transport.h"
#અન્તર્ભાવ "log.h"

#અન્તર્ભાવ <infiniband/verbs.h>

#અન્તર્ભાવ <cerrno>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <dlfcn.h>
#અન્તર્ભાવ <poll.h>
#અન્તર્ભાવ <sys/socket.h>
#અન્તર્ભાવ <unistd.h>

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
સ્થિર constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
સ્થિર constexpr પૂર્ણાંક      RDMA_NBUF        = 16;          // ring depth (frames per direction)
સ્થિર constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
સ્થિર constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
સ્થિર constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
સ્થિર constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
સ્થિર constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
સ્થિર constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
સ્થિર constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
સ્થિર constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

સંરચના rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
સ્થિર constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - કદ(rdma_seg_hdr);

સંરચના apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(કદ(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

સંરચના apple_rdma::impl {
    પૂર્ણાંક fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    સંરચના ibv_context * ctx = nullptr;
    સંરચના ibv_pd * pd = nullptr;
    સંરચના ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    સંરચના ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    સંરચના ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    સંરચના ibv_mr * recv_mr  = nullptr;

    પૂર્ણાંક      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    સંરચના { પૂર્ણાંક buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    પૂર્ણાંક      inq_head = 0;
    પૂર્ણાંક      inq_ગણતરી = 0;
    પૂર્ણાંક      pend_buf = -1;
    uint32_t pend_len = 0;
    તાર્કિક     broken = ખોટું;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    પૂર્ણાંક          gid_idx = 0;
    ગણના ibv_mtu path_mtu = IBV_MTU_1024;

    પૂર્ણાંક  progress();
    તાર્કિક acquire_pending();
    તાર્કિક post_pending();

    તાર્કિક post_recv(પૂર્ણાંક i) {
        સંરચના ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        સંરચના ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.અંક_sge = 1;
        પાછા ibv_post_recv(qp, &wr, &bad) == 0;
    }

    તાર્કિક post_send(પૂર્ણાંક i, size_t len) {
        સંરચના ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        સંરચના ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.અંક_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        પાછા ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = સાચું;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        જો (qp) ibv_destroy_qp(qp);
        જો (send_mr) ibv_dereg_mr(send_mr);
        જો (recv_mr) ibv_dereg_mr(recv_mr);
        મુક્ત(send_mem);
        મુક્ત(recv_mem);
        જો (cq)  ibv_destroy_cq(cq);
        જો (pd)  ibv_dealloc_pd(pd);
        જો (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = મૂળભૂત;

તાર્કિક apple_rdma::broken() અચલ {
    પાછા pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
સ્થિર તાર્કિક tcp_send_byte(પૂર્ણાંક fd, uint8_t b) {
    ssize_t n;
    કરો { n = ::send(fd, &b, કદ(b), 0); } જ્યારે (n < 0 && errno == EINTR);
    પાછા n == કદ(b);
}

સ્થિર તાર્કિક tcp_recv_byte(પૂર્ણાંક fd, uint8_t * b) {
    ssize_t n;
    કરો { n = ::recv(fd, b, કદ(*b), 0); } જ્યારે (n < 0 && errno == EINTR);
    પાછા n == (ssize_t)કદ(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
સ્થિર પૂર્ણાંક rdma_match_gid(સંરચના ibv_context * ctx, uint8_t port, પૂર્ણાંક gid_tbl_len,
                          અચલ uint8_t * target, સંઘ ibv_gid * out) {
    માટે (પૂર્ણાંક i = 0; i < gid_tbl_len; i++) {
        સંઘ ibv_gid g;
        જો (ibv_query_gid(ctx, port, i, &g) != 0) ચાલુ;
        જો (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) ચાલુ;
        જો (out) *out = g;
        પાછા i;
    }
    પાછા -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
સ્થિર uint8_t rdma_first_active_port(સંરચના ibv_context * ctx, સંરચના ibv_port_attr * out) {
    સંરચના ibv_device_attr da;
    જો (ibv_query_device(ctx, &da) != 0) પાછા 0;
    માટે (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        સંરચના ibv_port_attr pa;
        જો (ibv_query_port(ctx, p, &pa) != 0) ચાલુ;
        જો (pa.state == IBV_PORT_ACTIVE) { જો (out) *out = pa; પાછા p; }
    }
    પાછા 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
સ્થિર તાર્કિક rdma_library_present() {
    સ્થિર અચલ તાર્કિક present = [] {
        શૂન્ય * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        જો (handle == nullptr) {
            પાછા ખોટું;
        }
        dlclose(handle);
        પાછા સાચું;
    }();
    પાછા present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(પૂર્ણાંક fd, અચલ uint8_t * target_gid, uint8_t * caps) {
    જો (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        પાછા nullptr;
    }
    પૂર્ણાંક ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    જો (!devs) પાછા nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    સંરચના ibv_port_attr pa = {};
    સંઘ ibv_gid gid = {};
    પૂર્ણાંક gid_idx = -1;
    std::string matched;
    માટે (પૂર્ણાંક d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        જો (!c) ચાલુ;
        સંરચના ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        પૂર્ણાંક gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        જો (gi < 0) { ibv_close_device(c); ચાલુ; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        અચલ અક્ષર * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        તોડો;
    }
    ibv_free_device_list(devs);
    જો (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        પાછા nullptr;
    }

    std::unique_ptr<impl> c(નવું impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    જો (!c->pd) પાછા nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    જો (!c->cq) પાછા nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    જો (!c->qp) પાછા nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_અંક = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        જો (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            પાછા nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    જો (page <= 0) page = 4096;
    અચલ size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    જો (posix_memalign((શૂન્ય **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    જો (posix_memalign((શૂન્ય **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    જો (!c->send_mem || !c->recv_mem) પાછા nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    અચલ પૂર્ણાંક mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    જો (!c->send_mr || !c->recv_mr) પાછા nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_અંક;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, કદ(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    પાછા std::unique_ptr<apple_rdma>(નવું apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
તાર્કિક apple_rdma::activate(અચલ uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, કદ(rc));

    તાર્કિક ok = સાચું;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_અંક = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_અંક      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        જો (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = ખોટું;
        }
    }
    જો (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        જો (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = ખોટું;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    માટે (પૂર્ણાંક i = 0; ok && i < RDMA_NBUF; i++) {
        જો (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = ખોટું;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    જો (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        પાછા ખોટું;
    }
    જો (!ok || peer_ready != RDMA_SYNC_READY) {
        પાછા ખોટું;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    પાછા સાચું;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
પૂર્ણાંક apple_rdma::impl::progress() {
    સંરચના ibv_wc wc[RDMA_NBUF * 2];
    પૂર્ણાંક n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    જો (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = સાચું; પાછા -1; }
    માટે (પૂર્ણાંક j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        તાર્કિક is_recv = (id & RDMA_RECV_WR) != 0;
        જો (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = સાચું;
            પાછા -1;
        }
        જો (is_recv) {
            પૂર્ણાંક b = (પૂર્ણાંક)(id & RDMA_WR_IDX_MASK);
            અચલ rdma_seg_hdr * h = (અચલ rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            જો (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = સાચું; પાછા -1; }
            જો (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = સાચું; પાછા -1; }
            પૂર્ણાંક slot = (inq_head + inq_ગણતરી) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_ગણતરી++;
        } નહીંતર {
            send_busy[(પૂર્ણાંક)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    પાછા n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
તાર્કિક apple_rdma::impl::acquire_pending() {
    જો (pend_buf >= 0) પાછા સાચું;
    માટે (;;) {
        જો (broken) પાછા ખોટું;
        માટે (પૂર્ણાંક k = 0; k < RDMA_NBUF; k++) જો (!send_busy[k]) { pend_buf = k; pend_len = 0; પાછા સાચું; }
        જો (progress() < 0) પાછા ખોટું;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
તાર્કિક apple_rdma::impl::post_pending() {
    જો (pend_buf < 0) પાછા સાચું;
    પૂર્ણાંક i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    જો (!post_send(i, RDMA_STRIDE)) { broken = સાચું; પાછા ખોટું; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    પાછા સાચું;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
તાર્કિક apple_rdma::send(અચલ શૂન્ય * data, size_t size) {
    impl * c = pimpl.get();
    અચલ uint8_t * p = (અચલ uint8_t *)data;
    જ્યારે (size > 0) {
        જો (c->broken) પાછા ખોટું;
        જો (!c->acquire_pending()) પાછા ખોટું;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + કદ(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        જો (c->pend_len == RDMA_PAYLOAD) { જો (!c->post_pending()) પાછા ખોટું; }
    }
    પાછા સાચું;
}

તાર્કિક apple_rdma::recv(શૂન્ય * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    જો (!c->post_pending()) પાછા ખોટું;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    જ્યારે (size > 0) {
        જો (c->inq_ગણતરી == 0) {
            જો (c->broken) પાછા ખોટું;
            પૂર્ણાંક n = c->progress();
            જો (n < 0) પાછા ખોટું;
            જો (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                જો ((++idle & 0xFFFFF) == 0) {
                    સંરચના pollfd pfd = { c->fd, POLLIN, 0 };
                    જો (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        પાછા ખોટું;
                    }
                }
            } નહીંતર {
                idle = 0;
            }
            ચાલુ;
        }
        idle = 0;
        પૂર્ણાંક slot = c->inq_head;
        પૂર્ણાંક b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + કદ(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        જો (c->inq[slot].off == c->inq[slot].len) {
            જો (!c->post_recv(b)) { c->broken = સાચું; પાછા ખોટું; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_ગણતરી--;
        }
    }
    પાછા સાચું;
}

તાર્કિક apple_rdma::flush() {
    પાછા pimpl->post_pending();
}
