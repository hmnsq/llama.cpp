#அந்தர்பாவ "transport-apple.h"
#அந்தர்பாவ "transport.h"
#அந்தர்பாவ "log.h"

#அந்தர்பாவ <infiniband/verbs.h>

#அந்தர்பாவ <cerrno>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <string>
#அந்தர்பாவ <dlfcn.h>
#அந்தர்பாவ <poll.h>
#அந்தர்பாவ <sys/socket.h>
#அந்தர்பாவ <unistd.h>

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
நிலையான constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
நிலையான constexpr முழு      RDMA_NBUF        = 16;          // ring depth (frames per direction)
நிலையான constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
நிலையான constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
நிலையான constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
நிலையான constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
நிலையான constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
நிலையான constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
நிலையான constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
நிலையான constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

கட்டமைப்பு rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
நிலையான constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - அளவு(rdma_seg_hdr);

கட்டமைப்பு apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(அளவு(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

கட்டமைப்பு apple_rdma::impl {
    முழு fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    கட்டமைப்பு ibv_context * ctx = nullptr;
    கட்டமைப்பு ibv_pd * pd = nullptr;
    கட்டமைப்பு ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    கட்டமைப்பு ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    கட்டமைப்பு ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    கட்டமைப்பு ibv_mr * recv_mr  = nullptr;

    முழு      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    கட்டமைப்பு { முழு buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    முழு      inq_head = 0;
    முழு      inq_மொத்தம் = 0;
    முழு      pend_buf = -1;
    uint32_t pend_len = 0;
    தருக்க     broken = பொய்;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    முழு          gid_idx = 0;
    எண்ணிக்கை ibv_mtu path_mtu = IBV_MTU_1024;

    முழு  progress();
    தருக்க acquire_pending();
    தருக்க post_pending();

    தருக்க post_recv(முழு i) {
        கட்டமைப்பு ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        கட்டமைப்பு ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.இலக்கம்_sge = 1;
        திரும்பு ibv_post_recv(qp, &wr, &bad) == 0;
    }

    தருக்க post_send(முழு i, size_t len) {
        கட்டமைப்பு ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        கட்டமைப்பு ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.இலக்கம்_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        திரும்பு ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = உண்மை;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        எனில் (qp) ibv_destroy_qp(qp);
        எனில் (send_mr) ibv_dereg_mr(send_mr);
        எனில் (recv_mr) ibv_dereg_mr(recv_mr);
        விடுவி(send_mem);
        விடுவி(recv_mem);
        எனில் (cq)  ibv_destroy_cq(cq);
        எனில் (pd)  ibv_dealloc_pd(pd);
        எனில் (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = இயல்பு;

தருக்க apple_rdma::broken() மாறிலி {
    திரும்பு pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
நிலையான தருக்க tcp_send_byte(முழு fd, uint8_t b) {
    ssize_t n;
    செய் { n = ::send(fd, &b, அளவு(b), 0); } போது (n < 0 && errno == EINTR);
    திரும்பு n == அளவு(b);
}

நிலையான தருக்க tcp_recv_byte(முழு fd, uint8_t * b) {
    ssize_t n;
    செய் { n = ::recv(fd, b, அளவு(*b), 0); } போது (n < 0 && errno == EINTR);
    திரும்பு n == (ssize_t)அளவு(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
நிலையான முழு rdma_match_gid(கட்டமைப்பு ibv_context * ctx, uint8_t port, முழு gid_tbl_len,
                          மாறிலி uint8_t * target, ஒன்றிப்பு ibv_gid * out) {
    க்கு (முழு i = 0; i < gid_tbl_len; i++) {
        ஒன்றிப்பு ibv_gid g;
        எனில் (ibv_query_gid(ctx, port, i, &g) != 0) தொடர்;
        எனில் (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) தொடர்;
        எனில் (out) *out = g;
        திரும்பு i;
    }
    திரும்பு -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
நிலையான uint8_t rdma_first_active_port(கட்டமைப்பு ibv_context * ctx, கட்டமைப்பு ibv_port_attr * out) {
    கட்டமைப்பு ibv_device_attr da;
    எனில் (ibv_query_device(ctx, &da) != 0) திரும்பு 0;
    க்கு (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        கட்டமைப்பு ibv_port_attr pa;
        எனில் (ibv_query_port(ctx, p, &pa) != 0) தொடர்;
        எனில் (pa.state == IBV_PORT_ACTIVE) { எனில் (out) *out = pa; திரும்பு p; }
    }
    திரும்பு 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
நிலையான தருக்க rdma_library_present() {
    நிலையான மாறிலி தருக்க present = [] {
        வெறுமை * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        எனில் (handle == nullptr) {
            திரும்பு பொய்;
        }
        dlclose(handle);
        திரும்பு உண்மை;
    }();
    திரும்பு present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(முழு fd, மாறிலி uint8_t * target_gid, uint8_t * caps) {
    எனில் (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        திரும்பு nullptr;
    }
    முழு ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    எனில் (!devs) திரும்பு nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    கட்டமைப்பு ibv_port_attr pa = {};
    ஒன்றிப்பு ibv_gid gid = {};
    முழு gid_idx = -1;
    std::string matched;
    க்கு (முழு d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        எனில் (!c) தொடர்;
        கட்டமைப்பு ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        முழு gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        எனில் (gi < 0) { ibv_close_device(c); தொடர்; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        மாறிலி எழுத்து * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        நிறுத்து;
    }
    ibv_free_device_list(devs);
    எனில் (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        திரும்பு nullptr;
    }

    std::unique_ptr<impl> c(புதிய impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    எனில் (!c->pd) திரும்பு nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    எனில் (!c->cq) திரும்பு nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    எனில் (!c->qp) திரும்பு nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_இலக்கம் = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        எனில் (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            திரும்பு nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    எனில் (page <= 0) page = 4096;
    மாறிலி size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    எனில் (posix_memalign((வெறுமை **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    எனில் (posix_memalign((வெறுமை **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    எனில் (!c->send_mem || !c->recv_mem) திரும்பு nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    மாறிலி முழு mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    எனில் (!c->send_mr || !c->recv_mr) திரும்பு nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_இலக்கம்;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, அளவு(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    திரும்பு std::unique_ptr<apple_rdma>(புதிய apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
தருக்க apple_rdma::activate(மாறிலி uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, அளவு(rc));

    தருக்க ok = உண்மை;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_இலக்கம் = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_இலக்கம்      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        எனில் (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = பொய்;
        }
    }
    எனில் (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        எனில் (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = பொய்;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    க்கு (முழு i = 0; ok && i < RDMA_NBUF; i++) {
        எனில் (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = பொய்;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    எனில் (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        திரும்பு பொய்;
    }
    எனில் (!ok || peer_ready != RDMA_SYNC_READY) {
        திரும்பு பொய்;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    திரும்பு உண்மை;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
முழு apple_rdma::impl::progress() {
    கட்டமைப்பு ibv_wc wc[RDMA_NBUF * 2];
    முழு n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    எனில் (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = உண்மை; திரும்பு -1; }
    க்கு (முழு j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        தருக்க is_recv = (id & RDMA_RECV_WR) != 0;
        எனில் (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = உண்மை;
            திரும்பு -1;
        }
        எனில் (is_recv) {
            முழு b = (முழு)(id & RDMA_WR_IDX_MASK);
            மாறிலி rdma_seg_hdr * h = (மாறிலி rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            எனில் (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = உண்மை; திரும்பு -1; }
            எனில் (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = உண்மை; திரும்பு -1; }
            முழு slot = (inq_head + inq_மொத்தம்) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_மொத்தம்++;
        } இல்லையெனில் {
            send_busy[(முழு)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    திரும்பு n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
தருக்க apple_rdma::impl::acquire_pending() {
    எனில் (pend_buf >= 0) திரும்பு உண்மை;
    க்கு (;;) {
        எனில் (broken) திரும்பு பொய்;
        க்கு (முழு k = 0; k < RDMA_NBUF; k++) எனில் (!send_busy[k]) { pend_buf = k; pend_len = 0; திரும்பு உண்மை; }
        எனில் (progress() < 0) திரும்பு பொய்;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
தருக்க apple_rdma::impl::post_pending() {
    எனில் (pend_buf < 0) திரும்பு உண்மை;
    முழு i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    எனில் (!post_send(i, RDMA_STRIDE)) { broken = உண்மை; திரும்பு பொய்; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    திரும்பு உண்மை;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
தருக்க apple_rdma::send(மாறிலி வெறுமை * data, size_t size) {
    impl * c = pimpl.get();
    மாறிலி uint8_t * p = (மாறிலி uint8_t *)data;
    போது (size > 0) {
        எனில் (c->broken) திரும்பு பொய்;
        எனில் (!c->acquire_pending()) திரும்பு பொய்;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + அளவு(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        எனில் (c->pend_len == RDMA_PAYLOAD) { எனில் (!c->post_pending()) திரும்பு பொய்; }
    }
    திரும்பு உண்மை;
}

தருக்க apple_rdma::recv(வெறுமை * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    எனில் (!c->post_pending()) திரும்பு பொய்;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    போது (size > 0) {
        எனில் (c->inq_மொத்தம் == 0) {
            எனில் (c->broken) திரும்பு பொய்;
            முழு n = c->progress();
            எனில் (n < 0) திரும்பு பொய்;
            எனில் (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                எனில் ((++idle & 0xFFFFF) == 0) {
                    கட்டமைப்பு pollfd pfd = { c->fd, POLLIN, 0 };
                    எனில் (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        திரும்பு பொய்;
                    }
                }
            } இல்லையெனில் {
                idle = 0;
            }
            தொடர்;
        }
        idle = 0;
        முழு slot = c->inq_head;
        முழு b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + அளவு(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        எனில் (c->inq[slot].off == c->inq[slot].len) {
            எனில் (!c->post_recv(b)) { c->broken = உண்மை; திரும்பு பொய்; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_மொத்தம்--;
        }
    }
    திரும்பு உண்மை;
}

தருக்க apple_rdma::flush() {
    திரும்பு pimpl->post_pending();
}
