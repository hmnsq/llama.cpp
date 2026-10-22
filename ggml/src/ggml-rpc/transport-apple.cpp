#అన్తర్భావ "transport-apple.h"
#అన్తర్భావ "transport.h"
#అన్తర్భావ "log.h"

#అన్తర్భావ <infiniband/verbs.h>

#అన్తర్భావ <cerrno>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <string>
#అన్తర్భావ <dlfcn.h>
#అన్తర్భావ <poll.h>
#అన్తర్భావ <sys/socket.h>
#అన్తర్భావ <unistd.h>

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
స్థిర constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
స్థిర constexpr పూర్ణాంకం      RDMA_NBUF        = 16;          // ring depth (frames per direction)
స్థిర constexpr size_t   RDMA_చట్రం       = 4096;        // Thunderbolt frame (fixed on Apple)
స్థిర constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
స్థిర constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
స్థిర constexpr size_t   RDMA_GID_కొలత    = 16;

static_assert(RDMA_STRIDE % RDMA_చట్రం == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
స్థిర constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_చట్రం);
స్థిర constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
స్థిర constexpr uint64_t RDMA_WR_IDX_ముసుగు = 0xffff;      // buffer index in the low bits of wr_id
స్థిర constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

నిర్మాణం rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
స్థిర constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - పరిమాణం(rdma_seg_hdr);

నిర్మాణం apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_కొలత];
};

static_assert(పరిమాణం(apple_rdma_caps) == RPC_CONN_CAPS_కొలత, "apple_rdma_caps must match conn_caps size");

నిర్మాణం apple_rdma::impl {
    పూర్ణాంకం fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    నిర్మాణం ibv_context * ctx = శూన్యసూచిక;
    నిర్మాణం ibv_pd * pd = శూన్యసూచిక;
    నిర్మాణం ibv_cq * cq = శూన్యసూచిక;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    నిర్మాణం ibv_qp * qp = శూన్యసూచిక;

    uint8_t       * send_mem = శూన్యసూచిక;
    నిర్మాణం ibv_mr * send_mr  = శూన్యసూచిక;
    uint8_t       * recv_mem = శూన్యసూచిక;
    నిర్మాణం ibv_mr * recv_mr  = శూన్యసూచిక;

    పూర్ణాంకం      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    నిర్మాణం { పూర్ణాంకం buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    పూర్ణాంకం      inq_తల = 0;
    పూర్ణాంకం      inq_లెక్క = 0;
    పూర్ణాంకం      pend_buf = -1;
    uint32_t pend_len = 0;
    తార్కిక     broken = అబద్ధం;

    uint32_t     qpn = 0;
    uint8_t      పోర్టు = 0;
    పూర్ణాంకం          gid_idx = 0;
    గణన ibv_mtu మార్గం_mtu = IBV_MTU_1024;

    పూర్ణాంకం  పురోగతి();
    తార్కిక acquire_pending();
    తార్కిక post_pending();

    తార్కిక post_recv(పూర్ణాంకం i) {
        నిర్మాణం ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.నిడివి = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        నిర్మాణం ibv_recv_wr wr = {}, * bad = శూన్యసూచిక;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.అంకె_sge = 1;
        తిరిగి ibv_post_recv(qp, &wr, &bad) == 0;
    }

    తార్కిక post_send(పూర్ణాంకం i, size_t len) {
        నిర్మాణం ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.నిడివి = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        నిర్మాణం ibv_send_wr wr = {}, * bad = శూన్యసూచిక;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.అంకె_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        తిరిగి ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = నిజం;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        ఒకవేళ (qp) ibv_destroy_qp(qp);
        ఒకవేళ (send_mr) ibv_dereg_mr(send_mr);
        ఒకవేళ (recv_mr) ibv_dereg_mr(recv_mr);
        విడుదల(send_mem);
        విడుదల(recv_mem);
        ఒకవేళ (cq)  ibv_destroy_cq(cq);
        ఒకవేళ (pd)  ibv_dealloc_pd(pd);
        ఒకవేళ (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = అప్రమేయ;

తార్కిక apple_rdma::broken() స్థిరరాశి {
    తిరిగి pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
స్థిర తార్కిక tcp_send_byte(పూర్ణాంకం fd, uint8_t b) {
    ssize_t n;
    చేయి { n = ::send(fd, &b, పరిమాణం(b), 0); } అయితే (n < 0 && errno == EINTR);
    తిరిగి n == పరిమాణం(b);
}

స్థిర తార్కిక tcp_recv_byte(పూర్ణాంకం fd, uint8_t * b) {
    ssize_t n;
    చేయి { n = ::recv(fd, b, పరిమాణం(*b), 0); } అయితే (n < 0 && errno == EINTR);
    తిరిగి n == (ssize_t)పరిమాణం(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
స్థిర పూర్ణాంకం rdma_సరిపోలిక_gid(నిర్మాణం ibv_context * ctx, uint8_t పోర్టు, పూర్ణాంకం gid_tbl_len,
                          స్థిరరాశి uint8_t * లక్ష్యం, సంఘం ibv_gid * బహిర్గతం) {
    కొరకు (పూర్ణాంకం i = 0; i < gid_tbl_len; i++) {
        సంఘం ibv_gid g;
        ఒకవేళ (ibv_query_gid(ctx, పోర్టు, i, &g) != 0) కొనసాగు;
        ఒకవేళ (memcmp(g.raw, లక్ష్యం, RDMA_GID_కొలత) != 0) కొనసాగు;
        ఒకవేళ (బహిర్గతం) *బహిర్గతం = g;
        తిరిగి i;
    }
    తిరిగి -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
స్థిర uint8_t rdma_మొదటి_చురుకైన_పోర్టు(నిర్మాణం ibv_context * ctx, నిర్మాణం ibv_పోర్టు_attr * బహిర్గతం) {
    నిర్మాణం ibv_device_attr da;
    ఒకవేళ (ibv_query_device(ctx, &da) != 0) తిరిగి 0;
    కొరకు (uint8_t p = 1; p <= da.phys_పోర్టు_cnt; p++) {
        నిర్మాణం ibv_పోర్టు_attr pa;
        ఒకవేళ (ibv_query_పోర్టు(ctx, p, &pa) != 0) కొనసాగు;
        ఒకవేళ (pa.state == IBV_పోర్టు_చురుకైన) { ఒకవేళ (బహిర్గతం) *బహిర్గతం = pa; తిరిగి p; }
    }
    తిరిగి 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
స్థిర తార్కిక rdma_library_present() {
    స్థిర స్థిరరాశి తార్కిక present = [] {
        శూన్య * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        ఒకవేళ (handle == శూన్యసూచిక) {
            తిరిగి అబద్ధం;
        }
        dlclose(handle);
        తిరిగి నిజం;
    }();
    తిరిగి present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(పూర్ణాంకం fd, స్థిరరాశి uint8_t * లక్ష్యం_gid, uint8_t * caps) {
    ఒకవేళ (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        తిరిగి శూన్యసూచిక;
    }
    పూర్ణాంకం ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    ఒకవేళ (!devs) తిరిగి శూన్యసూచిక;

    ibv_context * ctx = శూన్యసూచిక;
    uint8_t పోర్టు = 0;
    నిర్మాణం ibv_పోర్టు_attr pa = {};
    సంఘం ibv_gid gid = {};
    పూర్ణాంకం gid_idx = -1;
    std::string matched;
    కొరకు (పూర్ణాంకం d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        ఒకవేళ (!c) కొనసాగు;
        నిర్మాణం ibv_పోర్టు_attr p = {};
        uint8_t pt = rdma_మొదటి_చురుకైన_పోర్టు(c, &p);
        పూర్ణాంకం gi = pt ? rdma_సరిపోలిక_gid(c, pt, p.gid_tbl_len, లక్ష్యం_gid, &gid) : -1;
        ఒకవేళ (gi < 0) { ibv_close_device(c); కొనసాగు; }
        ctx = c; పోర్టు = pt; pa = p; gid_idx = gi;
        స్థిరరాశి అక్షరం * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        ఆపు;
    }
    ibv_free_device_list(devs);
    ఒకవేళ (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        తిరిగి శూన్యసూచిక;
    }

    std::unique_ptr<impl> c(కొత్త impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->పోర్టు = పోర్టు;
    c->gid_idx = gid_idx;
    c->మార్గం_mtu = pa.చురుకైన_mtu;

    c->pd = ibv_alloc_pd(ctx);
    ఒకవేళ (!c->pd) తిరిగి శూన్యసూచిక;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, శూన్యసూచిక, శూన్యసూచిక, 0);
    ఒకవేళ (!c->cq) తిరిగి శూన్యసూచిక;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    ఒకవేళ (!c->qp) తిరిగి శూన్యసూచిక;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_సూచీ = 0;
        a.పోర్టు_అంకె = పోర్టు;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ఒకవేళ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_సూచీ | IBV_QP_పోర్టు | IBV_QP_ACCESS_FLAGS) != 0) {
            తిరిగి శూన్యసూచిక;
        }
    }

    దీర్ఘ పుట = sysconf(_SC_PAGESIZE);
    ఒకవేళ (పుట <= 0) పుట = 4096;
    స్థిరరాశి size_t ring_బైట్లు = (size_t)RDMA_NBUF * RDMA_STRIDE;
    ఒకవేళ (posix_memalign((శూన్య **)&c->send_mem, (size_t)పుట, ring_బైట్లు) != 0) c->send_mem = శూన్యసూచిక;
    ఒకవేళ (posix_memalign((శూన్య **)&c->recv_mem, (size_t)పుట, ring_బైట్లు) != 0) c->recv_mem = శూన్యసూచిక;
    ఒకవేళ (!c->send_mem || !c->recv_mem) తిరిగి శూన్యసూచిక;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    స్థిరరాశి పూర్ణాంకం mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_బైట్లు, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_బైట్లు, mr_flags);
    ఒకవేళ (!c->send_mr || !c->recv_mr) తిరిగి శూన్యసూచిక;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_అంకె;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    జ్ఞాపకం_నకలు(rc.gid, gid.raw, RDMA_GID_కొలత);
    జ్ఞాపకం_నకలు(caps, &rc, పరిమాణం(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), పోర్టు, gid_idx, c->qpn, (గుర్తులేని)pa.lid, 128 << c->మార్గం_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    తిరిగి std::unique_ptr<apple_rdma>(కొత్త apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
తార్కిక apple_rdma::activate(స్థిరరాశి uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    జ్ఞాపకం_నకలు(&rc, caps, పరిమాణం(rc));

    తార్కిక ok = నిజం;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.మార్గం_mtu   = c->మార్గం_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_అంకె = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.పోర్టు_అంకె      = c->పోర్టు;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_మార్గం_బిట్లు = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_హద్దు  = 1;
        a.ah_attr.grh.sgid_సూచీ = (uint8_t)c->gid_idx;
        జ్ఞాపకం_నకలు(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_కొలత);
        ఒకవేళ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_మార్గం_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = అబద్ధం;
        }
    }
    ఒకవేళ (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        ఒకవేళ (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = అబద్ధం;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    కొరకు (పూర్ణాంకం i = 0; ok && i < RDMA_NBUF; i++) {
        ఒకవేళ (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = అబద్ధం;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    ఒకవేళ (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ok || peer_ready != RDMA_SYNC_READY) {
        తిరిగి అబద్ధం;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->మార్గం_mtu, RDMA_NBUF);
    తిరిగి నిజం;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
పూర్ణాంకం apple_rdma::impl::పురోగతి() {
    నిర్మాణం ibv_wc wc[RDMA_NBUF * 2];
    పూర్ణాంకం n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    ఒకవేళ (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = నిజం; తిరిగి -1; }
    కొరకు (పూర్ణాంకం j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        తార్కిక is_recv = (id & RDMA_RECV_WR) != 0;
        ఒకవేళ (wc[j].స్థితి_2 != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].స్థితి_2);
            broken = నిజం;
            తిరిగి -1;
        }
        ఒకవేళ (is_recv) {
            పూర్ణాంకం b = (పూర్ణాంకం)(id & RDMA_WR_IDX_ముసుగు);
            స్థిరరాశి rdma_seg_hdr * h = (స్థిరరాశి rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            ఒకవేళ (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = నిజం; తిరిగి -1; }
            ఒకవేళ (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = నిజం; తిరిగి -1; }
            పూర్ణాంకం స్థానం = (inq_తల + inq_లెక్క) % RDMA_NBUF;
            inq[స్థానం].buf  = b;
            inq[స్థానం].off  = 0;
            inq[స్థానం].len  = h->len;
            inq_లెక్క++;
        } లేకపోతే {
            send_busy[(పూర్ణాంకం)(id & RDMA_WR_IDX_ముసుగు)] = 0;
        }
    }
    తిరిగి n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
తార్కిక apple_rdma::impl::acquire_pending() {
    ఒకవేళ (pend_buf >= 0) తిరిగి నిజం;
    కొరకు (;;) {
        ఒకవేళ (broken) తిరిగి అబద్ధం;
        కొరకు (పూర్ణాంకం k = 0; k < RDMA_NBUF; k++) ఒకవేళ (!send_busy[k]) { pend_buf = k; pend_len = 0; తిరిగి నిజం; }
        ఒకవేళ (పురోగతి() < 0) తిరిగి అబద్ధం;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
తార్కిక apple_rdma::impl::post_pending() {
    ఒకవేళ (pend_buf < 0) తిరిగి నిజం;
    పూర్ణాంకం i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    ఒకవేళ (!post_send(i, RDMA_STRIDE)) { broken = నిజం; తిరిగి అబద్ధం; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    తిరిగి నిజం;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
తార్కిక apple_rdma::send(స్థిరరాశి శూన్య * దత్తాంశం, size_t size) {
    impl * c = pimpl.get();
    స్థిరరాశి uint8_t * p = (స్థిరరాశి uint8_t *)దత్తాంశం;
    అయితే (size > 0) {
        ఒకవేళ (c->broken) తిరిగి అబద్ధం;
        ఒకవేళ (!c->acquire_pending()) తిరిగి అబద్ధం;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        జ్ఞాపకం_నకలు(sb + పరిమాణం(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        ఒకవేళ (c->pend_len == RDMA_PAYLOAD) { ఒకవేళ (!c->post_pending()) తిరిగి అబద్ధం; }
    }
    తిరిగి నిజం;
}

తార్కిక apple_rdma::recv(శూన్య * దత్తాంశం, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)దత్తాంశం;
    ఒకవేళ (!c->post_pending()) తిరిగి అబద్ధం;   // turnaround: flush the coalesced request
    గుర్తులేని idle = 0;
    అయితే (size > 0) {
        ఒకవేళ (c->inq_లెక్క == 0) {
            ఒకవేళ (c->broken) తిరిగి అబద్ధం;
            పూర్ణాంకం n = c->పురోగతి();
            ఒకవేళ (n < 0) తిరిగి అబద్ధం;
            ఒకవేళ (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                ఒకవేళ ((++idle & 0xFFFFF) == 0) {
                    నిర్మాణం pollfd pfd = { c->fd, POLLIN, 0 };
                    ఒకవేళ (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        తిరిగి అబద్ధం;
                    }
                }
            } లేకపోతే {
                idle = 0;
            }
            కొనసాగు;
        }
        idle = 0;
        పూర్ణాంకం స్థానం = c->inq_తల;
        పూర్ణాంకం b = c->inq[స్థానం].buf;
        uint32_t avail = c->inq[స్థానం].len - c->inq[స్థానం].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        జ్ఞాపకం_నకలు(p, c->recv_mem + (size_t)b * RDMA_STRIDE + పరిమాణం(rdma_seg_hdr) + c->inq[స్థానం].off, take);
        p += take;
        size -= take;
        c->inq[స్థానం].off += take;
        ఒకవేళ (c->inq[స్థానం].off == c->inq[స్థానం].len) {
            ఒకవేళ (!c->post_recv(b)) { c->broken = నిజం; తిరిగి అబద్ధం; }
            c->inq_తల = (c->inq_తల + 1) % RDMA_NBUF;
            c->inq_లెక్క--;
        }
    }
    తిరిగి నిజం;
}

తార్కిక apple_rdma::flush() {
    తిరిగి pimpl->post_pending();
}
