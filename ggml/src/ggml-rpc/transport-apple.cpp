#अन्तर्भाव "transport-apple.h"
#अन्तर्भाव "transport.h"
#अन्तर्भाव "log.h"

#अन्तर्भाव <infiniband/verbs.h>

#अन्तर्भाव <cerrno>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <string>
#अन्तर्भाव <dlfcn.h>
#अन्तर्भाव <poll.h>
#अन्तर्भाव <sys/socket.h>
#अन्तर्भाव <unistd.h>

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
स्थिर constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
स्थिर constexpr पूर्णांक      RDMA_NBUF        = 16;          // ring depth (frames per direction)
स्थिर constexpr माप_प्रकार   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
स्थिर constexpr माप_प्रकार   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
स्थिर constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
स्थिर constexpr माप_प्रकार   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
स्थिर constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
स्थिर constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
स्थिर constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
स्थिर constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

संरचना rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
स्थिर constexpr माप_प्रकार RDMA_PAYLOAD = RDMA_STRIDE - आकार(rdma_seg_hdr);

संरचना apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(आकार(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

संरचना apple_rdma::impl {
    पूर्णांक fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    संरचना ibv_context * ctx = nullptr;
    संरचना ibv_pd * pd = nullptr;
    संरचना ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    संरचना ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    संरचना ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    संरचना ibv_mr * recv_mr  = nullptr;

    पूर्णांक      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    संरचना { पूर्णांक buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    पूर्णांक      inq_head = 0;
    पूर्णांक      inq_मोजणी = 0;
    पूर्णांक      pend_buf = -1;
    uint32_t pend_len = 0;
    तार्किक     broken = असत्य;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    पूर्णांक          gid_idx = 0;
    गणना ibv_mtu path_mtu = IBV_MTU_1024;

    पूर्णांक  progress();
    तार्किक acquire_pending();
    तार्किक post_pending();

    तार्किक post_recv(पूर्णांक i) {
        संरचना ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (माप_प्रकार)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        संरचना ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;
        परत ibv_post_recv(qp, &wr, &bad) == 0;
    }

    तार्किक post_send(पूर्णांक i, माप_प्रकार len) {
        संरचना ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (माप_प्रकार)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        संरचना ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        परत ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = सत्य;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        जर (qp) ibv_destroy_qp(qp);
        जर (send_mr) ibv_dereg_mr(send_mr);
        जर (recv_mr) ibv_dereg_mr(recv_mr);
        मुक्त(send_mem);
        मुक्त(recv_mem);
        जर (cq)  ibv_destroy_cq(cq);
        जर (pd)  ibv_dealloc_pd(pd);
        जर (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = पूर्वनिर्धारित;

तार्किक apple_rdma::broken() स्थिरांक {
    परत pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
स्थिर तार्किक tcp_send_byte(पूर्णांक fd, uint8_t b) {
    ssize_t n;
    करा { n = ::send(fd, &b, आकार(b), 0); } पर्यंत (n < 0 && त्रुटि_सं == EINTR);
    परत n == आकार(b);
}

स्थिर तार्किक tcp_recv_byte(पूर्णांक fd, uint8_t * b) {
    ssize_t n;
    करा { n = ::recv(fd, b, आकार(*b), 0); } पर्यंत (n < 0 && त्रुटि_सं == EINTR);
    परत n == (ssize_t)आकार(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
स्थिर पूर्णांक rdma_match_gid(संरचना ibv_context * ctx, uint8_t port, पूर्णांक gid_tbl_len,
                          स्थिरांक uint8_t * target, संघ ibv_gid * out) {
    साठी (पूर्णांक i = 0; i < gid_tbl_len; i++) {
        संघ ibv_gid g;
        जर (ibv_query_gid(ctx, port, i, &g) != 0) सुरूठेवा;
        जर (स_भेद(g.raw, target, RDMA_GID_SIZE) != 0) सुरूठेवा;
        जर (out) *out = g;
        परत i;
    }
    परत -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
स्थिर uint8_t rdma_first_active_port(संरचना ibv_context * ctx, संरचना ibv_port_attr * out) {
    संरचना ibv_device_attr da;
    जर (ibv_query_device(ctx, &da) != 0) परत 0;
    साठी (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        संरचना ibv_port_attr pa;
        जर (ibv_query_port(ctx, p, &pa) != 0) सुरूठेवा;
        जर (pa.state == IBV_PORT_ACTIVE) { जर (out) *out = pa; परत p; }
    }
    परत 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
स्थिर तार्किक rdma_library_present() {
    स्थिर स्थिरांक तार्किक present = [] {
        शून्य * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        जर (handle == nullptr) {
            परत असत्य;
        }
        dlclose(handle);
        परत सत्य;
    }();
    परत present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(पूर्णांक fd, स्थिरांक uint8_t * target_gid, uint8_t * caps) {
    जर (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        परत nullptr;
    }
    पूर्णांक ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    जर (!devs) परत nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    संरचना ibv_port_attr pa = {};
    संघ ibv_gid gid = {};
    पूर्णांक gid_idx = -1;
    std::string matched;
    साठी (पूर्णांक d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        जर (!c) सुरूठेवा;
        संरचना ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        पूर्णांक gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        जर (gi < 0) { ibv_close_device(c); सुरूठेवा; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        स्थिरांक अक्षर * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        थांबा;
    }
    ibv_free_device_list(devs);
    जर (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        परत nullptr;
    }

    std::unique_ptr<impl> c(नवीन impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    जर (!c->pd) परत nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    जर (!c->cq) परत nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    जर (!c->qp) परत nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_अंक = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        जर (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            परत nullptr;
        }
    }

    दीर्घ page = sysconf(_SC_PAGESIZE);
    जर (page <= 0) page = 4096;
    स्थिरांक माप_प्रकार ring_bytes = (माप_प्रकार)RDMA_NBUF * RDMA_STRIDE;
    जर (posix_memalign((शून्य **)&c->send_mem, (माप_प्रकार)page, ring_bytes) != 0) c->send_mem = nullptr;
    जर (posix_memalign((शून्य **)&c->recv_mem, (माप_प्रकार)page, ring_bytes) != 0) c->recv_mem = nullptr;
    जर (!c->send_mem || !c->recv_mem) परत nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    स्थिरांक पूर्णांक mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    जर (!c->send_mr || !c->recv_mr) परत nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_अंक;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    स_नकल(rc.gid, gid.raw, RDMA_GID_SIZE);
    स_नकल(caps, &rc, आकार(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (अचिन्हित)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    परत std::unique_ptr<apple_rdma>(नवीन apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
तार्किक apple_rdma::activate(स्थिरांक uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    स_नकल(&rc, caps, आकार(rc));

    तार्किक ok = सत्य;
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
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        स_नकल(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        जर (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", म_त्रुटि(त्रुटि_सं));
            ok = असत्य;
        }
    }
    जर (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        जर (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", म_त्रुटि(त्रुटि_सं));
            ok = असत्य;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    साठी (पूर्णांक i = 0; ok && i < RDMA_NBUF; i++) {
        जर (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = असत्य;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    जर (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        परत असत्य;
    }
    जर (!ok || peer_ready != RDMA_SYNC_READY) {
        परत असत्य;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    परत सत्य;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
पूर्णांक apple_rdma::impl::progress() {
    संरचना ibv_wc wc[RDMA_NBUF * 2];
    पूर्णांक n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    जर (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = सत्य; परत -1; }
    साठी (पूर्णांक j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        तार्किक is_recv = (id & RDMA_RECV_WR) != 0;
        जर (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = सत्य;
            परत -1;
        }
        जर (is_recv) {
            पूर्णांक b = (पूर्णांक)(id & RDMA_WR_IDX_MASK);
            स्थिरांक rdma_seg_hdr * h = (स्थिरांक rdma_seg_hdr *)(recv_mem + (माप_प्रकार)b * RDMA_STRIDE);
            जर (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = सत्य; परत -1; }
            जर (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = सत्य; परत -1; }
            पूर्णांक slot = (inq_head + inq_मोजणी) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_मोजणी++;
        } अन्यथा {
            send_busy[(पूर्णांक)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    परत n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
तार्किक apple_rdma::impl::acquire_pending() {
    जर (pend_buf >= 0) परत सत्य;
    साठी (;;) {
        जर (broken) परत असत्य;
        साठी (पूर्णांक k = 0; k < RDMA_NBUF; k++) जर (!send_busy[k]) { pend_buf = k; pend_len = 0; परत सत्य; }
        जर (progress() < 0) परत असत्य;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
तार्किक apple_rdma::impl::post_pending() {
    जर (pend_buf < 0) परत सत्य;
    पूर्णांक i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (माप_प्रकार)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    जर (!post_send(i, RDMA_STRIDE)) { broken = सत्य; परत असत्य; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    परत सत्य;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
तार्किक apple_rdma::send(स्थिरांक शून्य * data, माप_प्रकार size) {
    impl * c = pimpl.get();
    स्थिरांक uint8_t * p = (स्थिरांक uint8_t *)data;
    पर्यंत (size > 0) {
        जर (c->broken) परत असत्य;
        जर (!c->acquire_pending()) परत असत्य;
        uint8_t * sb = c->send_mem + (माप_प्रकार)c->pend_buf * RDMA_STRIDE;
        माप_प्रकार space = RDMA_PAYLOAD - c->pend_len;
        माप_प्रकार chunk = size < space ? size : space;
        स_नकल(sb + आकार(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        जर (c->pend_len == RDMA_PAYLOAD) { जर (!c->post_pending()) परत असत्य; }
    }
    परत सत्य;
}

तार्किक apple_rdma::recv(शून्य * data, माप_प्रकार size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    जर (!c->post_pending()) परत असत्य;   // turnaround: flush the coalesced request
    अचिन्हित idle = 0;
    पर्यंत (size > 0) {
        जर (c->inq_मोजणी == 0) {
            जर (c->broken) परत असत्य;
            पूर्णांक n = c->progress();
            जर (n < 0) परत असत्य;
            जर (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                जर ((++idle & 0xFFFFF) == 0) {
                    संरचना pollfd pfd = { c->fd, POLLIN, 0 };
                    जर (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        परत असत्य;
                    }
                }
            } अन्यथा {
                idle = 0;
            }
            सुरूठेवा;
        }
        idle = 0;
        पूर्णांक slot = c->inq_head;
        पूर्णांक b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (माप_प्रकार)avail) ? (uint32_t)size : avail;
        स_नकल(p, c->recv_mem + (माप_प्रकार)b * RDMA_STRIDE + आकार(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        जर (c->inq[slot].off == c->inq[slot].len) {
            जर (!c->post_recv(b)) { c->broken = सत्य; परत असत्य; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_मोजणी--;
        }
    }
    परत सत्य;
}

तार्किक apple_rdma::flush() {
    परत pimpl->post_pending();
}
