#అన్తర్భావ "transport.h"
#అన్తర్భావ "log.h"

#నిర్వచితమైతే _WIN32
#  define WIN32_LEAN_AND_సగటు
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#లేకపోతే
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#అయితే_ముగింపు
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <mutex>
#అన్తర్భావ <optional>

#నిర్వచితమైతే GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <chrono>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#అయితే_ముగింపు // GGML_RPC_RDMA

#నిర్వచితమైతే _WIN32
రకనిర్వచనం SOCKET sockfd_t;
ఉపయోగించి ssize_t = __int64;
#లేకపోతే
రకనిర్వచనం పూర్ణాంకం sockfd_t;
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_RPC_RDMA
స్థిర constexpr size_t RDMA_GID_కొలత = 16;            // RoCE GID / IB GID is always 16 bytes
ఉపయోగించి rdma_gid_t = std::array<uint8_t, RDMA_GID_కొలత>;
#అయితే_ముగింపు // GGML_RPC_RDMA

#ఒకవేళ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
స్థిర constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
స్థిర constexpr పూర్ణాంకం    RDMA_RX_లోతు = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
స్థిర constexpr స్వయంచాలక   RDMA_SPIN_సమయం = std::chrono::milliseconds(100);

నిర్మాణం rdma_conn {
    నిర్మాణం ibv_context * ctx = శూన్యసూచిక;
    నిర్మాణం ibv_pd * pd  = శూన్యసూచిక;
    నిర్మాణం ibv_cq * scq = శూన్యసూచిక;   // send completions
    నిర్మాణం ibv_cq * rcq = శూన్యసూచిక;   // recv completions
    నిర్మాణం ibv_qp * qp  = శూన్యసూచిక;
    నిర్మాణం ibv_comp_channel * ch = శూన్యసూచిక; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point చివరి_చురుకైన; // last completion or posted send

    శూన్య          * tx_buf = శూన్యసూచిక;
    నిర్మాణం ibv_mr * tx_mr  = శూన్యసూచిక;

    శూన్య          * rx_buf = శూన్యసూచిక; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    నిర్మాణం ibv_mr * rx_mr  = శూన్యసూచిక;
    పూర్ణాంకం             rx_తల = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_స్థానం(పూర్ణాంకం i) స్థిరరాశి {
        తిరిగి static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    తార్కిక post_rx(పూర్ణాంకం i) {
        నిర్మాణం ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_స్థానం(i);
        sge.నిడివి = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        నిర్మాణం ibv_recv_wr wr = {}, * bad = శూన్యసూచిక;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.అంకె_sge = 1;
        తిరిగి ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        ఒకవేళ (tx_mr) ibv_dereg_mr(tx_mr);
        ఒకవేళ (rx_mr) ibv_dereg_mr(rx_mr);
        విడుదల(tx_buf);
        విడుదల(rx_buf);
        ఒకవేళ (qp)  ibv_destroy_qp(qp);
        ఒకవేళ (scq) ibv_destroy_cq(scq);
        ఒకవేళ (rcq) ibv_destroy_cq(rcq);
        ఒకవేళ (ch)  ibv_destroy_comp_channel(ch);
        ఒకవేళ (pd)  ibv_dealloc_pd(pd);
        ఒకవేళ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
నిర్మాణం rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_కొలత] = {};
    uint8_t  ib_పోర్టు = 0;
    పూర్ణాంకం      gid_idx = 0;
    గణన ibv_mtu మార్గం_mtu = IBV_MTU_1024;
};

నిర్మాణం rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_కొలత];
};

static_assert(పరిమాణం(rdma_caps) == RPC_CONN_CAPS_కొలత, "rdma_caps must match conn_caps size");

#అయితే_ముగింపు // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

నిర్మాణం socket_t::impl {
    impl(sockfd_t fd) : use_rdma(అబద్ధం), fd(fd) {}
    ~impl();
    తార్కిక send_దత్తాంశం(స్థిరరాశి శూన్య * దత్తాంశం, size_t size);
    తార్కిక recv_దత్తాంశం(శూన్య * దత్తాంశం, size_t size);
    తార్కిక flush();
    శూన్య get_caps(uint8_t * local_caps);
    శూన్య update_caps(స్థిరరాశి uint8_t * remote_caps);

#నిర్వచితమైతే GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_లక్ష్యం_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  లేకపోతే
    తార్కిక rdma_probe();
    తార్కిక rdma_send(స్థిరరాశి శూన్య * దత్తాంశం, size_t size);
    తార్కిక rdma_recv(శూన్య * దత్తాంశం, size_t size);
    తార్కిక tcp_peer_closed();
    తార్కిక rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, స్థిరరాశి uint8_t * remote_gid);
    తార్కిక rdma_poll(నిర్మాణం ibv_cq * cq, నిర్మాణం ibv_wc * wc);
    తార్కిక rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#అయితే_ముగింపు // GGML_RPC_RDMA
    తార్కిక     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#నిర్వచితమైతే GGML_RPC_RDMA
    rdma.reset();
#అయితే_ముగింపు // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, ఇది->fd);
#నిర్వచితమైతే _WIN32
    ఒకవేళ (fd != INVALID_SOCKET) closesocket(ఇది->fd);
#లేకపోతే
    ఒకవేళ (fd >= 0) close(ఇది->fd);
#అయితే_ముగింపు
}

#నిర్వచితమైతే GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_లక్ష్యం_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = పరిమాణం(addr);
    ఒకవేళ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        తిరిగి std::nullopt;
    }
    rdma_gid_t లక్ష్యం = {};
    ఒకవేళ (addr.ss_family == AF_INET) {
        స్థిరరాశి స్వయంచాలక * a = reinterpret_cast<స్థిరరాశి sockaddr_లోపల *>(&addr);
        లక్ష్యం[10] = 0xff;
        లక్ష్యం[11] = 0xff;
        జ్ఞాపకం_నకలు(&లక్ష్యం[12], &a->sin_addr, 4);
        తిరిగి లక్ష్యం;
    }
    ఒకవేళ (addr.ss_family == AF_INET6) {
        స్థిరరాశి స్వయంచాలక * a = reinterpret_cast<స్థిరరాశి sockaddr_in6 *>(&addr);
        జ్ఞాపకం_నకలు(లక్ష్యం.దత్తాంశం(), &a->sin6_addr, RDMA_GID_కొలత);
        తిరిగి లక్ష్యం;
    }
    తిరిగి std::nullopt;
}

#నిర్వచితం_కాకపోతే GGML_RPC_RDMA_APPLE

తార్కిక socket_t::impl::tcp_peer_closed() {
    ఒకవేళ (fd < 0) తిరిగి అబద్ధం;
#నిర్వచితం_కాకపోతే _WIN32
    నిర్మాణం pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    పూర్ణాంకం r = poll(&pfd, 1, 0);
    తిరిగి r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#లేకపోతే
    తిరిగి అబద్ధం;
#అయితే_ముగింపు
}

తార్కిక socket_t::impl::rdma_probe() {
    స్థిరరాశి అక్షరం * dev_env = std::getenv("GGML_RDMA_DEV");
    స్థిరరాశి అక్షరం * gid_env = std::getenv("GGML_RDMA_GID");

    స్వయంచాలక లక్ష్యం_gid = rdma_build_లక్ష్యం_gid();
    ఒకవేళ (!లక్ష్యం_gid) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint8_t ib_పోర్టు = 1;
    పూర్ణాంకం అంకె_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&అంకె_devs);
    ఒకవేళ (!devs || అంకె_devs == 0) తిరిగి అబద్ధం;

    ibv_context * ibctx = శూన్యసూచిక;
    స్థిరరాశి అక్షరం * matched_dev = శూన్యసూచిక;
    పూర్ణాంకం gid_idx = gid_env ? atoi(gid_env) : -1;
    పూర్ణాంకం gid_సంస్కరణ = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    కొరకు (పూర్ణాంకం d = 0; d < అంకె_devs; d++) {
        స్థిరరాశి అక్షరం * dn = ibv_get_device_name(devs[d]);
        ఒకవేళ (dev_env && strcmp(dev_env, dn) != 0) కొనసాగు;

        ibv_context * ctx = ibv_open_device(devs[d]);
        ఒకవేళ (!ctx) కొనసాగు;

        ibv_పోర్టు_attr pa;
        ఒకవేళ (ibv_query_పోర్టు(ctx, ib_పోర్టు, &pa) != 0) { ibv_close_device(ctx); కొనసాగు; }

        పూర్ణాంకం దొరికింది_gid = gid_idx;
        పూర్ణాంకం దొరికింది_సంస్కరణ = IBV_GID_TYPE_IB;
        ఒకవేళ (దొరికింది_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            పూర్ణాంకం v2_idx = -1;
            పూర్ణాంకం v1_idx = -1;
            కొరకు (పూర్ణాంకం i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_నమోదు నమోదు_2 = {};
                ఒకవేళ (ibv_query_gid_ex(ctx, ib_పోర్టు, i, &నమోదు_2, 0) != 0) కొనసాగు;
                ఒకవేళ (memcmp(నమోదు_2.gid.raw, లక్ష్యం_gid->దత్తాంశం(), RDMA_GID_కొలత) != 0) కొనసాగు;
                ఒకవేళ (నమోదు_2.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } లేకపోతే ఒకవేళ (నమోదు_2.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            ఒకవేళ (v2_idx >= 0) {
                దొరికింది_gid = v2_idx;
                దొరికింది_సంస్కరణ = IBV_GID_TYPE_ROCE_V2;
            } లేకపోతే ఒకవేళ (v1_idx >= 0) {
                దొరికింది_gid = v1_idx;
                దొరికింది_సంస్కరణ = IBV_GID_TYPE_ROCE_V1;
            }
        } లేకపోతే {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_నమోదు నమోదు_2 = {};
            ఒకవేళ (ibv_query_gid_ex(ctx, ib_పోర్టు, దొరికింది_gid, &నమోదు_2, 0) == 0) {
                దొరికింది_సంస్కరణ = నమోదు_2.gid_type;
            }
        }
        ఒకవేళ (దొరికింది_gid >= 0) {
            ibctx = ctx;
            gid_idx = దొరికింది_gid;
            gid_సంస్కరణ = దొరికింది_సంస్కరణ;
            matched_dev = dn;
            rdma_local.మార్గం_mtu = pa.చురుకైన_mtu;
            ఆపు;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    ఒకవేళ (!ibctx) తిరిగి అబద్ధం;

    rdma_local.ib_పోర్టు = ib_పోర్టు;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    ఒకవేళ (!rdma->pd) తిరిగి అబద్ధం;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, శూన్యసూచిక, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_లోతు + 4, శూన్యసూచిక, rdma->ch, 0);
    ఒకవేళ (!rdma->scq || !rdma->rcq) తిరిగి అబద్ధం;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_లోతు + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_దత్తాంశం = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    ఒకవేళ (!rdma->qp) తిరిగి అబద్ధం;
    rdma->max_inline = qia.cap.max_inline_దత్తాంశం;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_లోతు) * RDMA_CHUNK);
    ఒకవేళ (!rdma->tx_buf || !rdma->rx_buf) తిరిగి అబద్ధం;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_లోతు) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ఒకవేళ (!rdma->tx_mr || !rdma->rx_mr) తిరిగి అబద్ధం;

    ibv_gid local_gid;
    ఒకవేళ (ibv_query_gid(ibctx, ib_పోర్టు, gid_idx, &local_gid) != 0) తిరిగి అబద్ధం;

    rdma_local.qpn = rdma->qp->qp_అంకె;
    rdma_local.psn = rdma->qp->qp_అంకె & 0xffffff;
    జ్ఞాపకం_నకలు(&rdma_local.gid, &local_gid, RDMA_GID_కొలత);

    స్థిరరాశి అక్షరం * ver_str = "";
    ఒకవేళ (gid_సంస్కరణ == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } లేకపోతే ఒకవేళ (gid_సంస్కరణ == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    తిరిగి నిజం;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
తార్కిక socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, స్థిరరాశి uint8_t * remote_gid) {
    // RESET -> INIT
    {
        నిర్మాణం ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.పోర్టు_అంకె        = rdma_local.ib_పోర్టు;
        a.pkey_సూచీ      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ఒకవేళ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_సూచీ | IBV_QP_పోర్టు | IBV_QP_ACCESS_FLAGS) != 0) {
            తిరిగి అబద్ధం;
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < RDMA_RX_లోతు; i++) {
        ఒకవేళ (!rdma->post_rx(i)) తిరిగి అబద్ధం;
    }

    // INIT -> RTR
    {
        నిర్మాణం ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.మార్గం_mtu           = rdma_local.మార్గం_mtu;
        a.dest_qp_అంకె        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        జ్ఞాపకం_నకలు(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_కొలత);
        a.ah_attr.grh.hop_హద్దు  = 1;
        a.ah_attr.grh.sgid_సూచీ = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.పోర్టు_అంకె   = rdma_local.ib_పోర్టు;
        ఒకవేళ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_మార్గం_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            తిరిగి అబద్ధం;
        }
    }

    // RTR -> RTS
    {
        నిర్మాణం ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        ఒకవేళ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            తిరిగి అబద్ధం;
        }
    }

    rdma->చివరి_చురుకైన = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.మార్గం_mtu, RDMA_RX_లోతు);
    తిరిగి నిజం;
}

// Sleep until the completion channel has an event or the TCP peer closes.
తార్కిక socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    నిర్మాణం pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    ఒకవేళ (poll(pfds, 2, -1) < 0) {
        తిరిగి errno == EINTR;
    }
    ఒకవేళ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (pfds[0].revents & POLLIN) {
        నిర్మాణం ibv_cq * ev_cq  = శూన్యసూచిక;
        శూన్య          * ev_ctx = శూన్యసూచిక;
        ఒకవేళ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            తిరిగి అబద్ధం;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    తిరిగి నిజం;
}

తార్కిక socket_t::impl::rdma_poll(నిర్మాణం ibv_cq * cq, నిర్మాణం ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    తార్కిక armed = అబద్ధం;
    కొరకు (uint64_t s = 1; ; s++) {
        పూర్ణాంకం n = ibv_poll_cq(cq, 1, wc);
        ఒకవేళ (n > 0) {
            c->చివరి_చురుకైన = std::chrono::steady_clock::now();
            ఒకవేళ (wc->స్థితి_2 != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->స్థితి_2, ibv_wc_స్థితి_str(wc->స్థితి_2), wc->vendor_err);
            }
            తిరిగి wc->స్థితి_2 == IBV_WC_SUCCESS;
        }
        ఒకవేళ (n < 0) తిరిగి అబద్ధం;
        ఒకవేళ (armed) {
            // armed and still empty: sleep until the next completion
            ఒకవేళ (!rdma_wait_event()) {
                తిరిగి అబద్ధం;
            }
            armed = అబద్ధం;
            కొనసాగు;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        ఒకవేళ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->చివరి_చురుకైన > RDMA_SPIN_సమయం) {
            ఒకవేళ (ibv_req_notify_cq(cq, 0) != 0) {
                తిరిగి అబద్ధం;
            }
            armed = నిజం;
            కొనసాగు;
        }
        ఒకవేళ ((s & 0xFFFFF) == 0) {
            ఒకవేళ (tcp_peer_closed()) {
                తిరిగి అబద్ధం;
            }
        }
    }
}

తార్కిక socket_t::impl::rdma_send(స్థిరరాశి శూన్య * దత్తాంశం, size_t size) {
    rdma_conn * c = rdma.get();
    స్థిరరాశి uint8_t * src = (స్థిరరాశి uint8_t *)దత్తాంశం;
    size_t rem = size;
    అయితే (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        నిర్మాణం ibv_sge sge = {};
        నిర్మాణం ibv_send_wr wr = {}, * bad = శూన్యసూచిక;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.అంకె_sge = 1;

        ఒకవేళ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.నిడివి = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } లేకపోతే {
            జ్ఞాపకం_నకలు(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.నిడివి = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        ఒకవేళ (ibv_post_send(c->qp, &wr, &bad) != 0) తిరిగి అబద్ధం;
        c->చివరి_చురుకైన = std::chrono::steady_clock::now();
        నిర్మాణం ibv_wc wc;
        ఒకవేళ (!rdma_poll(c->scq, &wc)) తిరిగి అబద్ధం;

        src += chunk;
        rem -= chunk;
    }
    తిరిగి నిజం;
}

తార్కిక socket_t::impl::rdma_recv(శూన్య * దత్తాంశం, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)దత్తాంశం;
    size_t rem = size;
    అయితే (rem > 0) {
        నిర్మాణం ibv_wc wc;
        ఒకవేళ (!rdma_poll(c->rcq, &wc)) తిరిగి అబద్ధం;

        పూర్ణాంకం స్థానం = (పూర్ణాంకం)wc.wr_id;
        size_t got = wc.byte_len;
        జ్ఞాపకం_నకలు(dst, c->rx_స్థానం(స్థానం), got);

        ఒకవేళ (!c->post_rx(స్థానం)) తిరిగి అబద్ధం;

        dst += got;
        rem -= got;
    }
    తిరిగి నిజం;
}

#అయితే_ముగింపు // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#అయితే_ముగింపు // GGML_RPC_RDMA

తార్కిక socket_t::impl::send_దత్తాంశం(స్థిరరాశి శూన్య * దత్తాంశం, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_కొలత(size).c_str());
#నిర్వచితమైతే GGML_RPC_RDMA_APPLE
    ఒకవేళ (use_rdma) {
        తిరిగి rdma->send(దత్తాంశం, size);
    }
#లేదావేళ defined(GGML_RPC_RDMA)
    ఒకవేళ (use_rdma) {
        తిరిగి rdma_send(దత్తాంశం, size);
    }
#అయితే_ముగింపు
    size_t బైట్లు_పంపిన = 0;
    అయితే (బైట్లు_పంపిన < size) {
        size_t కొలత_గమ్యం_send = std::min(size - బైట్లు_పంపిన, MAX_CHUNK_కొలత);
        ssize_t n = send(fd, (స్థిరరాశి అక్షరం *)దత్తాంశం + బైట్లు_పంపిన, కొలత_గమ్యం_send, 0);
        ఒకవేళ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           బైట్లు_పంపిన, కొలత_గమ్యం_send);
            తిరిగి అబద్ధం;
        }
        బైట్లు_పంపిన += (size_t)n;
    }
    తిరిగి నిజం;
}

తార్కిక socket_t::impl::recv_దత్తాంశం(శూన్య * దత్తాంశం, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_కొలత(size).c_str());
#నిర్వచితమైతే GGML_RPC_RDMA_APPLE
    ఒకవేళ (use_rdma) {
        తిరిగి rdma->recv(దత్తాంశం, size);
    }
#లేదావేళ defined(GGML_RPC_RDMA)
    ఒకవేళ (use_rdma) {
        తిరిగి rdma_recv(దత్తాంశం, size);
    }
#అయితే_ముగింపు
    size_t బైట్లు_recv = 0;
    అయితే (బైట్లు_recv < size) {
        size_t కొలత_గమ్యం_recv = std::min(size - బైట్లు_recv, MAX_CHUNK_కొలత);
        ssize_t n = recv(fd, (అక్షరం *)దత్తాంశం + బైట్లు_recv, కొలత_గమ్యం_recv, 0);
        ఒకవేళ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           బైట్లు_recv, కొలత_గమ్యం_recv);
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            తిరిగి అబద్ధం;
        }
        బైట్లు_recv += (size_t)n;
    }
    తిరిగి నిజం;
}

శూన్య socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_కొలత);
#నిర్వచితమైతే GGML_RPC_RDMA
    ఒకవేళ (std::getenv("GGML_RPC_NO_RDMA")) {
        తిరిగి;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    స్వయంచాలక లక్ష్యం_gid = rdma_build_లక్ష్యం_gid();
    ఒకవేళ (లక్ష్యం_gid) {
        rdma = apple_rdma::probe(fd, లక్ష్యం_gid->దత్తాంశం(), local_caps);
    }
    ఒకవేళ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  లేకపోతే
    rdma_local = {};
    ఒకవేళ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        జ్ఞాపకం_నకలు(rc.gid, rdma_local.gid, RDMA_GID_కొలత);
        జ్ఞాపకం_నకలు(local_caps, &rc, పరిమాణం(rc));
    } లేకపోతే {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#అయితే_ముగింపు // GGML_RPC_RDMA
}

శూన్య socket_t::impl::update_caps(స్థిరరాశి uint8_t * remote_caps) {
#నిర్వచితమైతే GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    తార్కిక remote_rdma = అబద్ధం;
    కొరకు (size_t i = 0; i < RPC_CONN_CAPS_కొలత; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    ఒకవేళ (!rdma || !remote_rdma) {
        ఒకవేళ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        తిరిగి;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    తార్కిక activated = rdma->activate(remote_caps);
#  లేకపోతే
    rdma_caps rc = {};
    జ్ఞాపకం_నకలు(&rc, remote_caps, పరిమాణం(rc));
    తార్కిక activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    ఒకవేళ (activated) {
        use_rdma = నిజం;
    } లేకపోతే {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#లేకపోతే
    (శూన్య)remote_caps;
#అయితే_ముగింపు // GGML_RPC_RDMA
}

తార్కిక socket_t::impl::flush() {
#నిర్వచితమైతే GGML_RPC_RDMA_APPLE
    ఒకవేళ (use_rdma) {
        తిరిగి rdma->flush();
    }
#అయితే_ముగింపు
    తిరిగి నిజం;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = అప్రమేయ;

తార్కిక socket_t::send_దత్తాంశం(స్థిరరాశి శూన్య * దత్తాంశం, size_t size) {
    తిరిగి pimpl->send_దత్తాంశం(దత్తాంశం, size);
}

తార్కిక socket_t::recv_దత్తాంశం(శూన్య * దత్తాంశం, size_t size) {
    తిరిగి pimpl->recv_దత్తాంశం(దత్తాంశం, size);
}

తార్కిక socket_t::flush() {
    తిరిగి pimpl->flush();
}

శూన్య socket_t::get_caps(uint8_t * local_caps) {
    తిరిగి pimpl->get_caps(local_caps);
}

శూన్య socket_t::update_caps(స్థిరరాశి uint8_t * remote_caps) {
    తిరిగి pimpl->update_caps(remote_caps);
}

స్థిరరాశి అక్షరం * socket_t::transport_name() స్థిరరాశి {
    తిరిగి pimpl->use_rdma ? "RDMA" : "TCP";
}

స్థిర తార్కిక is_valid_fd(sockfd_t sockfd) {
#నిర్వచితమైతే _WIN32
    తిరిగి sockfd != INVALID_SOCKET;
#లేకపోతే
    తిరిగి sockfd >= 0;
#అయితే_ముగింపు
}

స్థిర తార్కిక సమితి_no_delay(sockfd_t sockfd) {
    పూర్ణాంకం flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    పూర్ణాంకం ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (అక్షరం *)&flag, పరిమాణం(పూర్ణాంకం));
    తిరిగి ret == 0;
}

స్థిర తార్కిక సమితి_reuse_addr(sockfd_t sockfd) {
    పూర్ణాంకం flag = 1;
    పూర్ణాంకం ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (అక్షరం *)&flag, పరిమాణం(పూర్ణాంకం));
    తిరిగి ret == 0;
}

socket_ptr socket_t::accept() {
    స్వయంచాలక client_socket_fd = ::accept(pimpl->fd, ఖాళీ, ఖాళీ);
    ఒకవేళ (!is_valid_fd(client_socket_fd)) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (!సమితి_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        తిరిగి శూన్యసూచిక;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    తిరిగి socket_ptr(కొత్త socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(స్థిరరాశి అక్షరం * host, పూర్ణాంకం పోర్టు) {
    స్వయంచాలక sockfd = socket(AF_INET, SOCK_ప్రవాహం, 0);
    ఒకవేళ (!is_valid_fd(sockfd)) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (!సమితి_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        తిరిగి శూన్యసూచిక;
    }
    నిర్మాణం sockaddr_లోపల serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_పోర్టు = htons(పోర్టు);

    ఒకవేళ (bind(sockfd, (నిర్మాణం sockaddr *) &serv_addr, పరిమాణం(serv_addr)) < 0) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (listen(sockfd, 1) < 0) {
        తిరిగి శూన్యసూచిక;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, పోర్టు);
    తిరిగి socket_ptr(కొత్త socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(స్థిరరాశి అక్షరం * host, పూర్ణాంకం పోర్టు) {
    స్వయంచాలక sockfd = socket(AF_INET, SOCK_ప్రవాహం, 0);
    ఒకవేళ (!is_valid_fd(sockfd)) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (!సమితి_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        తిరిగి శూన్యసూచిక;
    }
    నిర్మాణం sockaddr_లోపల addr;
    addr.sin_family = AF_INET;
    addr.sin_పోర్టు = htons(పోర్టు);
    నిర్మాణం hostent * server = gethostbyname(host);
    ఒకవేళ (server == ఖాళీ) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        తిరిగి శూన్యసూచిక;
    }
    జ్ఞాపకం_నకలు(&addr.sin_addr.s_addr, server->h_addr, server->h_నిడివి);
    ఒకవేళ (::connect(sockfd, (నిర్మాణం sockaddr *)&addr, పరిమాణం(addr)) < 0) {
        తిరిగి శూన్యసూచిక;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, పోర్టు);
    తిరిగి socket_ptr(కొత్త socket_t(std::make_unique<impl>(sockfd)));
}

#నిర్వచితమైతే _WIN32
స్థిర std::mutex g_rpc_transport_mu;
స్థిర తార్కిక g_rpc_transport_wsa_started = అబద్ధం;
#అయితే_ముగింపు

తార్కిక rpc_transport_init() {
#నిర్వచితమైతే _WIN32
    std::lock_guard<std::mutex> తాళం(g_rpc_transport_mu);
    ఒకవేళ (g_rpc_transport_wsa_started) {
        తిరిగి నిజం;
    }
    WSADATA wsa_దత్తాంశం;
    పూర్ణాంకం res = WSAStartup(MAKEWORD(2, 2), &wsa_దత్తాంశం);
    ఒకవేళ (res != 0) {
        తిరిగి అబద్ధం;
    }
    g_rpc_transport_wsa_started = నిజం;
    తిరిగి నిజం;
#లేకపోతే
    తిరిగి నిజం;
#అయితే_ముగింపు
}

శూన్య rpc_transport_shutdown() {
#నిర్వచితమైతే _WIN32
    std::lock_guard<std::mutex> తాళం(g_rpc_transport_mu);
    ఒకవేళ (!g_rpc_transport_wsa_started) {
        తిరిగి;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = అబద్ధం;
#అయితే_ముగింపు
}
