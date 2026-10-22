#अन्तर्भाव "transport.h"
#अन्तर्भाव "log.h"

#यदि_परिभाषितम् _WIN32
#  define WIN32_LEAN_AND_माध्यम्
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#अन्यथा
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#यदि_अन्तः
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <mutex>
#अन्तर्भाव <optional>

#यदि_परिभाषितम् GGML_RPC_RDMA
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
#यदि_अन्तः // GGML_RPC_RDMA

#यदि_परिभाषितम् _WIN32
प्रकारनिर्देश SOCKET sockfd_t;
उपयुज्य ssize_t = __int64;
#अन्यथा
प्रकारनिर्देश पूर्णाङ्क sockfd_t;
#यदि_अन्तः

#यदि_परिभाषितम् GGML_RPC_RDMA
स्थिर constexpr size_t RDMA_GID_आकार = 16;            // RoCE GID / IB GID is always 16 bytes
उपयुज्य rdma_gid_t = std::array<uint8_t, RDMA_GID_आकार>;
#यदि_अन्तः // GGML_RPC_RDMA

#यदि defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
स्थिर constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
स्थिर constexpr पूर्णाङ्क    RDMA_RX_गभीरता = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
स्थिर constexpr स्वचालित   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

संरचना rdma_conn {
    संरचना ibv_context * ctx = शून्यसूचकः;
    संरचना ibv_pd * pd  = शून्यसूचकः;
    संरचना ibv_cq * scq = शून्यसूचकः;   // send completions
    संरचना ibv_cq * rcq = शून्यसूचकः;   // recv completions
    संरचना ibv_qp * qp  = शून्यसूचकः;
    संरचना ibv_comp_channel * ch = शून्यसूचकः; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point चरमः_सक्रिय; // last completion or posted send

    शून्य          * tx_buf = शून्यसूचकः;
    संरचना ibv_mr * tx_mr  = शून्यसूचकः;

    शून्य          * rx_buf = शून्यसूचकः; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    संरचना ibv_mr * rx_mr  = शून्यसूचकः;
    पूर्णाङ्क             rx_शीर्ष = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(पूर्णाङ्क i) ध्रुव {
        प्रत्यागम static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    तार्किक post_rx(पूर्णाङ्क i) {
        संरचना ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.दीर्घता = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        संरचना ibv_recv_wr wr = {}, * bad = शून्यसूचकः;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.अङ्कः_sge = 1;
        प्रत्यागम ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        यदि (tx_mr) ibv_dereg_mr(tx_mr);
        यदि (rx_mr) ibv_dereg_mr(rx_mr);
        मुक्त(tx_buf);
        मुक्त(rx_buf);
        यदि (qp)  ibv_destroy_qp(qp);
        यदि (scq) ibv_destroy_cq(scq);
        यदि (rcq) ibv_destroy_cq(rcq);
        यदि (ch)  ibv_destroy_comp_channel(ch);
        यदि (pd)  ibv_dealloc_pd(pd);
        यदि (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
संरचना rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_आकार] = {};
    uint8_t  ib_पत्तनम् = 0;
    पूर्णाङ्क      gid_idx = 0;
    गणना ibv_mtu मार्गः_mtu = IBV_MTU_1024;
};

संरचना rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_आकार];
};

static_assert(परिमाण(rdma_caps) == RPC_CONN_CAPS_आकार, "rdma_caps must match conn_caps size");

#यदि_अन्तः // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

संरचना socket_t::impl {
    impl(sockfd_t fd) : use_rdma(असत्य), fd(fd) {}
    ~impl();
    तार्किक send_दत्त(ध्रुव शून्य * दत्त, size_t size);
    तार्किक recv_दत्त(शून्य * दत्त, size_t size);
    तार्किक flush();
    शून्य get_caps(uint8_t * local_caps);
    शून्य update_caps(ध्रुव uint8_t * remote_caps);

#यदि_परिभाषितम् GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  अन्यथा
    तार्किक rdma_probe();
    तार्किक rdma_send(ध्रुव शून्य * दत्त, size_t size);
    तार्किक rdma_recv(शून्य * दत्त, size_t size);
    तार्किक tcp_peer_closed();
    तार्किक rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ध्रुव uint8_t * remote_gid);
    तार्किक rdma_poll(संरचना ibv_cq * cq, संरचना ibv_wc * wc);
    तार्किक rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#यदि_अन्तः // GGML_RPC_RDMA
    तार्किक     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#यदि_परिभाषितम् GGML_RPC_RDMA
    rdma.reset();
#यदि_अन्तः // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, एतत्->fd);
#यदि_परिभाषितम् _WIN32
    यदि (fd != INVALID_SOCKET) closesocket(एतत्->fd);
#अन्यथा
    यदि (fd >= 0) close(एतत्->fd);
#यदि_अन्तः
}

#यदि_परिभाषितम् GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = परिमाण(addr);
    यदि (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        प्रत्यागम std::nullopt;
    }
    rdma_gid_t target = {};
    यदि (addr.ss_family == AF_INET) {
        ध्रुव स्वचालित * a = reinterpret_cast<ध्रुव sockaddr_अन्तर् *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        स्मृतिं_प्रतिलिख(&target[12], &a->sin_addr, 4);
        प्रत्यागम target;
    }
    यदि (addr.ss_family == AF_INET6) {
        ध्रुव स्वचालित * a = reinterpret_cast<ध्रुव sockaddr_in6 *>(&addr);
        स्मृतिं_प्रतिलिख(target.दत्त(), &a->sin6_addr, RDMA_GID_आकार);
        प्रत्यागम target;
    }
    प्रत्यागम std::nullopt;
}

#यदि_अपरिभाषितम् GGML_RPC_RDMA_APPLE

तार्किक socket_t::impl::tcp_peer_closed() {
    यदि (fd < 0) प्रत्यागम असत्य;
#यदि_अपरिभाषितम् _WIN32
    संरचना pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    पूर्णाङ्क r = poll(&pfd, 1, 0);
    प्रत्यागम r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#अन्यथा
    प्रत्यागम असत्य;
#यदि_अन्तः
}

तार्किक socket_t::impl::rdma_probe() {
    ध्रुव वर्ण * dev_env = std::getenv("GGML_RDMA_DEV");
    ध्रुव वर्ण * gid_env = std::getenv("GGML_RDMA_GID");

    स्वचालित target_gid = rdma_build_target_gid();
    यदि (!target_gid) {
        प्रत्यागम असत्य;
    }

    ध्रुव uint8_t ib_पत्तनम् = 1;
    पूर्णाङ्क अङ्कः_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&अङ्कः_devs);
    यदि (!devs || अङ्कः_devs == 0) प्रत्यागम असत्य;

    ibv_context * ibctx = शून्यसूचकः;
    ध्रुव वर्ण * matched_dev = शून्यसूचकः;
    पूर्णाङ्क gid_idx = gid_env ? atoi(gid_env) : -1;
    पूर्णाङ्क gid_संस्करणम् = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    प्रति (पूर्णाङ्क d = 0; d < अङ्कः_devs; d++) {
        ध्रुव वर्ण * dn = ibv_get_device_name(devs[d]);
        यदि (dev_env && वाक्यं_तुलय(dev_env, dn) != 0) अनुवर्त;

        ibv_context * ctx = ibv_open_device(devs[d]);
        यदि (!ctx) अनुवर्त;

        ibv_पत्तनम्_attr pa;
        यदि (ibv_query_पत्तनम्(ctx, ib_पत्तनम्, &pa) != 0) { ibv_close_device(ctx); अनुवर्त; }

        पूर्णाङ्क प्राप्तम्_gid = gid_idx;
        पूर्णाङ्क प्राप्तम्_संस्करणम् = IBV_GID_TYPE_IB;
        यदि (प्राप्तम्_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            पूर्णाङ्क v2_idx = -1;
            पूर्णाङ्क v1_idx = -1;
            प्रति (पूर्णाङ्क i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_प्रविष्टिः प्रविष्टिः = {};
                यदि (ibv_query_gid_ex(ctx, ib_पत्तनम्, i, &प्रविष्टिः, 0) != 0) अनुवर्त;
                यदि (memcmp(प्रविष्टिः.gid.raw, target_gid->दत्त(), RDMA_GID_आकार) != 0) अनुवर्त;
                यदि (प्रविष्टिः.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } अन्यथा यदि (प्रविष्टिः.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            यदि (v2_idx >= 0) {
                प्राप्तम्_gid = v2_idx;
                प्राप्तम्_संस्करणम् = IBV_GID_TYPE_ROCE_V2;
            } अन्यथा यदि (v1_idx >= 0) {
                प्राप्तम्_gid = v1_idx;
                प्राप्तम्_संस्करणम् = IBV_GID_TYPE_ROCE_V1;
            }
        } अन्यथा {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_प्रविष्टिः प्रविष्टिः = {};
            यदि (ibv_query_gid_ex(ctx, ib_पत्तनम्, प्राप्तम्_gid, &प्रविष्टिः, 0) == 0) {
                प्राप्तम्_संस्करणम् = प्रविष्टिः.gid_type;
            }
        }
        यदि (प्राप्तम्_gid >= 0) {
            ibctx = ctx;
            gid_idx = प्राप्तम्_gid;
            gid_संस्करणम् = प्राप्तम्_संस्करणम्;
            matched_dev = dn;
            rdma_local.मार्गः_mtu = pa.सक्रिय_mtu;
            भङ्ग;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    यदि (!ibctx) प्रत्यागम असत्य;

    rdma_local.ib_पत्तनम् = ib_पत्तनम्;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    यदि (!rdma->pd) प्रत्यागम असत्य;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, शून्यसूचकः, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_गभीरता + 4, शून्यसूचकः, rdma->ch, 0);
    यदि (!rdma->scq || !rdma->rcq) प्रत्यागम असत्य;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_गभीरता + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_दत्त = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    यदि (!rdma->qp) प्रत्यागम असत्य;
    rdma->max_inline = qia.cap.max_inline_दत्त;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_गभीरता) * RDMA_CHUNK);
    यदि (!rdma->tx_buf || !rdma->rx_buf) प्रत्यागम असत्य;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_गभीरता) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    यदि (!rdma->tx_mr || !rdma->rx_mr) प्रत्यागम असत्य;

    ibv_gid local_gid;
    यदि (ibv_query_gid(ibctx, ib_पत्तनम्, gid_idx, &local_gid) != 0) प्रत्यागम असत्य;

    rdma_local.qpn = rdma->qp->qp_अङ्कः;
    rdma_local.psn = rdma->qp->qp_अङ्कः & 0xffffff;
    स्मृतिं_प्रतिलिख(&rdma_local.gid, &local_gid, RDMA_GID_आकार);

    ध्रुव वर्ण * ver_str = "";
    यदि (gid_संस्करणम् == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } अन्यथा यदि (gid_संस्करणम् == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    प्रत्यागम सत्य;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
तार्किक socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ध्रुव uint8_t * remote_gid) {
    // RESET -> INIT
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.पत्तनम्_अङ्कः        = rdma_local.ib_पत्तनम्;
        a.pkey_सूचकांक      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        यदि (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_सूचकांक | IBV_QP_पत्तनम् | IBV_QP_ACCESS_FLAGS) != 0) {
            प्रत्यागम असत्य;
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < RDMA_RX_गभीरता; i++) {
        यदि (!rdma->post_rx(i)) प्रत्यागम असत्य;
    }

    // INIT -> RTR
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.मार्गः_mtu           = rdma_local.मार्गः_mtu;
        a.dest_qp_अङ्कः        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        स्मृतिं_प्रतिलिख(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_आकार);
        a.ah_attr.grh.hop_सीमा  = 1;
        a.ah_attr.grh.sgid_सूचकांक = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.पत्तनम्_अङ्कः   = rdma_local.ib_पत्तनम्;
        यदि (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_मार्गः_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            प्रत्यागम असत्य;
        }
    }

    // RTR -> RTS
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        यदि (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            प्रत्यागम असत्य;
        }
    }

    rdma->चरमः_सक्रिय = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.मार्गः_mtu, RDMA_RX_गभीरता);
    प्रत्यागम सत्य;
}

// Sleep until the completion channel has an event or the TCP peer closes.
तार्किक socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    संरचना pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    यदि (poll(pfds, 2, -1) < 0) {
        प्रत्यागम errno == EINTR;
    }
    यदि (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        प्रत्यागम असत्य;
    }
    यदि (pfds[0].revents & POLLIN) {
        संरचना ibv_cq * ev_cq  = शून्यसूचकः;
        शून्य          * ev_ctx = शून्यसूचकः;
        यदि (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            प्रत्यागम असत्य;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    प्रत्यागम सत्य;
}

तार्किक socket_t::impl::rdma_poll(संरचना ibv_cq * cq, संरचना ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    तार्किक armed = असत्य;
    प्रति (uint64_t s = 1; ; s++) {
        पूर्णाङ्क n = ibv_poll_cq(cq, 1, wc);
        यदि (n > 0) {
            c->चरमः_सक्रिय = std::chrono::steady_clock::now();
            यदि (wc->स्थितिः != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->स्थितिः, ibv_wc_स्थितिः_str(wc->स्थितिः), wc->vendor_err);
            }
            प्रत्यागम wc->स्थितिः == IBV_WC_SUCCESS;
        }
        यदि (n < 0) प्रत्यागम असत्य;
        यदि (armed) {
            // armed and still empty: sleep until the next completion
            यदि (!rdma_wait_event()) {
                प्रत्यागम असत्य;
            }
            armed = असत्य;
            अनुवर्त;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        यदि (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->चरमः_सक्रिय > RDMA_SPIN_TIME) {
            यदि (ibv_req_notify_cq(cq, 0) != 0) {
                प्रत्यागम असत्य;
            }
            armed = सत्य;
            अनुवर्त;
        }
        यदि ((s & 0xFFFFF) == 0) {
            यदि (tcp_peer_closed()) {
                प्रत्यागम असत्य;
            }
        }
    }
}

तार्किक socket_t::impl::rdma_send(ध्रुव शून्य * दत्त, size_t size) {
    rdma_conn * c = rdma.get();
    ध्रुव uint8_t * src = (ध्रुव uint8_t *)दत्त;
    size_t rem = size;
    यावत् (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        संरचना ibv_sge sge = {};
        संरचना ibv_send_wr wr = {}, * bad = शून्यसूचकः;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.अङ्कः_sge = 1;

        यदि (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.दीर्घता = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } अन्यथा {
            स्मृतिं_प्रतिलिख(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.दीर्घता = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        यदि (ibv_post_send(c->qp, &wr, &bad) != 0) प्रत्यागम असत्य;
        c->चरमः_सक्रिय = std::chrono::steady_clock::now();
        संरचना ibv_wc wc;
        यदि (!rdma_poll(c->scq, &wc)) प्रत्यागम असत्य;

        src += chunk;
        rem -= chunk;
    }
    प्रत्यागम सत्य;
}

तार्किक socket_t::impl::rdma_recv(शून्य * दत्त, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)दत्त;
    size_t rem = size;
    यावत् (rem > 0) {
        संरचना ibv_wc wc;
        यदि (!rdma_poll(c->rcq, &wc)) प्रत्यागम असत्य;

        पूर्णाङ्क slot = (पूर्णाङ्क)wc.wr_id;
        size_t got = wc.byte_len;
        स्मृतिं_प्रतिलिख(dst, c->rx_slot(slot), got);

        यदि (!c->post_rx(slot)) प्रत्यागम असत्य;

        dst += got;
        rem -= got;
    }
    प्रत्यागम सत्य;
}

#यदि_अन्तः // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#यदि_अन्तः // GGML_RPC_RDMA

तार्किक socket_t::impl::send_दत्त(ध्रुव शून्य * दत्त, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_आकार(size).c_str());
#यदि_परिभाषितम् GGML_RPC_RDMA_APPLE
    यदि (use_rdma) {
        प्रत्यागम rdma->send(दत्त, size);
    }
#नोचेद्यदि defined(GGML_RPC_RDMA)
    यदि (use_rdma) {
        प्रत्यागम rdma_send(दत्त, size);
    }
#यदि_अन्तः
    size_t अष्टकानि_प्रेषितम् = 0;
    यावत् (अष्टकानि_प्रेषितम् < size) {
        size_t आकार_गन्तव्यम्_send = std::min(size - अष्टकानि_प्रेषितम्, MAX_CHUNK_आकार);
        ssize_t n = send(fd, (ध्रुव वर्ण *)दत्त + अष्टकानि_प्रेषितम्, आकार_गन्तव्यम्_send, 0);
        यदि (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           अष्टकानि_प्रेषितम्, आकार_गन्तव्यम्_send);
            प्रत्यागम असत्य;
        }
        अष्टकानि_प्रेषितम् += (size_t)n;
    }
    प्रत्यागम सत्य;
}

तार्किक socket_t::impl::recv_दत्त(शून्य * दत्त, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_आकार(size).c_str());
#यदि_परिभाषितम् GGML_RPC_RDMA_APPLE
    यदि (use_rdma) {
        प्रत्यागम rdma->recv(दत्त, size);
    }
#नोचेद्यदि defined(GGML_RPC_RDMA)
    यदि (use_rdma) {
        प्रत्यागम rdma_recv(दत्त, size);
    }
#यदि_अन्तः
    size_t अष्टकानि_recv = 0;
    यावत् (अष्टकानि_recv < size) {
        size_t आकार_गन्तव्यम्_recv = std::min(size - अष्टकानि_recv, MAX_CHUNK_आकार);
        ssize_t n = recv(fd, (वर्ण *)दत्त + अष्टकानि_recv, आकार_गन्तव्यम्_recv, 0);
        यदि (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           अष्टकानि_recv, आकार_गन्तव्यम्_recv);
            प्रत्यागम असत्य;
        }
        यदि (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            प्रत्यागम असत्य;
        }
        अष्टकानि_recv += (size_t)n;
    }
    प्रत्यागम सत्य;
}

शून्य socket_t::impl::get_caps(uint8_t * local_caps) {
    स्मृतिं_पूरय(local_caps, 0, RPC_CONN_CAPS_आकार);
#यदि_परिभाषितम् GGML_RPC_RDMA
    यदि (std::getenv("GGML_RPC_NO_RDMA")) {
        प्रत्यागम;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    स्वचालित target_gid = rdma_build_target_gid();
    यदि (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->दत्त(), local_caps);
    }
    यदि (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  अन्यथा
    rdma_local = {};
    यदि (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        स्मृतिं_प्रतिलिख(rc.gid, rdma_local.gid, RDMA_GID_आकार);
        स्मृतिं_प्रतिलिख(local_caps, &rc, परिमाण(rc));
    } अन्यथा {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#यदि_अन्तः // GGML_RPC_RDMA
}

शून्य socket_t::impl::update_caps(ध्रुव uint8_t * remote_caps) {
#यदि_परिभाषितम् GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    तार्किक remote_rdma = असत्य;
    प्रति (size_t i = 0; i < RPC_CONN_CAPS_आकार; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    यदि (!rdma || !remote_rdma) {
        यदि (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        प्रत्यागम;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    तार्किक activated = rdma->activate(remote_caps);
#  अन्यथा
    rdma_caps rc = {};
    स्मृतिं_प्रतिलिख(&rc, remote_caps, परिमाण(rc));
    तार्किक activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    यदि (activated) {
        use_rdma = सत्य;
    } अन्यथा {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#अन्यथा
    (शून्य)remote_caps;
#यदि_अन्तः // GGML_RPC_RDMA
}

तार्किक socket_t::impl::flush() {
#यदि_परिभाषितम् GGML_RPC_RDMA_APPLE
    यदि (use_rdma) {
        प्रत्यागम rdma->flush();
    }
#यदि_अन्तः
    प्रत्यागम सत्य;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = स्वभाव;

तार्किक socket_t::send_दत्त(ध्रुव शून्य * दत्त, size_t size) {
    प्रत्यागम pimpl->send_दत्त(दत्त, size);
}

तार्किक socket_t::recv_दत्त(शून्य * दत्त, size_t size) {
    प्रत्यागम pimpl->recv_दत्त(दत्त, size);
}

तार्किक socket_t::flush() {
    प्रत्यागम pimpl->flush();
}

शून्य socket_t::get_caps(uint8_t * local_caps) {
    प्रत्यागम pimpl->get_caps(local_caps);
}

शून्य socket_t::update_caps(ध्रुव uint8_t * remote_caps) {
    प्रत्यागम pimpl->update_caps(remote_caps);
}

ध्रुव वर्ण * socket_t::transport_name() ध्रुव {
    प्रत्यागम pimpl->use_rdma ? "RDMA" : "TCP";
}

स्थिर तार्किक is_valid_fd(sockfd_t sockfd) {
#यदि_परिभाषितम् _WIN32
    प्रत्यागम sockfd != INVALID_SOCKET;
#अन्यथा
    प्रत्यागम sockfd >= 0;
#यदि_अन्तः
}

स्थिर तार्किक समुच्चयः_no_delay(sockfd_t sockfd) {
    पूर्णाङ्क flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    पूर्णाङ्क ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (वर्ण *)&flag, परिमाण(पूर्णाङ्क));
    प्रत्यागम ret == 0;
}

स्थिर तार्किक समुच्चयः_reuse_addr(sockfd_t sockfd) {
    पूर्णाङ्क flag = 1;
    पूर्णाङ्क ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (वर्ण *)&flag, परिमाण(पूर्णाङ्क));
    प्रत्यागम ret == 0;
}

socket_ptr socket_t::accept() {
    स्वचालित client_socket_fd = ::accept(pimpl->fd, रिक्त, रिक्त);
    यदि (!is_valid_fd(client_socket_fd)) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (!समुच्चयः_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        प्रत्यागम शून्यसूचकः;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    प्रत्यागम socket_ptr(नूतन socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ध्रुव वर्ण * host, पूर्णाङ्क पत्तनम्) {
    स्वचालित sockfd = socket(AF_INET, SOCK_प्रवाहः, 0);
    यदि (!is_valid_fd(sockfd)) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (!समुच्चयः_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        प्रत्यागम शून्यसूचकः;
    }
    यदि (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        प्रत्यागम शून्यसूचकः;
    }
    संरचना sockaddr_अन्तर् serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_पत्तनम् = htons(पत्तनम्);

    यदि (bind(sockfd, (संरचना sockaddr *) &serv_addr, परिमाण(serv_addr)) < 0) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (listen(sockfd, 1) < 0) {
        प्रत्यागम शून्यसूचकः;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, पत्तनम्);
    प्रत्यागम socket_ptr(नूतन socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ध्रुव वर्ण * host, पूर्णाङ्क पत्तनम्) {
    स्वचालित sockfd = socket(AF_INET, SOCK_प्रवाहः, 0);
    यदि (!is_valid_fd(sockfd)) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (!समुच्चयः_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        प्रत्यागम शून्यसूचकः;
    }
    संरचना sockaddr_अन्तर् addr;
    addr.sin_family = AF_INET;
    addr.sin_पत्तनम् = htons(पत्तनम्);
    संरचना hostent * server = gethostbyname(host);
    यदि (server == रिक्त) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        प्रत्यागम शून्यसूचकः;
    }
    स्मृतिं_प्रतिलिख(&addr.sin_addr.s_addr, server->h_addr, server->h_दीर्घता);
    यदि (::connect(sockfd, (संरचना sockaddr *)&addr, परिमाण(addr)) < 0) {
        प्रत्यागम शून्यसूचकः;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, पत्तनम्);
    प्रत्यागम socket_ptr(नूतन socket_t(std::make_unique<impl>(sockfd)));
}

#यदि_परिभाषितम् _WIN32
स्थिर std::mutex g_rpc_transport_mu;
स्थिर तार्किक g_rpc_transport_wsa_started = असत्य;
#यदि_अन्तः

तार्किक rpc_transport_init() {
#यदि_परिभाषितम् _WIN32
    std::lock_guard<std::mutex> तालकम्(g_rpc_transport_mu);
    यदि (g_rpc_transport_wsa_started) {
        प्रत्यागम सत्य;
    }
    WSADATA wsa_दत्त;
    पूर्णाङ्क res = WSAStartup(MAKEWORD(2, 2), &wsa_दत्त);
    यदि (res != 0) {
        प्रत्यागम असत्य;
    }
    g_rpc_transport_wsa_started = सत्य;
    प्रत्यागम सत्य;
#अन्यथा
    प्रत्यागम सत्य;
#यदि_अन्तः
}

शून्य rpc_transport_shutdown() {
#यदि_परिभाषितम् _WIN32
    std::lock_guard<std::mutex> तालकम्(g_rpc_transport_mu);
    यदि (!g_rpc_transport_wsa_started) {
        प्रत्यागम;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = असत्य;
#यदि_अन्तः
}
