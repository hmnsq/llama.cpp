#ضمن "transport.h"
#ضمن "log.h"

#جيڪڏهن_وصف_ٿيل _WIN32
#  define WIN32_LEAN_AND_سراسري
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#وإلا
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#جيڪڏهن_پورو
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <mutex>
#ضمن <optional>

#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
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
#جيڪڏهن_پورو // GGML_RPC_RDMA

#جيڪڏهن_وصف_ٿيل _WIN32
تعريفنوع SOCKET sockfd_t;
استعمال ssize_t = __int64;
#وإلا
تعريفنوع صحيح sockfd_t;
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
ساكن constexpr size_t RDMA_GID_سائيز = 16;            // RoCE GID / IB GID is always 16 bytes
استعمال rdma_gid_t = std::array<uint8_t, RDMA_GID_سائيز>;
#جيڪڏهن_پورو // GGML_RPC_RDMA

#إذا defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ساكن constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ساكن constexpr صحيح    RDMA_RX_اونهائي = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ساكن constexpr خودڪار   RDMA_SPIN_وقت = std::chrono::milliseconds(100);

بنية rdma_conn {
    بنية ibv_context * ctx = خالي_اشارو;
    بنية ibv_pd * pd  = خالي_اشارو;
    بنية ibv_cq * scq = خالي_اشارو;   // send completions
    بنية ibv_cq * rcq = خالي_اشارو;   // recv completions
    بنية ibv_qp * qp  = خالي_اشارو;
    بنية ibv_comp_channel * ch = خالي_اشارو; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point آخري_سرگرم; // last completion or posted send

    فراغ          * tx_buf = خالي_اشارو;
    بنية ibv_mr * tx_mr  = خالي_اشارو;

    فراغ          * rx_buf = خالي_اشارو; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    بنية ibv_mr * rx_mr  = خالي_اشارو;
    صحيح             rx_سر = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_جڳهه(صحيح i) ثابت {
        أرجع static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    منطقي post_rx(صحيح i) {
        بنية ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_جڳهه(i);
        sge.ڊيگهائي = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        بنية ibv_recv_wr wr = {}, * bad = خالي_اشارو;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.انگ_sge = 1;
        أرجع ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        إذا (tx_mr) ibv_dereg_mr(tx_mr);
        إذا (rx_mr) ibv_dereg_mr(rx_mr);
        حرر(tx_buf);
        حرر(rx_buf);
        إذا (qp)  ibv_destroy_qp(qp);
        إذا (scq) ibv_destroy_cq(scq);
        إذا (rcq) ibv_destroy_cq(rcq);
        إذا (ch)  ibv_destroy_comp_channel(ch);
        إذا (pd)  ibv_dealloc_pd(pd);
        إذا (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
بنية rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_سائيز] = {};
    uint8_t  ib_پورٽ = 0;
    صحيح      gid_idx = 0;
    تعداد ibv_mtu رستو_mtu = IBV_MTU_1024;
};

بنية rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_سائيز];
};

static_assert(حجم(rdma_caps) == RPC_CONN_CAPS_سائيز, "rdma_caps must match conn_caps size");

#جيڪڏهن_پورو // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

بنية socket_t::impl {
    impl(sockfd_t fd) : use_rdma(خطأ), fd(fd) {}
    ~impl();
    منطقي send_معلومات(ثابت فراغ * معلومات, size_t size);
    منطقي recv_معلومات(فراغ * معلومات, size_t size);
    منطقي flush();
    فراغ get_caps(uint8_t * local_caps);
    فراغ update_caps(ثابت uint8_t * remote_caps);

#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_هدف_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  وإلا
    منطقي rdma_probe();
    منطقي rdma_send(ثابت فراغ * معلومات, size_t size);
    منطقي rdma_recv(فراغ * معلومات, size_t size);
    منطقي tcp_peer_closed();
    منطقي rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid);
    منطقي rdma_poll(بنية ibv_cq * cq, بنية ibv_wc * wc);
    منطقي rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#جيڪڏهن_پورو // GGML_RPC_RDMA
    منطقي     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
    rdma.reset();
#جيڪڏهن_پورو // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, هذا->fd);
#جيڪڏهن_وصف_ٿيل _WIN32
    إذا (fd != INVALID_SOCKET) closesocket(هذا->fd);
#وإلا
    إذا (fd >= 0) close(هذا->fd);
#جيڪڏهن_پورو
}

#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_هدف_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = حجم(addr);
    إذا (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        أرجع std::nullopt;
    }
    rdma_gid_t هدف = {};
    إذا (addr.ss_family == AF_INET) {
        ثابت خودڪار * a = reinterpret_cast<ثابت sockaddr_اندريون *>(&addr);
        هدف[10] = 0xff;
        هدف[11] = 0xff;
        يادگيري_نقل(&هدف[12], &a->sin_addr, 4);
        أرجع هدف;
    }
    إذا (addr.ss_family == AF_INET6) {
        ثابت خودڪار * a = reinterpret_cast<ثابت sockaddr_in6 *>(&addr);
        يادگيري_نقل(هدف.معلومات(), &a->sin6_addr, RDMA_GID_سائيز);
        أرجع هدف;
    }
    أرجع std::nullopt;
}

#جيڪڏهن_وصف_نه_ٿيل GGML_RPC_RDMA_APPLE

منطقي socket_t::impl::tcp_peer_closed() {
    إذا (fd < 0) أرجع خطأ;
#جيڪڏهن_وصف_نه_ٿيل _WIN32
    بنية pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    صحيح r = poll(&pfd, 1, 0);
    أرجع r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#وإلا
    أرجع خطأ;
#جيڪڏهن_پورو
}

منطقي socket_t::impl::rdma_probe() {
    ثابت حرف * dev_env = std::getenv("GGML_RDMA_DEV");
    ثابت حرف * gid_env = std::getenv("GGML_RDMA_GID");

    خودڪار هدف_gid = rdma_build_هدف_gid();
    إذا (!هدف_gid) {
        أرجع خطأ;
    }

    ثابت uint8_t ib_پورٽ = 1;
    صحيح انگ_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&انگ_devs);
    إذا (!devs || انگ_devs == 0) أرجع خطأ;

    ibv_context * ibctx = خالي_اشارو;
    ثابت حرف * matched_dev = خالي_اشارو;
    صحيح gid_idx = gid_env ? atoi(gid_env) : -1;
    صحيح gid_نسخو = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    لكل (صحيح d = 0; d < انگ_devs; d++) {
        ثابت حرف * dn = ibv_get_device_name(devs[d]);
        إذا (dev_env && strcmp(dev_env, dn) != 0) استمر;

        ibv_context * ctx = ibv_open_device(devs[d]);
        إذا (!ctx) استمر;

        ibv_پورٽ_attr pa;
        إذا (ibv_query_پورٽ(ctx, ib_پورٽ, &pa) != 0) { ibv_close_device(ctx); استمر; }

        صحيح مليو_gid = gid_idx;
        صحيح مليو_نسخو = IBV_GID_TYPE_IB;
        إذا (مليو_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            صحيح v2_idx = -1;
            صحيح v1_idx = -1;
            لكل (صحيح i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_داخلا داخلا = {};
                إذا (ibv_query_gid_ex(ctx, ib_پورٽ, i, &داخلا, 0) != 0) استمر;
                إذا (memcmp(داخلا.gid.raw, هدف_gid->معلومات(), RDMA_GID_سائيز) != 0) استمر;
                إذا (داخلا.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } وإلا إذا (داخلا.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            إذا (v2_idx >= 0) {
                مليو_gid = v2_idx;
                مليو_نسخو = IBV_GID_TYPE_ROCE_V2;
            } وإلا إذا (v1_idx >= 0) {
                مليو_gid = v1_idx;
                مليو_نسخو = IBV_GID_TYPE_ROCE_V1;
            }
        } وإلا {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_داخلا داخلا = {};
            إذا (ibv_query_gid_ex(ctx, ib_پورٽ, مليو_gid, &داخلا, 0) == 0) {
                مليو_نسخو = داخلا.gid_type;
            }
        }
        إذا (مليو_gid >= 0) {
            ibctx = ctx;
            gid_idx = مليو_gid;
            gid_نسخو = مليو_نسخو;
            matched_dev = dn;
            rdma_local.رستو_mtu = pa.سرگرم_mtu;
            اكسر;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    إذا (!ibctx) أرجع خطأ;

    rdma_local.ib_پورٽ = ib_پورٽ;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    إذا (!rdma->pd) أرجع خطأ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, خالي_اشارو, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_اونهائي + 4, خالي_اشارو, rdma->ch, 0);
    إذا (!rdma->scq || !rdma->rcq) أرجع خطأ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_اونهائي + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_معلومات = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    إذا (!rdma->qp) أرجع خطأ;
    rdma->max_inline = qia.cap.max_inline_معلومات;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_اونهائي) * RDMA_CHUNK);
    إذا (!rdma->tx_buf || !rdma->rx_buf) أرجع خطأ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_اونهائي) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    إذا (!rdma->tx_mr || !rdma->rx_mr) أرجع خطأ;

    ibv_gid local_gid;
    إذا (ibv_query_gid(ibctx, ib_پورٽ, gid_idx, &local_gid) != 0) أرجع خطأ;

    rdma_local.qpn = rdma->qp->qp_انگ;
    rdma_local.psn = rdma->qp->qp_انگ & 0xffffff;
    يادگيري_نقل(&rdma_local.gid, &local_gid, RDMA_GID_سائيز);

    ثابت حرف * ver_str = "";
    إذا (gid_نسخو == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } وإلا إذا (gid_نسخو == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    أرجع سچ;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
منطقي socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid) {
    // RESET -> INIT
    {
        بنية ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.پورٽ_انگ        = rdma_local.ib_پورٽ;
        a.pkey_اشاريو      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        إذا (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_اشاريو | IBV_QP_پورٽ | IBV_QP_ACCESS_FLAGS) != 0) {
            أرجع خطأ;
        }
    }

    لكل (صحيح i = 0; i < RDMA_RX_اونهائي; i++) {
        إذا (!rdma->post_rx(i)) أرجع خطأ;
    }

    // INIT -> RTR
    {
        بنية ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.رستو_mtu           = rdma_local.رستو_mtu;
        a.dest_qp_انگ        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        يادگيري_نقل(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_سائيز);
        a.ah_attr.grh.hop_حد_بندي  = 1;
        a.ah_attr.grh.sgid_اشاريو = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.پورٽ_انگ   = rdma_local.ib_پورٽ;
        إذا (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_رستو_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            أرجع خطأ;
        }
    }

    // RTR -> RTS
    {
        بنية ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        إذا (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            أرجع خطأ;
        }
    }

    rdma->آخري_سرگرم = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.رستو_mtu, RDMA_RX_اونهائي);
    أرجع سچ;
}

// Sleep until the completion channel has an event or the TCP peer closes.
منطقي socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    بنية pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    إذا (poll(pfds, 2, -1) < 0) {
        أرجع errno == EINTR;
    }
    إذا (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        أرجع خطأ;
    }
    إذا (pfds[0].revents & POLLIN) {
        بنية ibv_cq * ev_cq  = خالي_اشارو;
        فراغ          * ev_ctx = خالي_اشارو;
        إذا (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            أرجع خطأ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    أرجع سچ;
}

منطقي socket_t::impl::rdma_poll(بنية ibv_cq * cq, بنية ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    منطقي armed = خطأ;
    لكل (uint64_t s = 1; ; s++) {
        صحيح n = ibv_poll_cq(cq, 1, wc);
        إذا (n > 0) {
            c->آخري_سرگرم = std::chrono::steady_clock::now();
            إذا (wc->حالت != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->حالت, ibv_wc_حالت_str(wc->حالت), wc->vendor_err);
            }
            أرجع wc->حالت == IBV_WC_SUCCESS;
        }
        إذا (n < 0) أرجع خطأ;
        إذا (armed) {
            // armed and still empty: sleep until the next completion
            إذا (!rdma_wait_event()) {
                أرجع خطأ;
            }
            armed = خطأ;
            استمر;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        إذا (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->آخري_سرگرم > RDMA_SPIN_وقت) {
            إذا (ibv_req_notify_cq(cq, 0) != 0) {
                أرجع خطأ;
            }
            armed = سچ;
            استمر;
        }
        إذا ((s & 0xFFFFF) == 0) {
            إذا (tcp_peer_closed()) {
                أرجع خطأ;
            }
        }
    }
}

منطقي socket_t::impl::rdma_send(ثابت فراغ * معلومات, size_t size) {
    rdma_conn * c = rdma.get();
    ثابت uint8_t * src = (ثابت uint8_t *)معلومات;
    size_t rem = size;
    طالما (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        بنية ibv_sge sge = {};
        بنية ibv_send_wr wr = {}, * bad = خالي_اشارو;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.انگ_sge = 1;

        إذا (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.ڊيگهائي = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } وإلا {
            يادگيري_نقل(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.ڊيگهائي = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        إذا (ibv_post_send(c->qp, &wr, &bad) != 0) أرجع خطأ;
        c->آخري_سرگرم = std::chrono::steady_clock::now();
        بنية ibv_wc wc;
        إذا (!rdma_poll(c->scq, &wc)) أرجع خطأ;

        src += chunk;
        rem -= chunk;
    }
    أرجع سچ;
}

منطقي socket_t::impl::rdma_recv(فراغ * معلومات, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)معلومات;
    size_t rem = size;
    طالما (rem > 0) {
        بنية ibv_wc wc;
        إذا (!rdma_poll(c->rcq, &wc)) أرجع خطأ;

        صحيح جڳهه = (صحيح)wc.wr_id;
        size_t got = wc.byte_len;
        يادگيري_نقل(dst, c->rx_جڳهه(جڳهه), got);

        إذا (!c->post_rx(جڳهه)) أرجع خطأ;

        dst += got;
        rem -= got;
    }
    أرجع سچ;
}

#جيڪڏهن_پورو // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#جيڪڏهن_پورو // GGML_RPC_RDMA

منطقي socket_t::impl::send_معلومات(ثابت فراغ * معلومات, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_سائيز(size).c_str());
#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA_APPLE
    إذا (use_rdma) {
        أرجع rdma->send(معلومات, size);
    }
#وإلاإذا defined(GGML_RPC_RDMA)
    إذا (use_rdma) {
        أرجع rdma_send(معلومات, size);
    }
#جيڪڏهن_پورو
    size_t بائيٽون_موڪليل = 0;
    طالما (بائيٽون_موڪليل < size) {
        size_t سائيز_منزل_send = std::min(size - بائيٽون_موڪليل, MAX_CHUNK_سائيز);
        ssize_t n = send(fd, (ثابت حرف *)معلومات + بائيٽون_موڪليل, سائيز_منزل_send, 0);
        إذا (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           بائيٽون_موڪليل, سائيز_منزل_send);
            أرجع خطأ;
        }
        بائيٽون_موڪليل += (size_t)n;
    }
    أرجع سچ;
}

منطقي socket_t::impl::recv_معلومات(فراغ * معلومات, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_سائيز(size).c_str());
#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA_APPLE
    إذا (use_rdma) {
        أرجع rdma->recv(معلومات, size);
    }
#وإلاإذا defined(GGML_RPC_RDMA)
    إذا (use_rdma) {
        أرجع rdma_recv(معلومات, size);
    }
#جيڪڏهن_پورو
    size_t بائيٽون_recv = 0;
    طالما (بائيٽون_recv < size) {
        size_t سائيز_منزل_recv = std::min(size - بائيٽون_recv, MAX_CHUNK_سائيز);
        ssize_t n = recv(fd, (حرف *)معلومات + بائيٽون_recv, سائيز_منزل_recv, 0);
        إذا (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           بائيٽون_recv, سائيز_منزل_recv);
            أرجع خطأ;
        }
        إذا (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            أرجع خطأ;
        }
        بائيٽون_recv += (size_t)n;
    }
    أرجع سچ;
}

فراغ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_سائيز);
#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
    إذا (std::getenv("GGML_RPC_NO_RDMA")) {
        أرجع;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    خودڪار هدف_gid = rdma_build_هدف_gid();
    إذا (هدف_gid) {
        rdma = apple_rdma::probe(fd, هدف_gid->معلومات(), local_caps);
    }
    إذا (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  وإلا
    rdma_local = {};
    إذا (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        يادگيري_نقل(rc.gid, rdma_local.gid, RDMA_GID_سائيز);
        يادگيري_نقل(local_caps, &rc, حجم(rc));
    } وإلا {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#جيڪڏهن_پورو // GGML_RPC_RDMA
}

فراغ socket_t::impl::update_caps(ثابت uint8_t * remote_caps) {
#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    منطقي remote_rdma = خطأ;
    لكل (size_t i = 0; i < RPC_CONN_CAPS_سائيز; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    إذا (!rdma || !remote_rdma) {
        إذا (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        أرجع;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    منطقي activated = rdma->activate(remote_caps);
#  وإلا
    rdma_caps rc = {};
    يادگيري_نقل(&rc, remote_caps, حجم(rc));
    منطقي activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    إذا (activated) {
        use_rdma = سچ;
    } وإلا {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#وإلا
    (فراغ)remote_caps;
#جيڪڏهن_پورو // GGML_RPC_RDMA
}

منطقي socket_t::impl::flush() {
#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA_APPLE
    إذا (use_rdma) {
        أرجع rdma->flush();
    }
#جيڪڏهن_پورو
    أرجع سچ;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = افتراضي;

منطقي socket_t::send_معلومات(ثابت فراغ * معلومات, size_t size) {
    أرجع pimpl->send_معلومات(معلومات, size);
}

منطقي socket_t::recv_معلومات(فراغ * معلومات, size_t size) {
    أرجع pimpl->recv_معلومات(معلومات, size);
}

منطقي socket_t::flush() {
    أرجع pimpl->flush();
}

فراغ socket_t::get_caps(uint8_t * local_caps) {
    أرجع pimpl->get_caps(local_caps);
}

فراغ socket_t::update_caps(ثابت uint8_t * remote_caps) {
    أرجع pimpl->update_caps(remote_caps);
}

ثابت حرف * socket_t::transport_name() ثابت {
    أرجع pimpl->use_rdma ? "RDMA" : "TCP";
}

ساكن منطقي is_valid_fd(sockfd_t sockfd) {
#جيڪڏهن_وصف_ٿيل _WIN32
    أرجع sockfd != INVALID_SOCKET;
#وإلا
    أرجع sockfd >= 0;
#جيڪڏهن_پورو
}

ساكن منطقي سيٽ_no_delay(sockfd_t sockfd) {
    صحيح flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    صحيح ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (حرف *)&flag, حجم(صحيح));
    أرجع ret == 0;
}

ساكن منطقي سيٽ_reuse_addr(sockfd_t sockfd) {
    صحيح flag = 1;
    صحيح ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (حرف *)&flag, حجم(صحيح));
    أرجع ret == 0;
}

socket_ptr socket_t::accept() {
    خودڪار client_socket_fd = ::accept(pimpl->fd, عدم, عدم);
    إذا (!is_valid_fd(client_socket_fd)) {
        أرجع خالي_اشارو;
    }
    إذا (!سيٽ_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        أرجع خالي_اشارو;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ثابت حرف * host, صحيح پورٽ) {
    خودڪار sockfd = socket(AF_INET, SOCK_وهڪرو, 0);
    إذا (!is_valid_fd(sockfd)) {
        أرجع خالي_اشارو;
    }
    إذا (!سيٽ_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        أرجع خالي_اشارو;
    }
    إذا (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        أرجع خالي_اشارو;
    }
    بنية sockaddr_اندريون serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_پورٽ = htons(پورٽ);

    إذا (bind(sockfd, (بنية sockaddr *) &serv_addr, حجم(serv_addr)) < 0) {
        أرجع خالي_اشارو;
    }
    إذا (listen(sockfd, 1) < 0) {
        أرجع خالي_اشارو;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, پورٽ);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ثابت حرف * host, صحيح پورٽ) {
    خودڪار sockfd = socket(AF_INET, SOCK_وهڪرو, 0);
    إذا (!is_valid_fd(sockfd)) {
        أرجع خالي_اشارو;
    }
    إذا (!سيٽ_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        أرجع خالي_اشارو;
    }
    بنية sockaddr_اندريون addr;
    addr.sin_family = AF_INET;
    addr.sin_پورٽ = htons(پورٽ);
    بنية hostent * server = gethostbyname(host);
    إذا (server == عدم) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        أرجع خالي_اشارو;
    }
    يادگيري_نقل(&addr.sin_addr.s_addr, server->h_addr, server->h_ڊيگهائي);
    إذا (::connect(sockfd, (بنية sockaddr *)&addr, حجم(addr)) < 0) {
        أرجع خالي_اشارو;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, پورٽ);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(sockfd)));
}

#جيڪڏهن_وصف_ٿيل _WIN32
ساكن std::mutex g_rpc_transport_mu;
ساكن منطقي g_rpc_transport_wsa_started = خطأ;
#جيڪڏهن_پورو

منطقي rpc_transport_init() {
#جيڪڏهن_وصف_ٿيل _WIN32
    std::lock_guard<std::mutex> تالو(g_rpc_transport_mu);
    إذا (g_rpc_transport_wsa_started) {
        أرجع سچ;
    }
    WSADATA wsa_معلومات;
    صحيح res = WSAStartup(MAKEWORD(2, 2), &wsa_معلومات);
    إذا (res != 0) {
        أرجع خطأ;
    }
    g_rpc_transport_wsa_started = سچ;
    أرجع سچ;
#وإلا
    أرجع سچ;
#جيڪڏهن_پورو
}

فراغ rpc_transport_shutdown() {
#جيڪڏهن_وصف_ٿيل _WIN32
    std::lock_guard<std::mutex> تالو(g_rpc_transport_mu);
    إذا (!g_rpc_transport_wsa_started) {
        أرجع;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = خطأ;
#جيڪڏهن_پورو
}
