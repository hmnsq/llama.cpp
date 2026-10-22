#شامل "transport.h"
#شامل "log.h"

#اگر_تعریف_شدہ _WIN32
#  define WIN32_LEAN_AND_اوسط
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#ورنہ
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#اگر_ختم
#شامل <cstdlib>
#شامل <cstring>
#شامل <mutex>
#شامل <optional>

#اگر_تعریف_شدہ GGML_RPC_RDMA
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
#اگر_ختم // GGML_RPC_RDMA

#اگر_تعریف_شدہ _WIN32
قسمتعریف SOCKET sockfd_t;
استعمال ssize_t = __int64;
#ورنہ
قسمتعریف صحیح sockfd_t;
#اگر_ختم

#اگر_تعریف_شدہ GGML_RPC_RDMA
ساکن constexpr size_t RDMA_GID_سائز = 16;            // RoCE GID / IB GID is always 16 bytes
استعمال rdma_gid_t = std::array<uint8_t, RDMA_GID_سائز>;
#اگر_ختم // GGML_RPC_RDMA

#اگر defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ساکن constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ساکن constexpr صحیح    RDMA_RX_گہرائی = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ساکن constexpr خودکار   RDMA_SPIN_وقت = std::chrono::milliseconds(100);

ساخت rdma_conn {
    ساخت ibv_context * ctx = خالی_اشاریہ;
    ساخت ibv_pd * pd  = خالی_اشاریہ;
    ساخت ibv_cq * scq = خالی_اشاریہ;   // send completions
    ساخت ibv_cq * rcq = خالی_اشاریہ;   // recv completions
    ساخت ibv_qp * qp  = خالی_اشاریہ;
    ساخت ibv_comp_channel * ch = خالی_اشاریہ; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point آخری_فعال; // last completion or posted send

    خالی          * tx_buf = خالی_اشاریہ;
    ساخت ibv_mr * tx_mr  = خالی_اشاریہ;

    خالی          * rx_buf = خالی_اشاریہ; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ساخت ibv_mr * rx_mr  = خالی_اشاریہ;
    صحیح             rx_سر = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_خانہ(صحیح i) ثابت {
        واپس static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    منطقی post_rx(صحیح i) {
        ساخت ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_خانہ(i);
        sge.لمبائی = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ساخت ibv_recv_wr wr = {}, * bad = خالی_اشاریہ;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.عدد_sge = 1;
        واپس ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        اگر (tx_mr) ibv_dereg_mr(tx_mr);
        اگر (rx_mr) ibv_dereg_mr(rx_mr);
        آزاد(tx_buf);
        آزاد(rx_buf);
        اگر (qp)  ibv_destroy_qp(qp);
        اگر (scq) ibv_destroy_cq(scq);
        اگر (rcq) ibv_destroy_cq(rcq);
        اگر (ch)  ibv_destroy_comp_channel(ch);
        اگر (pd)  ibv_dealloc_pd(pd);
        اگر (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ساخت rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_سائز] = {};
    uint8_t  ib_پورٹ = 0;
    صحیح      gid_idx = 0;
    شمار ibv_mtu راستہ_mtu = IBV_MTU_1024;
};

ساخت rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_سائز];
};

static_assert(حجم(rdma_caps) == RPC_CONN_CAPS_سائز, "rdma_caps must match conn_caps size");

#اگر_ختم // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ساخت socket_t::impl {
    impl(sockfd_t fd) : use_rdma(جھوٹ), fd(fd) {}
    ~impl();
    منطقی send_ڈیٹا(ثابت خالی * ڈیٹا, size_t size);
    منطقی recv_ڈیٹا(خالی * ڈیٹا, size_t size);
    منطقی flush();
    خالی get_caps(uint8_t * local_caps);
    خالی update_caps(ثابت uint8_t * remote_caps);

#اگر_تعریف_شدہ GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_ہدف_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  ورنہ
    منطقی rdma_probe();
    منطقی rdma_send(ثابت خالی * ڈیٹا, size_t size);
    منطقی rdma_recv(خالی * ڈیٹا, size_t size);
    منطقی tcp_peer_closed();
    منطقی rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid);
    منطقی rdma_poll(ساخت ibv_cq * cq, ساخت ibv_wc * wc);
    منطقی rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#اگر_ختم // GGML_RPC_RDMA
    منطقی     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#اگر_تعریف_شدہ GGML_RPC_RDMA
    rdma.reset();
#اگر_ختم // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, یہ->fd);
#اگر_تعریف_شدہ _WIN32
    اگر (fd != INVALID_SOCKET) closesocket(یہ->fd);
#ورنہ
    اگر (fd >= 0) close(یہ->fd);
#اگر_ختم
}

#اگر_تعریف_شدہ GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_ہدف_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = حجم(addr);
    اگر (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        واپس std::nullopt;
    }
    rdma_gid_t ہدف = {};
    اگر (addr.ss_family == AF_INET) {
        ثابت خودکار * a = reinterpret_cast<ثابت sockaddr_اندر *>(&addr);
        ہدف[10] = 0xff;
        ہدف[11] = 0xff;
        حافظہ_نقل(&ہدف[12], &a->sin_addr, 4);
        واپس ہدف;
    }
    اگر (addr.ss_family == AF_INET6) {
        ثابت خودکار * a = reinterpret_cast<ثابت sockaddr_in6 *>(&addr);
        حافظہ_نقل(ہدف.ڈیٹا(), &a->sin6_addr, RDMA_GID_سائز);
        واپس ہدف;
    }
    واپس std::nullopt;
}

#اگر_غیر_تعریف_شدہ GGML_RPC_RDMA_APPLE

منطقی socket_t::impl::tcp_peer_closed() {
    اگر (fd < 0) واپس جھوٹ;
#اگر_غیر_تعریف_شدہ _WIN32
    ساخت pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    صحیح r = poll(&pfd, 1, 0);
    واپس r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#ورنہ
    واپس جھوٹ;
#اگر_ختم
}

منطقی socket_t::impl::rdma_probe() {
    ثابت حرف * dev_env = std::getenv("GGML_RDMA_DEV");
    ثابت حرف * gid_env = std::getenv("GGML_RDMA_GID");

    خودکار ہدف_gid = rdma_build_ہدف_gid();
    اگر (!ہدف_gid) {
        واپس جھوٹ;
    }

    ثابت uint8_t ib_پورٹ = 1;
    صحیح عدد_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&عدد_devs);
    اگر (!devs || عدد_devs == 0) واپس جھوٹ;

    ibv_context * ibctx = خالی_اشاریہ;
    ثابت حرف * matched_dev = خالی_اشاریہ;
    صحیح gid_idx = gid_env ? atoi(gid_env) : -1;
    صحیح gid_نسخہ = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    برائے (صحیح d = 0; d < عدد_devs; d++) {
        ثابت حرف * dn = ibv_get_device_name(devs[d]);
        اگر (dev_env && strcmp(dev_env, dn) != 0) جاری;

        ibv_context * ctx = ibv_open_device(devs[d]);
        اگر (!ctx) جاری;

        ibv_پورٹ_attr pa;
        اگر (ibv_query_پورٹ(ctx, ib_پورٹ, &pa) != 0) { ibv_close_device(ctx); جاری; }

        صحیح ملا_gid = gid_idx;
        صحیح ملا_نسخہ = IBV_GID_TYPE_IB;
        اگر (ملا_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            صحیح v2_idx = -1;
            صحیح v1_idx = -1;
            برائے (صحیح i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_اندراج اندراج_2 = {};
                اگر (ibv_query_gid_ex(ctx, ib_پورٹ, i, &اندراج_2, 0) != 0) جاری;
                اگر (memcmp(اندراج_2.gid.raw, ہدف_gid->ڈیٹا(), RDMA_GID_سائز) != 0) جاری;
                اگر (اندراج_2.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } ورنہ اگر (اندراج_2.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            اگر (v2_idx >= 0) {
                ملا_gid = v2_idx;
                ملا_نسخہ = IBV_GID_TYPE_ROCE_V2;
            } ورنہ اگر (v1_idx >= 0) {
                ملا_gid = v1_idx;
                ملا_نسخہ = IBV_GID_TYPE_ROCE_V1;
            }
        } ورنہ {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_اندراج اندراج_2 = {};
            اگر (ibv_query_gid_ex(ctx, ib_پورٹ, ملا_gid, &اندراج_2, 0) == 0) {
                ملا_نسخہ = اندراج_2.gid_type;
            }
        }
        اگر (ملا_gid >= 0) {
            ibctx = ctx;
            gid_idx = ملا_gid;
            gid_نسخہ = ملا_نسخہ;
            matched_dev = dn;
            rdma_local.راستہ_mtu = pa.فعال_mtu;
            توڑو;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    اگر (!ibctx) واپس جھوٹ;

    rdma_local.ib_پورٹ = ib_پورٹ;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    اگر (!rdma->pd) واپس جھوٹ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, خالی_اشاریہ, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_گہرائی + 4, خالی_اشاریہ, rdma->ch, 0);
    اگر (!rdma->scq || !rdma->rcq) واپس جھوٹ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_گہرائی + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_ڈیٹا = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    اگر (!rdma->qp) واپس جھوٹ;
    rdma->max_inline = qia.cap.max_inline_ڈیٹا;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_گہرائی) * RDMA_CHUNK);
    اگر (!rdma->tx_buf || !rdma->rx_buf) واپس جھوٹ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_گہرائی) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    اگر (!rdma->tx_mr || !rdma->rx_mr) واپس جھوٹ;

    ibv_gid local_gid;
    اگر (ibv_query_gid(ibctx, ib_پورٹ, gid_idx, &local_gid) != 0) واپس جھوٹ;

    rdma_local.qpn = rdma->qp->qp_عدد;
    rdma_local.psn = rdma->qp->qp_عدد & 0xffffff;
    حافظہ_نقل(&rdma_local.gid, &local_gid, RDMA_GID_سائز);

    ثابت حرف * ver_str = "";
    اگر (gid_نسخہ == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } ورنہ اگر (gid_نسخہ == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    واپس سچ;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
منطقی socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ساخت ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.پورٹ_عدد        = rdma_local.ib_پورٹ;
        a.pkey_اشاریہ      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        اگر (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_اشاریہ | IBV_QP_پورٹ | IBV_QP_ACCESS_FLAGS) != 0) {
            واپس جھوٹ;
        }
    }

    برائے (صحیح i = 0; i < RDMA_RX_گہرائی; i++) {
        اگر (!rdma->post_rx(i)) واپس جھوٹ;
    }

    // INIT -> RTR
    {
        ساخت ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.راستہ_mtu           = rdma_local.راستہ_mtu;
        a.dest_qp_عدد        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        حافظہ_نقل(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_سائز);
        a.ah_attr.grh.hop_حد_بندی  = 1;
        a.ah_attr.grh.sgid_اشاریہ = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.پورٹ_عدد   = rdma_local.ib_پورٹ;
        اگر (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_راستہ_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            واپس جھوٹ;
        }
    }

    // RTR -> RTS
    {
        ساخت ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        اگر (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            واپس جھوٹ;
        }
    }

    rdma->آخری_فعال = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.راستہ_mtu, RDMA_RX_گہرائی);
    واپس سچ;
}

// Sleep until the completion channel has an event or the TCP peer closes.
منطقی socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ساخت pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    اگر (poll(pfds, 2, -1) < 0) {
        واپس errno == EINTR;
    }
    اگر (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        واپس جھوٹ;
    }
    اگر (pfds[0].revents & POLLIN) {
        ساخت ibv_cq * ev_cq  = خالی_اشاریہ;
        خالی          * ev_ctx = خالی_اشاریہ;
        اگر (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            واپس جھوٹ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    واپس سچ;
}

منطقی socket_t::impl::rdma_poll(ساخت ibv_cq * cq, ساخت ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    منطقی armed = جھوٹ;
    برائے (uint64_t s = 1; ; s++) {
        صحیح n = ibv_poll_cq(cq, 1, wc);
        اگر (n > 0) {
            c->آخری_فعال = std::chrono::steady_clock::now();
            اگر (wc->حالت_2 != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->حالت_2, ibv_wc_حالت_str(wc->حالت_2), wc->vendor_err);
            }
            واپس wc->حالت_2 == IBV_WC_SUCCESS;
        }
        اگر (n < 0) واپس جھوٹ;
        اگر (armed) {
            // armed and still empty: sleep until the next completion
            اگر (!rdma_wait_event()) {
                واپس جھوٹ;
            }
            armed = جھوٹ;
            جاری;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        اگر (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->آخری_فعال > RDMA_SPIN_وقت) {
            اگر (ibv_req_notify_cq(cq, 0) != 0) {
                واپس جھوٹ;
            }
            armed = سچ;
            جاری;
        }
        اگر ((s & 0xFFFFF) == 0) {
            اگر (tcp_peer_closed()) {
                واپس جھوٹ;
            }
        }
    }
}

منطقی socket_t::impl::rdma_send(ثابت خالی * ڈیٹا, size_t size) {
    rdma_conn * c = rdma.get();
    ثابت uint8_t * src = (ثابت uint8_t *)ڈیٹا;
    size_t rem = size;
    جب (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ساخت ibv_sge sge = {};
        ساخت ibv_send_wr wr = {}, * bad = خالی_اشاریہ;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.عدد_sge = 1;

        اگر (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.لمبائی = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } ورنہ {
            حافظہ_نقل(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.لمبائی = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        اگر (ibv_post_send(c->qp, &wr, &bad) != 0) واپس جھوٹ;
        c->آخری_فعال = std::chrono::steady_clock::now();
        ساخت ibv_wc wc;
        اگر (!rdma_poll(c->scq, &wc)) واپس جھوٹ;

        src += chunk;
        rem -= chunk;
    }
    واپس سچ;
}

منطقی socket_t::impl::rdma_recv(خالی * ڈیٹا, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)ڈیٹا;
    size_t rem = size;
    جب (rem > 0) {
        ساخت ibv_wc wc;
        اگر (!rdma_poll(c->rcq, &wc)) واپس جھوٹ;

        صحیح خانہ = (صحیح)wc.wr_id;
        size_t got = wc.byte_len;
        حافظہ_نقل(dst, c->rx_خانہ(خانہ), got);

        اگر (!c->post_rx(خانہ)) واپس جھوٹ;

        dst += got;
        rem -= got;
    }
    واپس سچ;
}

#اگر_ختم // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#اگر_ختم // GGML_RPC_RDMA

منطقی socket_t::impl::send_ڈیٹا(ثابت خالی * ڈیٹا, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_سائز(size).c_str());
#اگر_تعریف_شدہ GGML_RPC_RDMA_APPLE
    اگر (use_rdma) {
        واپس rdma->send(ڈیٹا, size);
    }
#وگرنہ_اگر defined(GGML_RPC_RDMA)
    اگر (use_rdma) {
        واپس rdma_send(ڈیٹا, size);
    }
#اگر_ختم
    size_t بائٹیں_بھیجا = 0;
    جب (بائٹیں_بھیجا < size) {
        size_t سائز_مقصد_send = std::min(size - بائٹیں_بھیجا, MAX_CHUNK_سائز);
        ssize_t n = send(fd, (ثابت حرف *)ڈیٹا + بائٹیں_بھیجا, سائز_مقصد_send, 0);
        اگر (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           بائٹیں_بھیجا, سائز_مقصد_send);
            واپس جھوٹ;
        }
        بائٹیں_بھیجا += (size_t)n;
    }
    واپس سچ;
}

منطقی socket_t::impl::recv_ڈیٹا(خالی * ڈیٹا, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_سائز(size).c_str());
#اگر_تعریف_شدہ GGML_RPC_RDMA_APPLE
    اگر (use_rdma) {
        واپس rdma->recv(ڈیٹا, size);
    }
#وگرنہ_اگر defined(GGML_RPC_RDMA)
    اگر (use_rdma) {
        واپس rdma_recv(ڈیٹا, size);
    }
#اگر_ختم
    size_t بائٹیں_recv = 0;
    جب (بائٹیں_recv < size) {
        size_t سائز_مقصد_recv = std::min(size - بائٹیں_recv, MAX_CHUNK_سائز);
        ssize_t n = recv(fd, (حرف *)ڈیٹا + بائٹیں_recv, سائز_مقصد_recv, 0);
        اگر (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           بائٹیں_recv, سائز_مقصد_recv);
            واپس جھوٹ;
        }
        اگر (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            واپس جھوٹ;
        }
        بائٹیں_recv += (size_t)n;
    }
    واپس سچ;
}

خالی socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_سائز);
#اگر_تعریف_شدہ GGML_RPC_RDMA
    اگر (std::getenv("GGML_RPC_NO_RDMA")) {
        واپس;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    خودکار ہدف_gid = rdma_build_ہدف_gid();
    اگر (ہدف_gid) {
        rdma = apple_rdma::probe(fd, ہدف_gid->ڈیٹا(), local_caps);
    }
    اگر (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  ورنہ
    rdma_local = {};
    اگر (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        حافظہ_نقل(rc.gid, rdma_local.gid, RDMA_GID_سائز);
        حافظہ_نقل(local_caps, &rc, حجم(rc));
    } ورنہ {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#اگر_ختم // GGML_RPC_RDMA
}

خالی socket_t::impl::update_caps(ثابت uint8_t * remote_caps) {
#اگر_تعریف_شدہ GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    منطقی remote_rdma = جھوٹ;
    برائے (size_t i = 0; i < RPC_CONN_CAPS_سائز; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    اگر (!rdma || !remote_rdma) {
        اگر (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        واپس;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    منطقی activated = rdma->activate(remote_caps);
#  ورنہ
    rdma_caps rc = {};
    حافظہ_نقل(&rc, remote_caps, حجم(rc));
    منطقی activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    اگر (activated) {
        use_rdma = سچ;
    } ورنہ {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#ورنہ
    (خالی)remote_caps;
#اگر_ختم // GGML_RPC_RDMA
}

منطقی socket_t::impl::flush() {
#اگر_تعریف_شدہ GGML_RPC_RDMA_APPLE
    اگر (use_rdma) {
        واپس rdma->flush();
    }
#اگر_ختم
    واپس سچ;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = طےشدہ;

منطقی socket_t::send_ڈیٹا(ثابت خالی * ڈیٹا, size_t size) {
    واپس pimpl->send_ڈیٹا(ڈیٹا, size);
}

منطقی socket_t::recv_ڈیٹا(خالی * ڈیٹا, size_t size) {
    واپس pimpl->recv_ڈیٹا(ڈیٹا, size);
}

منطقی socket_t::flush() {
    واپس pimpl->flush();
}

خالی socket_t::get_caps(uint8_t * local_caps) {
    واپس pimpl->get_caps(local_caps);
}

خالی socket_t::update_caps(ثابت uint8_t * remote_caps) {
    واپس pimpl->update_caps(remote_caps);
}

ثابت حرف * socket_t::transport_name() ثابت {
    واپس pimpl->use_rdma ? "RDMA" : "TCP";
}

ساکن منطقی is_valid_fd(sockfd_t sockfd) {
#اگر_تعریف_شدہ _WIN32
    واپس sockfd != INVALID_SOCKET;
#ورنہ
    واپس sockfd >= 0;
#اگر_ختم
}

ساکن منطقی سیٹ_no_delay(sockfd_t sockfd) {
    صحیح flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    صحیح ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (حرف *)&flag, حجم(صحیح));
    واپس ret == 0;
}

ساکن منطقی سیٹ_reuse_addr(sockfd_t sockfd) {
    صحیح flag = 1;
    صحیح ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (حرف *)&flag, حجم(صحیح));
    واپس ret == 0;
}

socket_ptr socket_t::accept() {
    خودکار client_socket_fd = ::accept(pimpl->fd, تہی, تہی);
    اگر (!is_valid_fd(client_socket_fd)) {
        واپس خالی_اشاریہ;
    }
    اگر (!سیٹ_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        واپس خالی_اشاریہ;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    واپس socket_ptr(نیا socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ثابت حرف * host, صحیح پورٹ) {
    خودکار sockfd = socket(AF_INET, SOCK_دھارا, 0);
    اگر (!is_valid_fd(sockfd)) {
        واپس خالی_اشاریہ;
    }
    اگر (!سیٹ_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        واپس خالی_اشاریہ;
    }
    اگر (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        واپس خالی_اشاریہ;
    }
    ساخت sockaddr_اندر serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_پورٹ = htons(پورٹ);

    اگر (bind(sockfd, (ساخت sockaddr *) &serv_addr, حجم(serv_addr)) < 0) {
        واپس خالی_اشاریہ;
    }
    اگر (listen(sockfd, 1) < 0) {
        واپس خالی_اشاریہ;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, پورٹ);
    واپس socket_ptr(نیا socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ثابت حرف * host, صحیح پورٹ) {
    خودکار sockfd = socket(AF_INET, SOCK_دھارا, 0);
    اگر (!is_valid_fd(sockfd)) {
        واپس خالی_اشاریہ;
    }
    اگر (!سیٹ_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        واپس خالی_اشاریہ;
    }
    ساخت sockaddr_اندر addr;
    addr.sin_family = AF_INET;
    addr.sin_پورٹ = htons(پورٹ);
    ساخت hostent * server = gethostbyname(host);
    اگر (server == تہی) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        واپس خالی_اشاریہ;
    }
    حافظہ_نقل(&addr.sin_addr.s_addr, server->h_addr, server->h_لمبائی);
    اگر (::connect(sockfd, (ساخت sockaddr *)&addr, حجم(addr)) < 0) {
        واپس خالی_اشاریہ;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, پورٹ);
    واپس socket_ptr(نیا socket_t(std::make_unique<impl>(sockfd)));
}

#اگر_تعریف_شدہ _WIN32
ساکن std::mutex g_rpc_transport_mu;
ساکن منطقی g_rpc_transport_wsa_started = جھوٹ;
#اگر_ختم

منطقی rpc_transport_init() {
#اگر_تعریف_شدہ _WIN32
    std::lock_guard<std::mutex> تالا(g_rpc_transport_mu);
    اگر (g_rpc_transport_wsa_started) {
        واپس سچ;
    }
    WSADATA wsa_ڈیٹا;
    صحیح res = WSAStartup(MAKEWORD(2, 2), &wsa_ڈیٹا);
    اگر (res != 0) {
        واپس جھوٹ;
    }
    g_rpc_transport_wsa_started = سچ;
    واپس سچ;
#ورنہ
    واپس سچ;
#اگر_ختم
}

خالی rpc_transport_shutdown() {
#اگر_تعریف_شدہ _WIN32
    std::lock_guard<std::mutex> تالا(g_rpc_transport_mu);
    اگر (!g_rpc_transport_wsa_started) {
        واپس;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = جھوٹ;
#اگر_ختم
}
