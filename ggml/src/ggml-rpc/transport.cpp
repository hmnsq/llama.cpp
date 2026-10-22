#ضمن "transport.h"
#ضمن "log.h"

#اگر_تعریف_شده _WIN32
#  define WIN32_LEAN_AND_میانگین
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
#پایان_اگر
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <mutex>
#ضمن <optional>

#اگر_تعریف_شده GGML_RPC_RDMA
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
#پایان_اگر // GGML_RPC_RDMA

#اگر_تعریف_شده _WIN32
تعريفنوع SOCKET sockfd_t;
استفاده ssize_t = __int64;
#وإلا
تعريفنوع صحيح sockfd_t;
#پایان_اگر

#اگر_تعریف_شده GGML_RPC_RDMA
ساكن constexpr size_t RDMA_GID_اندازه = 16;            // RoCE GID / IB GID is always 16 bytes
استفاده rdma_gid_t = std::array<uint8_t, RDMA_GID_اندازه>;
#پایان_اگر // GGML_RPC_RDMA

#إذا defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ساكن constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ساكن constexpr صحيح    RDMA_RX_عمق = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ساكن constexpr خودکار   RDMA_SPIN_زمان = std::chrono::milliseconds(100);

بنية rdma_conn {
    بنية ibv_context * ctx = اشاره_تهی;
    بنية ibv_pd * pd  = اشاره_تهی;
    بنية ibv_cq * scq = اشاره_تهی;   // send completions
    بنية ibv_cq * rcq = اشاره_تهی;   // recv completions
    بنية ibv_qp * qp  = اشاره_تهی;
    بنية ibv_comp_channel * ch = اشاره_تهی; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point آخرین_فعال; // last completion or posted send

    فراغ          * tx_buf = اشاره_تهی;
    بنية ibv_mr * tx_mr  = اشاره_تهی;

    فراغ          * rx_buf = اشاره_تهی; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    بنية ibv_mr * rx_mr  = اشاره_تهی;
    صحيح             rx_سر = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_جایگاه(صحيح i) ثابت {
        أرجع static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    منطقي post_rx(صحيح i) {
        بنية ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_جایگاه(i);
        sge.درازی = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        بنية ibv_recv_wr wr = {}, * bad = اشاره_تهی;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.رقم_sge = 1;
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
    uint8_t  gid[RDMA_GID_اندازه] = {};
    uint8_t  ib_پورت = 0;
    صحيح      gid_idx = 0;
    تعداد ibv_mtu مسیر_mtu = IBV_MTU_1024;
};

بنية rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_اندازه];
};

static_assert(حجم(rdma_caps) == RPC_CONN_CAPS_اندازه, "rdma_caps must match conn_caps size");

#پایان_اگر // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

بنية socket_t::impl {
    impl(sockfd_t fd) : use_rdma(خطأ), fd(fd) {}
    ~impl();
    منطقي send_داده(ثابت فراغ * داده, size_t size);
    منطقي recv_داده(فراغ * داده, size_t size);
    منطقي flush();
    فراغ get_caps(uint8_t * local_caps);
    فراغ update_caps(ثابت uint8_t * remote_caps);

#اگر_تعریف_شده GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_هدف_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  وإلا
    منطقي rdma_probe();
    منطقي rdma_send(ثابت فراغ * داده, size_t size);
    منطقي rdma_recv(فراغ * داده, size_t size);
    منطقي tcp_peer_closed();
    منطقي rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid);
    منطقي rdma_poll(بنية ibv_cq * cq, بنية ibv_wc * wc);
    منطقي rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#پایان_اگر // GGML_RPC_RDMA
    منطقي     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#اگر_تعریف_شده GGML_RPC_RDMA
    rdma.reset();
#پایان_اگر // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, هذا->fd);
#اگر_تعریف_شده _WIN32
    إذا (fd != INVALID_SOCKET) closesocket(هذا->fd);
#وإلا
    إذا (fd >= 0) close(هذا->fd);
#پایان_اگر
}

#اگر_تعریف_شده GGML_RPC_RDMA

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
        ثابت خودکار * a = reinterpret_cast<ثابت sockaddr_درون *>(&addr);
        هدف[10] = 0xff;
        هدف[11] = 0xff;
        کپی_حافظه(&هدف[12], &a->sin_addr, 4);
        أرجع هدف;
    }
    إذا (addr.ss_family == AF_INET6) {
        ثابت خودکار * a = reinterpret_cast<ثابت sockaddr_in6 *>(&addr);
        کپی_حافظه(هدف.داده(), &a->sin6_addr, RDMA_GID_اندازه);
        أرجع هدف;
    }
    أرجع std::nullopt;
}

#اگر_تعریف_نشده GGML_RPC_RDMA_APPLE

منطقي socket_t::impl::tcp_peer_closed() {
    إذا (fd < 0) أرجع خطأ;
#اگر_تعریف_نشده _WIN32
    بنية pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    صحيح r = poll(&pfd, 1, 0);
    أرجع r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#وإلا
    أرجع خطأ;
#پایان_اگر
}

منطقي socket_t::impl::rdma_probe() {
    ثابت حرف * dev_env = std::getenv("GGML_RDMA_DEV");
    ثابت حرف * gid_env = std::getenv("GGML_RDMA_GID");

    خودکار هدف_gid = rdma_build_هدف_gid();
    إذا (!هدف_gid) {
        أرجع خطأ;
    }

    ثابت uint8_t ib_پورت = 1;
    صحيح رقم_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&رقم_devs);
    إذا (!devs || رقم_devs == 0) أرجع خطأ;

    ibv_context * ibctx = اشاره_تهی;
    ثابت حرف * matched_dev = اشاره_تهی;
    صحيح gid_idx = gid_env ? atoi(gid_env) : -1;
    صحيح gid_نسخه = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    لكل (صحيح d = 0; d < رقم_devs; d++) {
        ثابت حرف * dn = ibv_get_device_name(devs[d]);
        إذا (dev_env && strcmp(dev_env, dn) != 0) استمر;

        ibv_context * ctx = ibv_open_device(devs[d]);
        إذا (!ctx) استمر;

        ibv_پورت_attr pa;
        إذا (ibv_query_پورت(ctx, ib_پورت, &pa) != 0) { ibv_close_device(ctx); استمر; }

        صحيح یافت_gid = gid_idx;
        صحيح یافت_نسخه = IBV_GID_TYPE_IB;
        إذا (یافت_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            صحيح v2_idx = -1;
            صحيح v1_idx = -1;
            لكل (صحيح i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_مدخل مدخل = {};
                إذا (ibv_query_gid_ex(ctx, ib_پورت, i, &مدخل, 0) != 0) استمر;
                إذا (memcmp(مدخل.gid.raw, هدف_gid->داده(), RDMA_GID_اندازه) != 0) استمر;
                إذا (مدخل.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } وإلا إذا (مدخل.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            إذا (v2_idx >= 0) {
                یافت_gid = v2_idx;
                یافت_نسخه = IBV_GID_TYPE_ROCE_V2;
            } وإلا إذا (v1_idx >= 0) {
                یافت_gid = v1_idx;
                یافت_نسخه = IBV_GID_TYPE_ROCE_V1;
            }
        } وإلا {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_مدخل مدخل = {};
            إذا (ibv_query_gid_ex(ctx, ib_پورت, یافت_gid, &مدخل, 0) == 0) {
                یافت_نسخه = مدخل.gid_type;
            }
        }
        إذا (یافت_gid >= 0) {
            ibctx = ctx;
            gid_idx = یافت_gid;
            gid_نسخه = یافت_نسخه;
            matched_dev = dn;
            rdma_local.مسیر_mtu = pa.فعال_mtu;
            اكسر;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    إذا (!ibctx) أرجع خطأ;

    rdma_local.ib_پورت = ib_پورت;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    إذا (!rdma->pd) أرجع خطأ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, اشاره_تهی, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_عمق + 4, اشاره_تهی, rdma->ch, 0);
    إذا (!rdma->scq || !rdma->rcq) أرجع خطأ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_عمق + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_داده = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    إذا (!rdma->qp) أرجع خطأ;
    rdma->max_inline = qia.cap.max_inline_داده;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_عمق) * RDMA_CHUNK);
    إذا (!rdma->tx_buf || !rdma->rx_buf) أرجع خطأ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_عمق) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    إذا (!rdma->tx_mr || !rdma->rx_mr) أرجع خطأ;

    ibv_gid local_gid;
    إذا (ibv_query_gid(ibctx, ib_پورت, gid_idx, &local_gid) != 0) أرجع خطأ;

    rdma_local.qpn = rdma->qp->qp_رقم;
    rdma_local.psn = rdma->qp->qp_رقم & 0xffffff;
    کپی_حافظه(&rdma_local.gid, &local_gid, RDMA_GID_اندازه);

    ثابت حرف * ver_str = "";
    إذا (gid_نسخه == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } وإلا إذا (gid_نسخه == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    أرجع درست;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
منطقي socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ثابت uint8_t * remote_gid) {
    // RESET -> INIT
    {
        بنية ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.پورت_رقم        = rdma_local.ib_پورت;
        a.pkey_شاخص      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        إذا (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_شاخص | IBV_QP_پورت | IBV_QP_ACCESS_FLAGS) != 0) {
            أرجع خطأ;
        }
    }

    لكل (صحيح i = 0; i < RDMA_RX_عمق; i++) {
        إذا (!rdma->post_rx(i)) أرجع خطأ;
    }

    // INIT -> RTR
    {
        بنية ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.مسیر_mtu           = rdma_local.مسیر_mtu;
        a.dest_qp_رقم        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        کپی_حافظه(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_اندازه);
        a.ah_attr.grh.hop_حد  = 1;
        a.ah_attr.grh.sgid_شاخص = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.پورت_رقم   = rdma_local.ib_پورت;
        إذا (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_مسیر_MTU | IBV_QP_DEST_QPN |
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

    rdma->آخرین_فعال = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.مسیر_mtu, RDMA_RX_عمق);
    أرجع درست;
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
        بنية ibv_cq * ev_cq  = اشاره_تهی;
        فراغ          * ev_ctx = اشاره_تهی;
        إذا (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            أرجع خطأ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    أرجع درست;
}

منطقي socket_t::impl::rdma_poll(بنية ibv_cq * cq, بنية ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    منطقي armed = خطأ;
    لكل (uint64_t s = 1; ; s++) {
        صحيح n = ibv_poll_cq(cq, 1, wc);
        إذا (n > 0) {
            c->آخرین_فعال = std::chrono::steady_clock::now();
            إذا (wc->وضعیت != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->وضعیت, ibv_wc_وضعیت_str(wc->وضعیت), wc->vendor_err);
            }
            أرجع wc->وضعیت == IBV_WC_SUCCESS;
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
        إذا (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->آخرین_فعال > RDMA_SPIN_زمان) {
            إذا (ibv_req_notify_cq(cq, 0) != 0) {
                أرجع خطأ;
            }
            armed = درست;
            استمر;
        }
        إذا ((s & 0xFFFFF) == 0) {
            إذا (tcp_peer_closed()) {
                أرجع خطأ;
            }
        }
    }
}

منطقي socket_t::impl::rdma_send(ثابت فراغ * داده, size_t size) {
    rdma_conn * c = rdma.get();
    ثابت uint8_t * src = (ثابت uint8_t *)داده;
    size_t rem = size;
    طالما (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        بنية ibv_sge sge = {};
        بنية ibv_send_wr wr = {}, * bad = اشاره_تهی;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.رقم_sge = 1;

        إذا (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.درازی = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } وإلا {
            کپی_حافظه(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.درازی = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        إذا (ibv_post_send(c->qp, &wr, &bad) != 0) أرجع خطأ;
        c->آخرین_فعال = std::chrono::steady_clock::now();
        بنية ibv_wc wc;
        إذا (!rdma_poll(c->scq, &wc)) أرجع خطأ;

        src += chunk;
        rem -= chunk;
    }
    أرجع درست;
}

منطقي socket_t::impl::rdma_recv(فراغ * داده, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)داده;
    size_t rem = size;
    طالما (rem > 0) {
        بنية ibv_wc wc;
        إذا (!rdma_poll(c->rcq, &wc)) أرجع خطأ;

        صحيح جایگاه = (صحيح)wc.wr_id;
        size_t got = wc.byte_len;
        کپی_حافظه(dst, c->rx_جایگاه(جایگاه), got);

        إذا (!c->post_rx(جایگاه)) أرجع خطأ;

        dst += got;
        rem -= got;
    }
    أرجع درست;
}

#پایان_اگر // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#پایان_اگر // GGML_RPC_RDMA

منطقي socket_t::impl::send_داده(ثابت فراغ * داده, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_اندازه(size).c_str());
#اگر_تعریف_شده GGML_RPC_RDMA_APPLE
    إذا (use_rdma) {
        أرجع rdma->send(داده, size);
    }
#وإلاإذا defined(GGML_RPC_RDMA)
    إذا (use_rdma) {
        أرجع rdma_send(داده, size);
    }
#پایان_اگر
    size_t بایتها_فرستاده = 0;
    طالما (بایتها_فرستاده < size) {
        size_t اندازه_مقصد_send = std::min(size - بایتها_فرستاده, MAX_CHUNK_اندازه);
        ssize_t n = send(fd, (ثابت حرف *)داده + بایتها_فرستاده, اندازه_مقصد_send, 0);
        إذا (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           بایتها_فرستاده, اندازه_مقصد_send);
            أرجع خطأ;
        }
        بایتها_فرستاده += (size_t)n;
    }
    أرجع درست;
}

منطقي socket_t::impl::recv_داده(فراغ * داده, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_اندازه(size).c_str());
#اگر_تعریف_شده GGML_RPC_RDMA_APPLE
    إذا (use_rdma) {
        أرجع rdma->recv(داده, size);
    }
#وإلاإذا defined(GGML_RPC_RDMA)
    إذا (use_rdma) {
        أرجع rdma_recv(داده, size);
    }
#پایان_اگر
    size_t بایتها_recv = 0;
    طالما (بایتها_recv < size) {
        size_t اندازه_مقصد_recv = std::min(size - بایتها_recv, MAX_CHUNK_اندازه);
        ssize_t n = recv(fd, (حرف *)داده + بایتها_recv, اندازه_مقصد_recv, 0);
        إذا (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           بایتها_recv, اندازه_مقصد_recv);
            أرجع خطأ;
        }
        إذا (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            أرجع خطأ;
        }
        بایتها_recv += (size_t)n;
    }
    أرجع درست;
}

فراغ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_اندازه);
#اگر_تعریف_شده GGML_RPC_RDMA
    إذا (std::getenv("GGML_RPC_NO_RDMA")) {
        أرجع;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    خودکار هدف_gid = rdma_build_هدف_gid();
    إذا (هدف_gid) {
        rdma = apple_rdma::probe(fd, هدف_gid->داده(), local_caps);
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
        کپی_حافظه(rc.gid, rdma_local.gid, RDMA_GID_اندازه);
        کپی_حافظه(local_caps, &rc, حجم(rc));
    } وإلا {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#پایان_اگر // GGML_RPC_RDMA
}

فراغ socket_t::impl::update_caps(ثابت uint8_t * remote_caps) {
#اگر_تعریف_شده GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    منطقي remote_rdma = خطأ;
    لكل (size_t i = 0; i < RPC_CONN_CAPS_اندازه; i++) {
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
    کپی_حافظه(&rc, remote_caps, حجم(rc));
    منطقي activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    إذا (activated) {
        use_rdma = درست;
    } وإلا {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#وإلا
    (فراغ)remote_caps;
#پایان_اگر // GGML_RPC_RDMA
}

منطقي socket_t::impl::flush() {
#اگر_تعریف_شده GGML_RPC_RDMA_APPLE
    إذا (use_rdma) {
        أرجع rdma->flush();
    }
#پایان_اگر
    أرجع درست;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = افتراضي;

منطقي socket_t::send_داده(ثابت فراغ * داده, size_t size) {
    أرجع pimpl->send_داده(داده, size);
}

منطقي socket_t::recv_داده(فراغ * داده, size_t size) {
    أرجع pimpl->recv_داده(داده, size);
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
#اگر_تعریف_شده _WIN32
    أرجع sockfd != INVALID_SOCKET;
#وإلا
    أرجع sockfd >= 0;
#پایان_اگر
}

ساكن منطقي مجموعه_no_delay(sockfd_t sockfd) {
    صحيح flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    صحيح ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (حرف *)&flag, حجم(صحيح));
    أرجع ret == 0;
}

ساكن منطقي مجموعه_reuse_addr(sockfd_t sockfd) {
    صحيح flag = 1;
    صحيح ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (حرف *)&flag, حجم(صحيح));
    أرجع ret == 0;
}

socket_ptr socket_t::accept() {
    خودکار client_socket_fd = ::accept(pimpl->fd, عدم, عدم);
    إذا (!is_valid_fd(client_socket_fd)) {
        أرجع اشاره_تهی;
    }
    إذا (!مجموعه_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        أرجع اشاره_تهی;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ثابت حرف * host, صحيح پورت) {
    خودکار sockfd = socket(AF_INET, SOCK_جریان, 0);
    إذا (!is_valid_fd(sockfd)) {
        أرجع اشاره_تهی;
    }
    إذا (!مجموعه_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        أرجع اشاره_تهی;
    }
    إذا (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        أرجع اشاره_تهی;
    }
    بنية sockaddr_درون serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_پورت = htons(پورت);

    إذا (bind(sockfd, (بنية sockaddr *) &serv_addr, حجم(serv_addr)) < 0) {
        أرجع اشاره_تهی;
    }
    إذا (listen(sockfd, 1) < 0) {
        أرجع اشاره_تهی;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, پورت);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ثابت حرف * host, صحيح پورت) {
    خودکار sockfd = socket(AF_INET, SOCK_جریان, 0);
    إذا (!is_valid_fd(sockfd)) {
        أرجع اشاره_تهی;
    }
    إذا (!مجموعه_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        أرجع اشاره_تهی;
    }
    بنية sockaddr_درون addr;
    addr.sin_family = AF_INET;
    addr.sin_پورت = htons(پورت);
    بنية hostent * server = gethostbyname(host);
    إذا (server == عدم) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        أرجع اشاره_تهی;
    }
    کپی_حافظه(&addr.sin_addr.s_addr, server->h_addr, server->h_درازی);
    إذا (::connect(sockfd, (بنية sockaddr *)&addr, حجم(addr)) < 0) {
        أرجع اشاره_تهی;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, پورت);
    أرجع socket_ptr(جديد socket_t(std::make_unique<impl>(sockfd)));
}

#اگر_تعریف_شده _WIN32
ساكن std::mutex g_rpc_transport_mu;
ساكن منطقي g_rpc_transport_wsa_started = خطأ;
#پایان_اگر

منطقي rpc_transport_init() {
#اگر_تعریف_شده _WIN32
    std::lock_guard<std::mutex> قفل(g_rpc_transport_mu);
    إذا (g_rpc_transport_wsa_started) {
        أرجع درست;
    }
    WSADATA wsa_داده;
    صحيح res = WSAStartup(MAKEWORD(2, 2), &wsa_داده);
    إذا (res != 0) {
        أرجع خطأ;
    }
    g_rpc_transport_wsa_started = درست;
    أرجع درست;
#وإلا
    أرجع درست;
#پایان_اگر
}

فراغ rpc_transport_shutdown() {
#اگر_تعریف_شده _WIN32
    std::lock_guard<std::mutex> قفل(g_rpc_transport_mu);
    إذا (!g_rpc_transport_wsa_started) {
        أرجع;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = خطأ;
#پایان_اگر
}
