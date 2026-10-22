#समावेश "transport.h"
#समावेश "log.h"

#अगर_घोषित _WIN32
#  define WIN32_LEAN_AND_MEAN
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
#पूर्ण_अगर
#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <mutex>
#समावेश <optional>

#अगर_घोषित GGML_RPC_RDMA
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
#पूर्ण_अगर // GGML_RPC_RDMA

#अगर_घोषित _WIN32
प्रकार SOCKET sockfd_t;
using ssize_t = __int64;
#अन्यथा
प्रकार पूर्णांक sockfd_t;
#पूर्ण_अगर

#अगर_घोषित GGML_RPC_RDMA
अटल constexpr माप_प्रकार RDMA_GID_आकार = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_आकार>;
#पूर्ण_अगर // GGML_RPC_RDMA

#अगर defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
अटल constexpr माप_प्रकार RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
अटल constexpr पूर्णांक    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
अटल constexpr स्वतः   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

काष्ठा rdma_conn {
    काष्ठा ibv_context * ctx = nullptr;
    काष्ठा ibv_pd * pd  = nullptr;
    काष्ठा ibv_cq * scq = nullptr;   // send completions
    काष्ठा ibv_cq * rcq = nullptr;   // recv completions
    काष्ठा ibv_qp * qp  = nullptr;
    काष्ठा ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    व्योम          * tx_buf = nullptr;
    काष्ठा ibv_mr * tx_mr  = nullptr;

    व्योम          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    काष्ठा ibv_mr * rx_mr  = nullptr;
    पूर्णांक             rx_शीर्ष = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(पूर्णांक i) स्थिर {
        वापस static_cast<uint8_t *>(rx_buf) + static_cast<माप_प्रकार>(i) * RDMA_CHUNK;
    }

    bool post_rx(पूर्णांक i) {
        काष्ठा ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.लंबाई = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        काष्ठा ibv_recv_wr wr = {}, * बुरा = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;
        वापस ibv_post_recv(qp, &wr, &बुरा) == 0;
    }

    ~rdma_conn() {
        अगर (tx_mr) ibv_dereg_mr(tx_mr);
        अगर (rx_mr) ibv_dereg_mr(rx_mr);
        मुक्त(tx_buf);
        मुक्त(rx_buf);
        अगर (qp)  ibv_destroy_qp(qp);
        अगर (scq) ibv_destroy_cq(scq);
        अगर (rcq) ibv_destroy_cq(rcq);
        अगर (ch)  ibv_destroy_comp_channel(ch);
        अगर (pd)  ibv_dealloc_pd(pd);
        अगर (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
काष्ठा rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_आकार] = {};
    uint8_t  ib_port = 0;
    पूर्णांक      gid_idx = 0;
    क्रमागत ibv_mtu path_mtu = IBV_MTU_1024;
};

काष्ठा rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_आकार];
};

static_assert(माप(rdma_caps) == RPC_CONN_CAPS_आकार, "rdma_caps must match conn_caps size");

#पूर्ण_अगर // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

काष्ठा socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_तथ्य(स्थिर व्योम * तथ्य, माप_प्रकार size);
    bool recv_तथ्य(व्योम * तथ्य, माप_प्रकार size);
    bool सफाई();
    व्योम get_caps(uint8_t * local_caps);
    व्योम update_caps(स्थिर uint8_t * remote_caps);

#अगर_घोषित GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  अन्यथा
    bool rdma_probe();
    bool rdma_send(स्थिर व्योम * तथ्य, माप_प्रकार size);
    bool rdma_recv(व्योम * तथ्य, माप_प्रकार size);
    bool tcp_peer_closed();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, स्थिर uint8_t * remote_gid);
    bool rdma_poll(काष्ठा ibv_cq * cq, काष्ठा ibv_wc * wc);
    bool rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#पूर्ण_अगर // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#अगर_घोषित GGML_RPC_RDMA
    rdma.reset();
#पूर्ण_अगर // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, यह->fd);
#अगर_घोषित _WIN32
    अगर (fd != INVALID_SOCKET) closesocket(यह->fd);
#अन्यथा
    अगर (fd >= 0) बंद(यह->fd);
#पूर्ण_अगर
}

#अगर_घोषित GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = माप(addr);
    अगर (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        वापस std::nullopt;
    }
    rdma_gid_t target = {};
    अगर (addr.ss_family == AF_INET) {
        स्थिर स्वतः * a = reinterpret_cast<स्थिर sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        स_नकल(&target[12], &a->sin_addr, 4);
        वापस target;
    }
    अगर (addr.ss_family == AF_INET6) {
        स्थिर स्वतः * a = reinterpret_cast<स्थिर sockaddr_in6 *>(&addr);
        स_नकल(target.तथ्य(), &a->sin6_addr, RDMA_GID_आकार);
        वापस target;
    }
    वापस std::nullopt;
}

#अगर_अघोषित GGML_RPC_RDMA_APPLE

bool socket_t::impl::tcp_peer_closed() {
    अगर (fd < 0) वापस false;
#अगर_अघोषित _WIN32
    काष्ठा pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    पूर्णांक r = poll(&pfd, 1, 0);
    वापस r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#अन्यथा
    वापस false;
#पूर्ण_अगर
}

bool socket_t::impl::rdma_probe() {
    स्थिर अक्षर * dev_env = std::दो_पर्या("GGML_RDMA_DEV");
    स्थिर अक्षर * gid_env = std::दो_पर्या("GGML_RDMA_GID");

    स्वतः target_gid = rdma_build_target_gid();
    अगर (!target_gid) {
        वापस false;
    }

    स्थिर uint8_t ib_port = 1;
    पूर्णांक अंक_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&अंक_devs);
    अगर (!devs || अंक_devs == 0) वापस false;

    ibv_context * ibctx = nullptr;
    स्थिर अक्षर * matched_dev = nullptr;
    पूर्णांक gid_idx = gid_env ? म_से_प(gid_env) : -1;
    पूर्णांक gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    क्रम (पूर्णांक d = 0; d < अंक_devs; d++) {
        स्थिर अक्षर * dn = ibv_get_device_name(devs[d]);
        अगर (dev_env && म_भेद(dev_env, dn) != 0) जारी;

        ibv_context * ctx = ibv_open_device(devs[d]);
        अगर (!ctx) जारी;

        ibv_port_attr pa;
        अगर (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); जारी; }

        पूर्णांक found_gid = gid_idx;
        पूर्णांक found_version = IBV_GID_TYPE_IB;
        अगर (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            पूर्णांक v2_idx = -1;
            पूर्णांक v1_idx = -1;
            क्रम (पूर्णांक i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                अगर (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) जारी;
                अगर (स_भेद(entry.gid.raw, target_gid->तथ्य(), RDMA_GID_आकार) != 0) जारी;
                अगर (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } अन्यथा अगर (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            अगर (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } अन्यथा अगर (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } अन्यथा {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            अगर (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        अगर (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            अवरोध;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    अगर (!ibctx) वापस false;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    अगर (!rdma->pd) वापस false;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    अगर (!rdma->scq || !rdma->rcq) वापस false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_तथ्य = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    अगर (!rdma->qp) वापस false;
    rdma->max_inline = qia.cap.max_inline_तथ्य;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<माप_प्रकार>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    अगर (!rdma->tx_buf || !rdma->rx_buf) वापस false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<माप_प्रकार>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    अगर (!rdma->tx_mr || !rdma->rx_mr) वापस false;

    ibv_gid local_gid;
    अगर (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) वापस false;

    rdma_local.qpn = rdma->qp->qp_अंक;
    rdma_local.psn = rdma->qp->qp_अंक & 0xffffff;
    स_नकल(&rdma_local.gid, &local_gid, RDMA_GID_आकार);

    स्थिर अक्षर * ver_str = "";
    अगर (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } अन्यथा अगर (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    वापस true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, स्थिर uint8_t * remote_gid) {
    // RESET -> INIT
    {
        काष्ठा ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_अंक        = rdma_local.ib_port;
        a.pkey_सूचकांक      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        अगर (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_सूचकांक | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            वापस false;
        }
    }

    क्रम (पूर्णांक i = 0; i < RDMA_RX_DEPTH; i++) {
        अगर (!rdma->post_rx(i)) वापस false;
    }

    // INIT -> RTR
    {
        काष्ठा ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_अंक        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        स_नकल(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_आकार);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_सूचकांक = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_अंक   = rdma_local.ib_port;
        अगर (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            वापस false;
        }
    }

    // RTR -> RTS
    {
        काष्ठा ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        अगर (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            वापस false;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    वापस true;
}

// Sleep until the completion channel has an event or the TCP peer closes.
bool socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.लो();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    काष्ठा pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    अगर (poll(pfds, 2, -1) < 0) {
        वापस त्रुटि_सं == EINTR;
    }
    अगर (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        वापस false;
    }
    अगर (pfds[0].revents & POLLIN) {
        काष्ठा ibv_cq * ev_cq  = nullptr;
        व्योम          * ev_ctx = nullptr;
        अगर (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            वापस false;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    वापस true;
}

bool socket_t::impl::rdma_poll(काष्ठा ibv_cq * cq, काष्ठा ibv_wc * wc) {
    rdma_conn * c = rdma.लो();
    bool armed = false;
    क्रम (uint64_t s = 1; ; s++) {
        पूर्णांक n = ibv_poll_cq(cq, 1, wc);
        अगर (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            अगर (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            वापस wc->status == IBV_WC_SUCCESS;
        }
        अगर (n < 0) वापस false;
        अगर (armed) {
            // armed and still empty: sleep until the next completion
            अगर (!rdma_wait_event()) {
                वापस false;
            }
            armed = false;
            जारी;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        अगर (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            अगर (ibv_req_notify_cq(cq, 0) != 0) {
                वापस false;
            }
            armed = true;
            जारी;
        }
        अगर ((s & 0xFFFFF) == 0) {
            अगर (tcp_peer_closed()) {
                वापस false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(स्थिर व्योम * तथ्य, माप_प्रकार size) {
    rdma_conn * c = rdma.लो();
    स्थिर uint8_t * src = (स्थिर uint8_t *)तथ्य;
    माप_प्रकार rem = size;
    जबतक (rem > 0) {
        माप_प्रकार chunk = std::min(rem, RDMA_CHUNK);

        काष्ठा ibv_sge sge = {};
        काष्ठा ibv_send_wr wr = {}, * बुरा = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;

        अगर (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.लंबाई = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } अन्यथा {
            स_नकल(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.लंबाई = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        अगर (ibv_post_send(c->qp, &wr, &बुरा) != 0) वापस false;
        c->last_active = std::chrono::steady_clock::now();
        काष्ठा ibv_wc wc;
        अगर (!rdma_poll(c->scq, &wc)) वापस false;

        src += chunk;
        rem -= chunk;
    }
    वापस true;
}

bool socket_t::impl::rdma_recv(व्योम * तथ्य, माप_प्रकार size) {
    rdma_conn * c = rdma.लो();
    uint8_t * dst = (uint8_t *)तथ्य;
    माप_प्रकार rem = size;
    जबतक (rem > 0) {
        काष्ठा ibv_wc wc;
        अगर (!rdma_poll(c->rcq, &wc)) वापस false;

        पूर्णांक slot = (पूर्णांक)wc.wr_id;
        माप_प्रकार got = wc.byte_len;
        स_नकल(dst, c->rx_slot(slot), got);

        अगर (!c->post_rx(slot)) वापस false;

        dst += got;
        rem -= got;
    }
    वापस true;
}

#पूर्ण_अगर // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#पूर्ण_अगर // GGML_RPC_RDMA

bool socket_t::impl::send_तथ्य(स्थिर व्योम * तथ्य, माप_प्रकार size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_आकार(size).c_str());
#अगर_घोषित GGML_RPC_RDMA_APPLE
    अगर (use_rdma) {
        वापस rdma->send(तथ्य, size);
    }
#या_अगर defined(GGML_RPC_RDMA)
    अगर (use_rdma) {
        वापस rdma_send(तथ्य, size);
    }
#पूर्ण_अगर
    माप_प्रकार bytes_sent = 0;
    जबतक (bytes_sent < size) {
        माप_प्रकार आकार_to_send = std::min(size - bytes_sent, MAX_CHUNK_आकार);
        ssize_t n = send(fd, (स्थिर अक्षर *)तथ्य + bytes_sent, आकार_to_send, 0);
        अगर (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, आकार_to_send);
            वापस false;
        }
        bytes_sent += (माप_प्रकार)n;
    }
    वापस true;
}

bool socket_t::impl::recv_तथ्य(व्योम * तथ्य, माप_प्रकार size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_आकार(size).c_str());
#अगर_घोषित GGML_RPC_RDMA_APPLE
    अगर (use_rdma) {
        वापस rdma->recv(तथ्य, size);
    }
#या_अगर defined(GGML_RPC_RDMA)
    अगर (use_rdma) {
        वापस rdma_recv(तथ्य, size);
    }
#पूर्ण_अगर
    माप_प्रकार bytes_recv = 0;
    जबतक (bytes_recv < size) {
        माप_प्रकार आकार_to_recv = std::min(size - bytes_recv, MAX_CHUNK_आकार);
        ssize_t n = recv(fd, (अक्षर *)तथ्य + bytes_recv, आकार_to_recv, 0);
        अगर (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, आकार_to_recv);
            वापस false;
        }
        अगर (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            वापस false;
        }
        bytes_recv += (माप_प्रकार)n;
    }
    वापस true;
}

व्योम socket_t::impl::get_caps(uint8_t * local_caps) {
    स_रखो(local_caps, 0, RPC_CONN_CAPS_आकार);
#अगर_घोषित GGML_RPC_RDMA
    अगर (std::दो_पर्या("GGML_RPC_NO_RDMA")) {
        वापस;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    स्वतः target_gid = rdma_build_target_gid();
    अगर (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->तथ्य(), local_caps);
    }
    अगर (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  अन्यथा
    rdma_local = {};
    अगर (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        स_नकल(rc.gid, rdma_local.gid, RDMA_GID_आकार);
        स_नकल(local_caps, &rc, माप(rc));
    } अन्यथा {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#पूर्ण_अगर // GGML_RPC_RDMA
}

व्योम socket_t::impl::update_caps(स्थिर uint8_t * remote_caps) {
#अगर_घोषित GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    bool remote_rdma = false;
    क्रम (माप_प्रकार i = 0; i < RPC_CONN_CAPS_आकार; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    अगर (!rdma || !remote_rdma) {
        अगर (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        वापस;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    bool activated = rdma->activate(remote_caps);
#  अन्यथा
    rdma_caps rc = {};
    स_नकल(&rc, remote_caps, माप(rc));
    bool activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    अगर (activated) {
        use_rdma = true;
    } अन्यथा {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#अन्यथा
    (व्योम)remote_caps;
#पूर्ण_अगर // GGML_RPC_RDMA
}

bool socket_t::impl::सफाई() {
#अगर_घोषित GGML_RPC_RDMA_APPLE
    अगर (use_rdma) {
        वापस rdma->सफाई();
    }
#पूर्ण_अगर
    वापस true;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = शेष;

bool socket_t::send_तथ्य(स्थिर व्योम * तथ्य, माप_प्रकार size) {
    वापस pimpl->send_तथ्य(तथ्य, size);
}

bool socket_t::recv_तथ्य(व्योम * तथ्य, माप_प्रकार size) {
    वापस pimpl->recv_तथ्य(तथ्य, size);
}

bool socket_t::सफाई() {
    वापस pimpl->सफाई();
}

व्योम socket_t::get_caps(uint8_t * local_caps) {
    वापस pimpl->get_caps(local_caps);
}

व्योम socket_t::update_caps(स्थिर uint8_t * remote_caps) {
    वापस pimpl->update_caps(remote_caps);
}

स्थिर अक्षर * socket_t::transport_name() स्थिर {
    वापस pimpl->use_rdma ? "RDMA" : "TCP";
}

अटल bool is_valid_fd(sockfd_t sockfd) {
#अगर_घोषित _WIN32
    वापस sockfd != INVALID_SOCKET;
#अन्यथा
    वापस sockfd >= 0;
#पूर्ण_अगर
}

अटल bool set_no_delay(sockfd_t sockfd) {
    पूर्णांक flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    पूर्णांक ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (अक्षर *)&flag, माप(पूर्णांक));
    वापस ret == 0;
}

अटल bool set_reuse_addr(sockfd_t sockfd) {
    पूर्णांक flag = 1;
    पूर्णांक ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (अक्षर *)&flag, माप(पूर्णांक));
    वापस ret == 0;
}

socket_ptr socket_t::accept() {
    स्वतः client_socket_fd = ::accept(pimpl->fd, शून्य, शून्य);
    अगर (!is_valid_fd(client_socket_fd)) {
        वापस nullptr;
    }
    अगर (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        वापस nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    वापस socket_ptr(नया socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(स्थिर अक्षर * host, पूर्णांक port) {
    स्वतः sockfd = socket(AF_INET, SOCK_STREAM, 0);
    अगर (!is_valid_fd(sockfd)) {
        वापस nullptr;
    }
    अगर (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        वापस nullptr;
    }
    अगर (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        वापस nullptr;
    }
    काष्ठा sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    अगर (bind(sockfd, (काष्ठा sockaddr *) &serv_addr, माप(serv_addr)) < 0) {
        वापस nullptr;
    }
    अगर (listen(sockfd, 1) < 0) {
        वापस nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    वापस socket_ptr(नया socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(स्थिर अक्षर * host, पूर्णांक port) {
    स्वतः sockfd = socket(AF_INET, SOCK_STREAM, 0);
    अगर (!is_valid_fd(sockfd)) {
        वापस nullptr;
    }
    अगर (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        वापस nullptr;
    }
    काष्ठा sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    काष्ठा hostent * server = gethostbyname(host);
    अगर (server == शून्य) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        वापस nullptr;
    }
    स_नकल(&addr.sin_addr.s_addr, server->h_addr, server->h_लंबाई);
    अगर (::connect(sockfd, (काष्ठा sockaddr *)&addr, माप(addr)) < 0) {
        वापस nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    वापस socket_ptr(नया socket_t(std::make_unique<impl>(sockfd)));
}

#अगर_घोषित _WIN32
अटल std::mutex g_rpc_transport_mu;
अटल bool g_rpc_transport_wsa_started = false;
#पूर्ण_अगर

bool rpc_transport_init() {
#अगर_घोषित _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    अगर (g_rpc_transport_wsa_started) {
        वापस true;
    }
    WSADATA wsa_तथ्य;
    पूर्णांक res = WSAStartup(MAKEWORD(2, 2), &wsa_तथ्य);
    अगर (res != 0) {
        वापस false;
    }
    g_rpc_transport_wsa_started = true;
    वापस true;
#अन्यथा
    वापस true;
#पूर्ण_अगर
}

व्योम rpc_transport_shutdown() {
#अगर_घोषित _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    अगर (!g_rpc_transport_wsa_started) {
        वापस;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#पूर्ण_अगर
}
