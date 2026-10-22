#अन्तर्भाव "transport.h"
#अन्तर्भाव "log.h"

#ifdef _WIN32
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
#endif
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <mutex>
#अन्तर्भाव <optional>

#ifdef GGML_RPC_RDMA
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
#endif // GGML_RPC_RDMA

#ifdef _WIN32
प्रकारव्याख्या SOCKET sockfd_t;
using ssize_t = __int64;
#अन्यथा
प्रकारव्याख्या पूर्णांक sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
स्थिर constexpr माप_प्रकार RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#जर defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
स्थिर constexpr माप_प्रकार RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
स्थिर constexpr पूर्णांक    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
स्थिर constexpr स्वतः   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

संरचना rdma_conn {
    संरचना ibv_context * ctx = nullptr;
    संरचना ibv_pd * pd  = nullptr;
    संरचना ibv_cq * scq = nullptr;   // send completions
    संरचना ibv_cq * rcq = nullptr;   // recv completions
    संरचना ibv_qp * qp  = nullptr;
    संरचना ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    शून्य          * tx_buf = nullptr;
    संरचना ibv_mr * tx_mr  = nullptr;

    शून्य          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    संरचना ibv_mr * rx_mr  = nullptr;
    पूर्णांक             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(पूर्णांक i) स्थिरांक {
        परत static_cast<uint8_t *>(rx_buf) + static_cast<माप_प्रकार>(i) * RDMA_CHUNK;
    }

    तार्किक post_rx(पूर्णांक i) {
        संरचना ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        संरचना ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;
        परत ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        जर (tx_mr) ibv_dereg_mr(tx_mr);
        जर (rx_mr) ibv_dereg_mr(rx_mr);
        मुक्त(tx_buf);
        मुक्त(rx_buf);
        जर (qp)  ibv_destroy_qp(qp);
        जर (scq) ibv_destroy_cq(scq);
        जर (rcq) ibv_destroy_cq(rcq);
        जर (ch)  ibv_destroy_comp_channel(ch);
        जर (pd)  ibv_dealloc_pd(pd);
        जर (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
संरचना rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    पूर्णांक      gid_idx = 0;
    गणना ibv_mtu path_mtu = IBV_MTU_1024;
};

संरचना rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(आकार(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

संरचना socket_t::impl {
    impl(sockfd_t fd) : use_rdma(असत्य), fd(fd) {}
    ~impl();
    तार्किक send_data(स्थिरांक शून्य * data, माप_प्रकार size);
    तार्किक recv_data(शून्य * data, माप_प्रकार size);
    तार्किक flush();
    शून्य get_caps(uint8_t * local_caps);
    शून्य update_caps(स्थिरांक uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  अन्यथा
    तार्किक rdma_probe();
    तार्किक rdma_send(स्थिरांक शून्य * data, माप_प्रकार size);
    तार्किक rdma_recv(शून्य * data, माप_प्रकार size);
    तार्किक tcp_peer_closed();
    तार्किक rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, स्थिरांक uint8_t * remote_gid);
    तार्किक rdma_poll(संरचना ibv_cq * cq, संरचना ibv_wc * wc);
    तार्किक rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    तार्किक     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, हे->fd);
#ifdef _WIN32
    जर (fd != INVALID_SOCKET) closesocket(हे->fd);
#अन्यथा
    जर (fd >= 0) बंद(हे->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = आकार(addr);
    जर (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        परत std::nullopt;
    }
    rdma_gid_t target = {};
    जर (addr.ss_family == AF_INET) {
        स्थिरांक स्वतः * a = reinterpret_cast<स्थिरांक sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        स_नकल(&target[12], &a->sin_addr, 4);
        परत target;
    }
    जर (addr.ss_family == AF_INET6) {
        स्थिरांक स्वतः * a = reinterpret_cast<स्थिरांक sockaddr_in6 *>(&addr);
        स_नकल(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        परत target;
    }
    परत std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

तार्किक socket_t::impl::tcp_peer_closed() {
    जर (fd < 0) परत असत्य;
#ifndef _WIN32
    संरचना pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    पूर्णांक r = poll(&pfd, 1, 0);
    परत r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#अन्यथा
    परत असत्य;
#endif
}

तार्किक socket_t::impl::rdma_probe() {
    स्थिरांक अक्षर * dev_env = std::दो_पर्या("GGML_RDMA_DEV");
    स्थिरांक अक्षर * gid_env = std::दो_पर्या("GGML_RDMA_GID");

    स्वतः target_gid = rdma_build_target_gid();
    जर (!target_gid) {
        परत असत्य;
    }

    स्थिरांक uint8_t ib_port = 1;
    पूर्णांक अंक_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&अंक_devs);
    जर (!devs || अंक_devs == 0) परत असत्य;

    ibv_context * ibctx = nullptr;
    स्थिरांक अक्षर * matched_dev = nullptr;
    पूर्णांक gid_idx = gid_env ? म_से_प(gid_env) : -1;
    पूर्णांक gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    साठी (पूर्णांक d = 0; d < अंक_devs; d++) {
        स्थिरांक अक्षर * dn = ibv_get_device_name(devs[d]);
        जर (dev_env && म_भेद(dev_env, dn) != 0) सुरूठेवा;

        ibv_context * ctx = ibv_open_device(devs[d]);
        जर (!ctx) सुरूठेवा;

        ibv_port_attr pa;
        जर (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); सुरूठेवा; }

        पूर्णांक found_gid = gid_idx;
        पूर्णांक found_version = IBV_GID_TYPE_IB;
        जर (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            पूर्णांक v2_idx = -1;
            पूर्णांक v1_idx = -1;
            साठी (पूर्णांक i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                जर (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) सुरूठेवा;
                जर (स_भेद(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) सुरूठेवा;
                जर (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } अन्यथा जर (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            जर (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } अन्यथा जर (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } अन्यथा {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            जर (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        जर (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            थांबा;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    जर (!ibctx) परत असत्य;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    जर (!rdma->pd) परत असत्य;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    जर (!rdma->scq || !rdma->rcq) परत असत्य;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    जर (!rdma->qp) परत असत्य;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<माप_प्रकार>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    जर (!rdma->tx_buf || !rdma->rx_buf) परत असत्य;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<माप_प्रकार>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    जर (!rdma->tx_mr || !rdma->rx_mr) परत असत्य;

    ibv_gid local_gid;
    जर (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) परत असत्य;

    rdma_local.qpn = rdma->qp->qp_अंक;
    rdma_local.psn = rdma->qp->qp_अंक & 0xffffff;
    स_नकल(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    स्थिरांक अक्षर * ver_str = "";
    जर (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } अन्यथा जर (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    परत सत्य;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
तार्किक socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, स्थिरांक uint8_t * remote_gid) {
    // RESET -> INIT
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_अंक        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        जर (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            परत असत्य;
        }
    }

    साठी (पूर्णांक i = 0; i < RDMA_RX_DEPTH; i++) {
        जर (!rdma->post_rx(i)) परत असत्य;
    }

    // INIT -> RTR
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_अंक        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        स_नकल(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_अंक   = rdma_local.ib_port;
        जर (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            परत असत्य;
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
        जर (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            परत असत्य;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    परत सत्य;
}

// Sleep until the completion channel has an event or the TCP peer closes.
तार्किक socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    संरचना pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    जर (poll(pfds, 2, -1) < 0) {
        परत त्रुटि_सं == EINTR;
    }
    जर (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        परत असत्य;
    }
    जर (pfds[0].revents & POLLIN) {
        संरचना ibv_cq * ev_cq  = nullptr;
        शून्य          * ev_ctx = nullptr;
        जर (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            परत असत्य;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    परत सत्य;
}

तार्किक socket_t::impl::rdma_poll(संरचना ibv_cq * cq, संरचना ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    तार्किक armed = असत्य;
    साठी (uint64_t s = 1; ; s++) {
        पूर्णांक n = ibv_poll_cq(cq, 1, wc);
        जर (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            जर (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            परत wc->status == IBV_WC_SUCCESS;
        }
        जर (n < 0) परत असत्य;
        जर (armed) {
            // armed and still empty: sleep until the next completion
            जर (!rdma_wait_event()) {
                परत असत्य;
            }
            armed = असत्य;
            सुरूठेवा;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        जर (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            जर (ibv_req_notify_cq(cq, 0) != 0) {
                परत असत्य;
            }
            armed = सत्य;
            सुरूठेवा;
        }
        जर ((s & 0xFFFFF) == 0) {
            जर (tcp_peer_closed()) {
                परत असत्य;
            }
        }
    }
}

तार्किक socket_t::impl::rdma_send(स्थिरांक शून्य * data, माप_प्रकार size) {
    rdma_conn * c = rdma.get();
    स्थिरांक uint8_t * src = (स्थिरांक uint8_t *)data;
    माप_प्रकार rem = size;
    पर्यंत (rem > 0) {
        माप_प्रकार chunk = std::min(rem, RDMA_CHUNK);

        संरचना ibv_sge sge = {};
        संरचना ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.अंक_sge = 1;

        जर (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } अन्यथा {
            स_नकल(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        जर (ibv_post_send(c->qp, &wr, &bad) != 0) परत असत्य;
        c->last_active = std::chrono::steady_clock::now();
        संरचना ibv_wc wc;
        जर (!rdma_poll(c->scq, &wc)) परत असत्य;

        src += chunk;
        rem -= chunk;
    }
    परत सत्य;
}

तार्किक socket_t::impl::rdma_recv(शून्य * data, माप_प्रकार size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    माप_प्रकार rem = size;
    पर्यंत (rem > 0) {
        संरचना ibv_wc wc;
        जर (!rdma_poll(c->rcq, &wc)) परत असत्य;

        पूर्णांक slot = (पूर्णांक)wc.wr_id;
        माप_प्रकार got = wc.byte_len;
        स_नकल(dst, c->rx_slot(slot), got);

        जर (!c->post_rx(slot)) परत असत्य;

        dst += got;
        rem -= got;
    }
    परत सत्य;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

तार्किक socket_t::impl::send_data(स्थिरांक शून्य * data, माप_प्रकार size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    जर (use_rdma) {
        परत rdma->send(data, size);
    }
#किंवाजर defined(GGML_RPC_RDMA)
    जर (use_rdma) {
        परत rdma_send(data, size);
    }
#endif
    माप_प्रकार bytes_sent = 0;
    पर्यंत (bytes_sent < size) {
        माप_प्रकार size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (स्थिरांक अक्षर *)data + bytes_sent, size_to_send, 0);
        जर (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            परत असत्य;
        }
        bytes_sent += (माप_प्रकार)n;
    }
    परत सत्य;
}

तार्किक socket_t::impl::recv_data(शून्य * data, माप_प्रकार size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    जर (use_rdma) {
        परत rdma->recv(data, size);
    }
#किंवाजर defined(GGML_RPC_RDMA)
    जर (use_rdma) {
        परत rdma_recv(data, size);
    }
#endif
    माप_प्रकार bytes_recv = 0;
    पर्यंत (bytes_recv < size) {
        माप_प्रकार size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (अक्षर *)data + bytes_recv, size_to_recv, 0);
        जर (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            परत असत्य;
        }
        जर (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            परत असत्य;
        }
        bytes_recv += (माप_प्रकार)n;
    }
    परत सत्य;
}

शून्य socket_t::impl::get_caps(uint8_t * local_caps) {
    स_रखो(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    जर (std::दो_पर्या("GGML_RPC_NO_RDMA")) {
        परत;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    स्वतः target_gid = rdma_build_target_gid();
    जर (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    जर (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  अन्यथा
    rdma_local = {};
    जर (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        स_नकल(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        स_नकल(local_caps, &rc, आकार(rc));
    } अन्यथा {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

शून्य socket_t::impl::update_caps(स्थिरांक uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    तार्किक remote_rdma = असत्य;
    साठी (माप_प्रकार i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    जर (!rdma || !remote_rdma) {
        जर (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        परत;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    तार्किक activated = rdma->activate(remote_caps);
#  अन्यथा
    rdma_caps rc = {};
    स_नकल(&rc, remote_caps, आकार(rc));
    तार्किक activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    जर (activated) {
        use_rdma = सत्य;
    } अन्यथा {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#अन्यथा
    (शून्य)remote_caps;
#endif // GGML_RPC_RDMA
}

तार्किक socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    जर (use_rdma) {
        परत rdma->flush();
    }
#endif
    परत सत्य;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = पूर्वनिर्धारित;

तार्किक socket_t::send_data(स्थिरांक शून्य * data, माप_प्रकार size) {
    परत pimpl->send_data(data, size);
}

तार्किक socket_t::recv_data(शून्य * data, माप_प्रकार size) {
    परत pimpl->recv_data(data, size);
}

तार्किक socket_t::flush() {
    परत pimpl->flush();
}

शून्य socket_t::get_caps(uint8_t * local_caps) {
    परत pimpl->get_caps(local_caps);
}

शून्य socket_t::update_caps(स्थिरांक uint8_t * remote_caps) {
    परत pimpl->update_caps(remote_caps);
}

स्थिरांक अक्षर * socket_t::transport_name() स्थिरांक {
    परत pimpl->use_rdma ? "RDMA" : "TCP";
}

स्थिर तार्किक is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    परत sockfd != INVALID_SOCKET;
#अन्यथा
    परत sockfd >= 0;
#endif
}

स्थिर तार्किक set_no_delay(sockfd_t sockfd) {
    पूर्णांक flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    पूर्णांक ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (अक्षर *)&flag, आकार(पूर्णांक));
    परत ret == 0;
}

स्थिर तार्किक set_reuse_addr(sockfd_t sockfd) {
    पूर्णांक flag = 1;
    पूर्णांक ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (अक्षर *)&flag, आकार(पूर्णांक));
    परत ret == 0;
}

socket_ptr socket_t::accept() {
    स्वतः client_socket_fd = ::accept(pimpl->fd, रिक्त, रिक्त);
    जर (!is_valid_fd(client_socket_fd)) {
        परत nullptr;
    }
    जर (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        परत nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    परत socket_ptr(नवीन socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(स्थिरांक अक्षर * host, पूर्णांक port) {
    स्वतः sockfd = socket(AF_INET, SOCK_STREAM, 0);
    जर (!is_valid_fd(sockfd)) {
        परत nullptr;
    }
    जर (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        परत nullptr;
    }
    जर (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        परत nullptr;
    }
    संरचना sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    जर (bind(sockfd, (संरचना sockaddr *) &serv_addr, आकार(serv_addr)) < 0) {
        परत nullptr;
    }
    जर (listen(sockfd, 1) < 0) {
        परत nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    परत socket_ptr(नवीन socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(स्थिरांक अक्षर * host, पूर्णांक port) {
    स्वतः sockfd = socket(AF_INET, SOCK_STREAM, 0);
    जर (!is_valid_fd(sockfd)) {
        परत nullptr;
    }
    जर (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        परत nullptr;
    }
    संरचना sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    संरचना hostent * server = gethostbyname(host);
    जर (server == रिक्त) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        परत nullptr;
    }
    स_नकल(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    जर (::connect(sockfd, (संरचना sockaddr *)&addr, आकार(addr)) < 0) {
        परत nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    परत socket_ptr(नवीन socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
स्थिर std::mutex g_rpc_transport_mu;
स्थिर तार्किक g_rpc_transport_wsa_started = असत्य;
#endif

तार्किक rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    जर (g_rpc_transport_wsa_started) {
        परत सत्य;
    }
    WSADATA wsaData;
    पूर्णांक res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    जर (res != 0) {
        परत असत्य;
    }
    g_rpc_transport_wsa_started = सत्य;
    परत सत्य;
#अन्यथा
    परत सत्य;
#endif
}

शून्य rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    जर (!g_rpc_transport_wsa_started) {
        परत;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = असत्य;
#endif
}
