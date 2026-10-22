#अन्तर्भाव "transport.h"
#अन्तर्भाव "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#अण्णहा
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
प्रकारनिर्देश SOCKET sockfd_t;
using ssize_t = __int64;
#अण्णहा
प्रकारनिर्देश पूर्णाङ्क sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
स्थिर constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#जइ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
स्थिर constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
स्थिर constexpr पूर्णाङ्क    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
स्थिर constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

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
    पूर्णाङ्क             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(पूर्णाङ्क i) ध्रुव {
        प्रत्यागम static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    तार्किक post_rx(पूर्णाङ्क i) {
        संरचना ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        संरचना ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.अङ्कः_sge = 1;
        प्रत्यागम ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        जइ (tx_mr) ibv_dereg_mr(tx_mr);
        जइ (rx_mr) ibv_dereg_mr(rx_mr);
        मुक्त(tx_buf);
        मुक्त(rx_buf);
        जइ (qp)  ibv_destroy_qp(qp);
        जइ (scq) ibv_destroy_cq(scq);
        जइ (rcq) ibv_destroy_cq(rcq);
        जइ (ch)  ibv_destroy_comp_channel(ch);
        जइ (pd)  ibv_dealloc_pd(pd);
        जइ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
संरचना rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    पूर्णाङ्क      gid_idx = 0;
    गणना ibv_mtu path_mtu = IBV_MTU_1024;
};

संरचना rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(परिमाण(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

संरचना socket_t::impl {
    impl(sockfd_t fd) : use_rdma(मिच्छ), fd(fd) {}
    ~impl();
    तार्किक send_data(ध्रुव शून्य * data, size_t size);
    तार्किक recv_data(शून्य * data, size_t size);
    तार्किक flush();
    शून्य get_caps(uint8_t * local_caps);
    शून्य update_caps(ध्रुव uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  अण्णहा
    तार्किक rdma_probe();
    तार्किक rdma_send(ध्रुव शून्य * data, size_t size);
    तार्किक rdma_recv(शून्य * data, size_t size);
    तार्किक tcp_peer_closed();
    तार्किक rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ध्रुव uint8_t * remote_gid);
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
    LOG_DBG("[%s] closing socket %d\n", __func__, एतत्->fd);
#ifdef _WIN32
    जइ (fd != INVALID_SOCKET) closesocket(एतत्->fd);
#अण्णहा
    जइ (fd >= 0) close(एतत्->fd);
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
    socklen_t addr_len = परिमाण(addr);
    जइ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        प्रत्यागम std::nullopt;
    }
    rdma_gid_t target = {};
    जइ (addr.ss_family == AF_INET) {
        ध्रुव auto * a = reinterpret_cast<ध्रुव sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        प्रत्यागम target;
    }
    जइ (addr.ss_family == AF_INET6) {
        ध्रुव auto * a = reinterpret_cast<ध्रुव sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        प्रत्यागम target;
    }
    प्रत्यागम std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

तार्किक socket_t::impl::tcp_peer_closed() {
    जइ (fd < 0) प्रत्यागम मिच्छ;
#ifndef _WIN32
    संरचना pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    पूर्णाङ्क r = poll(&pfd, 1, 0);
    प्रत्यागम r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#अण्णहा
    प्रत्यागम मिच्छ;
#endif
}

तार्किक socket_t::impl::rdma_probe() {
    ध्रुव वर्ण * dev_env = std::getenv("GGML_RDMA_DEV");
    ध्रुव वर्ण * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    जइ (!target_gid) {
        प्रत्यागम मिच्छ;
    }

    ध्रुव uint8_t ib_port = 1;
    पूर्णाङ्क अङ्कः_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&अङ्कः_devs);
    जइ (!devs || अङ्कः_devs == 0) प्रत्यागम मिच्छ;

    ibv_context * ibctx = nullptr;
    ध्रुव वर्ण * matched_dev = nullptr;
    पूर्णाङ्क gid_idx = gid_env ? atoi(gid_env) : -1;
    पूर्णाङ्क gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    प्रति (पूर्णाङ्क d = 0; d < अङ्कः_devs; d++) {
        ध्रुव वर्ण * dn = ibv_get_device_name(devs[d]);
        जइ (dev_env && strcmp(dev_env, dn) != 0) अनुवर्त;

        ibv_context * ctx = ibv_open_device(devs[d]);
        जइ (!ctx) अनुवर्त;

        ibv_port_attr pa;
        जइ (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); अनुवर्त; }

        पूर्णाङ्क found_gid = gid_idx;
        पूर्णाङ्क found_version = IBV_GID_TYPE_IB;
        जइ (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            पूर्णाङ्क v2_idx = -1;
            पूर्णाङ्क v1_idx = -1;
            प्रति (पूर्णाङ्क i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                जइ (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) अनुवर्त;
                जइ (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) अनुवर्त;
                जइ (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } अण्णहा जइ (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            जइ (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } अण्णहा जइ (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } अण्णहा {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            जइ (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        जइ (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            भङ्ग;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    जइ (!ibctx) प्रत्यागम मिच्छ;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    जइ (!rdma->pd) प्रत्यागम मिच्छ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    जइ (!rdma->scq || !rdma->rcq) प्रत्यागम मिच्छ;

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
    जइ (!rdma->qp) प्रत्यागम मिच्छ;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    जइ (!rdma->tx_buf || !rdma->rx_buf) प्रत्यागम मिच्छ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    जइ (!rdma->tx_mr || !rdma->rx_mr) प्रत्यागम मिच्छ;

    ibv_gid local_gid;
    जइ (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) प्रत्यागम मिच्छ;

    rdma_local.qpn = rdma->qp->qp_अङ्कः;
    rdma_local.psn = rdma->qp->qp_अङ्कः & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    ध्रुव वर्ण * ver_str = "";
    जइ (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } अण्णहा जइ (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    प्रत्यागम सच्च;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
तार्किक socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ध्रुव uint8_t * remote_gid) {
    // RESET -> INIT
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_अङ्कः        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        जइ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            प्रत्यागम मिच्छ;
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < RDMA_RX_DEPTH; i++) {
        जइ (!rdma->post_rx(i)) प्रत्यागम मिच्छ;
    }

    // INIT -> RTR
    {
        संरचना ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_अङ्कः        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_अङ्कः   = rdma_local.ib_port;
        जइ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            प्रत्यागम मिच्छ;
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
        जइ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            प्रत्यागम मिच्छ;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    प्रत्यागम सच्च;
}

// Sleep until the completion channel has an event or the TCP peer closes.
तार्किक socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    संरचना pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    जइ (poll(pfds, 2, -1) < 0) {
        प्रत्यागम errno == EINTR;
    }
    जइ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        प्रत्यागम मिच्छ;
    }
    जइ (pfds[0].revents & POLLIN) {
        संरचना ibv_cq * ev_cq  = nullptr;
        शून्य          * ev_ctx = nullptr;
        जइ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            प्रत्यागम मिच्छ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    प्रत्यागम सच्च;
}

तार्किक socket_t::impl::rdma_poll(संरचना ibv_cq * cq, संरचना ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    तार्किक armed = मिच्छ;
    प्रति (uint64_t s = 1; ; s++) {
        पूर्णाङ्क n = ibv_poll_cq(cq, 1, wc);
        जइ (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            जइ (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            प्रत्यागम wc->status == IBV_WC_SUCCESS;
        }
        जइ (n < 0) प्रत्यागम मिच्छ;
        जइ (armed) {
            // armed and still empty: sleep until the next completion
            जइ (!rdma_wait_event()) {
                प्रत्यागम मिच्छ;
            }
            armed = मिच्छ;
            अनुवर्त;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        जइ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            जइ (ibv_req_notify_cq(cq, 0) != 0) {
                प्रत्यागम मिच्छ;
            }
            armed = सच्च;
            अनुवर्त;
        }
        जइ ((s & 0xFFFFF) == 0) {
            जइ (tcp_peer_closed()) {
                प्रत्यागम मिच्छ;
            }
        }
    }
}

तार्किक socket_t::impl::rdma_send(ध्रुव शून्य * data, size_t size) {
    rdma_conn * c = rdma.get();
    ध्रुव uint8_t * src = (ध्रुव uint8_t *)data;
    size_t rem = size;
    जाव (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        संरचना ibv_sge sge = {};
        संरचना ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.अङ्कः_sge = 1;

        जइ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } अण्णहा {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        जइ (ibv_post_send(c->qp, &wr, &bad) != 0) प्रत्यागम मिच्छ;
        c->last_active = std::chrono::steady_clock::now();
        संरचना ibv_wc wc;
        जइ (!rdma_poll(c->scq, &wc)) प्रत्यागम मिच्छ;

        src += chunk;
        rem -= chunk;
    }
    प्रत्यागम सच्च;
}

तार्किक socket_t::impl::rdma_recv(शून्य * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    जाव (rem > 0) {
        संरचना ibv_wc wc;
        जइ (!rdma_poll(c->rcq, &wc)) प्रत्यागम मिच्छ;

        पूर्णाङ्क slot = (पूर्णाङ्क)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        जइ (!c->post_rx(slot)) प्रत्यागम मिच्छ;

        dst += got;
        rem -= got;
    }
    प्रत्यागम सच्च;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

तार्किक socket_t::impl::send_data(ध्रुव शून्य * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    जइ (use_rdma) {
        प्रत्यागम rdma->send(data, size);
    }
#नोचेद्यदि defined(GGML_RPC_RDMA)
    जइ (use_rdma) {
        प्रत्यागम rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    जाव (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (ध्रुव वर्ण *)data + bytes_sent, size_to_send, 0);
        जइ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            प्रत्यागम मिच्छ;
        }
        bytes_sent += (size_t)n;
    }
    प्रत्यागम सच्च;
}

तार्किक socket_t::impl::recv_data(शून्य * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    जइ (use_rdma) {
        प्रत्यागम rdma->recv(data, size);
    }
#नोचेद्यदि defined(GGML_RPC_RDMA)
    जइ (use_rdma) {
        प्रत्यागम rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    जाव (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (वर्ण *)data + bytes_recv, size_to_recv, 0);
        जइ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            प्रत्यागम मिच्छ;
        }
        जइ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            प्रत्यागम मिच्छ;
        }
        bytes_recv += (size_t)n;
    }
    प्रत्यागम सच्च;
}

शून्य socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    जइ (std::getenv("GGML_RPC_NO_RDMA")) {
        प्रत्यागम;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    जइ (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    जइ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  अण्णहा
    rdma_local = {};
    जइ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, परिमाण(rc));
    } अण्णहा {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

शून्य socket_t::impl::update_caps(ध्रुव uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    तार्किक remote_rdma = मिच्छ;
    प्रति (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    जइ (!rdma || !remote_rdma) {
        जइ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        प्रत्यागम;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    तार्किक activated = rdma->activate(remote_caps);
#  अण्णहा
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, परिमाण(rc));
    तार्किक activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    जइ (activated) {
        use_rdma = सच्च;
    } अण्णहा {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#अण्णहा
    (शून्य)remote_caps;
#endif // GGML_RPC_RDMA
}

तार्किक socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    जइ (use_rdma) {
        प्रत्यागम rdma->flush();
    }
#endif
    प्रत्यागम सच्च;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = स्वभाव;

तार्किक socket_t::send_data(ध्रुव शून्य * data, size_t size) {
    प्रत्यागम pimpl->send_data(data, size);
}

तार्किक socket_t::recv_data(शून्य * data, size_t size) {
    प्रत्यागम pimpl->recv_data(data, size);
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
#ifdef _WIN32
    प्रत्यागम sockfd != INVALID_SOCKET;
#अण्णहा
    प्रत्यागम sockfd >= 0;
#endif
}

स्थिर तार्किक set_no_delay(sockfd_t sockfd) {
    पूर्णाङ्क flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    पूर्णाङ्क ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (वर्ण *)&flag, परिमाण(पूर्णाङ्क));
    प्रत्यागम ret == 0;
}

स्थिर तार्किक set_reuse_addr(sockfd_t sockfd) {
    पूर्णाङ्क flag = 1;
    पूर्णाङ्क ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (वर्ण *)&flag, परिमाण(पूर्णाङ्क));
    प्रत्यागम ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, रिक्त, रिक्त);
    जइ (!is_valid_fd(client_socket_fd)) {
        प्रत्यागम nullptr;
    }
    जइ (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        प्रत्यागम nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    प्रत्यागम socket_ptr(नूतन socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ध्रुव वर्ण * host, पूर्णाङ्क port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    जइ (!is_valid_fd(sockfd)) {
        प्रत्यागम nullptr;
    }
    जइ (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        प्रत्यागम nullptr;
    }
    जइ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        प्रत्यागम nullptr;
    }
    संरचना sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    जइ (bind(sockfd, (संरचना sockaddr *) &serv_addr, परिमाण(serv_addr)) < 0) {
        प्रत्यागम nullptr;
    }
    जइ (listen(sockfd, 1) < 0) {
        प्रत्यागम nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    प्रत्यागम socket_ptr(नूतन socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ध्रुव वर्ण * host, पूर्णाङ्क port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    जइ (!is_valid_fd(sockfd)) {
        प्रत्यागम nullptr;
    }
    जइ (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        प्रत्यागम nullptr;
    }
    संरचना sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    संरचना hostent * server = gethostbyname(host);
    जइ (server == रिक्त) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        प्रत्यागम nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    जइ (::connect(sockfd, (संरचना sockaddr *)&addr, परिमाण(addr)) < 0) {
        प्रत्यागम nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    प्रत्यागम socket_ptr(नूतन socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
स्थिर std::mutex g_rpc_transport_mu;
स्थिर तार्किक g_rpc_transport_wsa_started = मिच्छ;
#endif

तार्किक rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    जइ (g_rpc_transport_wsa_started) {
        प्रत्यागम सच्च;
    }
    WSADATA wsaData;
    पूर्णाङ्क res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    जइ (res != 0) {
        प्रत्यागम मिच्छ;
    }
    g_rpc_transport_wsa_started = सच्च;
    प्रत्यागम सच्च;
#अण्णहा
    प्रत्यागम सच्च;
#endif
}

शून्य rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    जइ (!g_rpc_transport_wsa_started) {
        प्रत्यागम;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = मिच्छ;
#endif
}
