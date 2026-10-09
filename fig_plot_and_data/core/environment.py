"""
核心功能：
-构建适用于Ising机的耦合矩阵J和横场向量h,用于密集型WLAN的高速信道分配
-本文的所有的物理层和MAC层参数适用于802.11g通信协议
-动态 UE 随机生成，并关联到最近的 AP
-计算出系统实际吞吐量、理论最大吞吐量
-导出相关矩阵
"""

# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple

class WLANIsingModel:
    def __init__(
        self,
        num_ap=20,
        num_ch=3,
        num_ue_per_ap=5,
        range_ap_ue=10,
        seed=None,
        max_dist=150,
        ap_seed=None,
        ue_seed=None,
        carrier_sense_range_m=70.0,
    ):
        """
        初始化 WLAN 伊辛模型仿真环境。
        所有的可变参数现在都是实例属性，不同实例互不干扰。
        """
        # ==================== 网络拓扑配置 ====================
        self.NUM_AP = num_ap #AP数量
        self.NUM_CH = num_ch #可用信道数量
        self.NUM_UE_PER_AP = num_ue_per_ap #平均每个AP关联的用户设备UE数量 UE密度
        self.TOTAL_UE=int(self.NUM_UE_PER_AP*self.NUM_AP)
        self.RANGE_AP_UE = range_ap_ue #每个AP关联的用户设备UE分布在AP的直径范围
        self.MAX_DIST = max_dist #模拟室内的矩阵边长
        if seed is not None:
            if ap_seed is None:
                ap_seed = seed
            if ue_seed is None:
                ue_seed = seed
        self.AP_SEED = 15 if ap_seed is None else int(ap_seed)
        self.UE_SEED = 16 if ue_seed is None else int(ue_seed)

        # ==================== 物理层参数 ====================
        self.BANDWIDTH_HZ = 20e6 #信道带宽（Hz),20MHz是wifi 2.4G的典型数值
        self.TX_POWER_DBM = 20.0 #AP发射功率（dBm)
        self.NOISE_POWER_DBM_HZ = -174.0 #噪声功率谱密度（dBm/Hz)
        self.PATH_LOSS_EXPONENT = 3.0 #路径损耗指数（室内环境为3）
        self.FREQUENCY_HZ = 2.4e9 #工作频率（2.4G wifi Hz)
        self.CARRIER_SENSE_RANGE_M = float(carrier_sense_range_m) #载波监听范围(m)

        # ==================== MAC层参数 ====================
        self.PAYLOAD_LEN_BITS = 8184 #数据载荷长度（比特）
        self.PHY_HEADER_LEN_BITS = 128 #物理层头部长度
        self.MAC_HEADER_LEN_BITS = 272 #MAC层头部长度
        self.SLOT_TIME_US = 5.0 #时隙时间（5ms)
        self.DIFS_US = 28.0 #分布式帧间间隔（微秒）
        self.SIFS_US = 128.0 # 短帧间间隔(微秒)
        self.ACK_LEN_BITS = 112 # ACK帧长度(比特)
        self.PROP_DELAY_US = 1.0 # 传播延迟(微秒)
        self.MIN_CONTENTION_WINDOW = 31 # 最小竞争窗口大小
        self.BACKOFF_STAGE_MAX = 4 # 最大退避阶段数

        # ==================== 伊辛模型权重 ====================
        self.V1_SCALE = 1.0 # 跨AP吞吐量项的缩放因子
        self.V2_SCALE = 100.0 # 同AP信道独占项的缩放因子

        # ==================== 网络拓扑 ====================
        # 初始化时自动生成位置和距离矩阵
        self.DIST_AP_UE, self.DIST_AP_UE_CROSS, self.AP_POSITIONS, self.UE_POSITIONS = \
            self._generate_synthetic_distances(self.AP_SEED, self.UE_SEED)


    def _generate_synthetic_distances(self, ap_seed: int, ue_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
           生成合成的AP-UE距离数据（用于仿真，实际场景需替换为真实测量值）
           参数：
               num_ap: AP数量
               num_ue_per_ap: 每个AP的UE数量
               max_dist: AP分布的最大范围(m)
               range_AP_UE:AP周围UE分布的最大距离
           返回：
               dist_ap_ue: (num_ap, num_ue_per_ap) - 每个AP到自身UE的距离
               dist_ap_ue_cross: (num_ap, num_ap, num_ue_per_ap, num_ue_per_ap) - AP间UE的交叉距离
               ap_positions: (num_ap, 2) - AP的二维坐标
               ue_positions: (num_ap, num_ue_per_ap, 2) - UE的二维坐标
        """
        ap_rng = np.random.default_rng(ap_seed)
        ue_rng = np.random.default_rng(ue_seed)
        # 生成AP的随机位置（0~max_dist范围内）
        ap_positions = ap_rng.uniform(0, self.MAX_DIST, size=(self.NUM_AP, 2))
        # 初始化UE位置：围绕对应AP的正态分布
        global_ue_positions = ue_rng.uniform(0,self.MAX_DIST,size=(self.TOTAL_UE, 2))
        #计算每个UE到每个AP的距离
        distances_to_aps=np.linalg.norm(global_ue_positions[:,np.newaxis,:]-ap_positions[np.newaxis,:,:], axis=2)
        #判断归属于哪个AP
        ue_to_ap=np.argmin(distances_to_aps,axis=1)
        #遍历每一个AP,找出他关联的所有的UE索引
        ap_ue_indices=[np.where(ue_to_ap==ap_idx)[0] for ap_idx in range(self.NUM_AP)]
        #统计每个AP下的UE的数量
        ue_counts_per_ap = np.array([len(indices) for indices in ap_ue_indices], dtype=int)
        #找到所有AP中的最多的UE数量
        max_ue_count = int(ue_counts_per_ap.max()) if ue_counts_per_ap.size > 0 else 0
        #存储每个AP下的UE坐标 形状（AP总数，最大UE数，2）
        ue_positions = np.full((self.NUM_AP, max_ue_count, 2), np.nan, dtype=float)
        #存储AP到自身UE的距离
        dist_ap_ue = np.full((self.NUM_AP, max_ue_count), np.inf, dtype=float)

        #存储跨AP的UE之间的距离 形状（AP数，AP数，最大UE数，最大UE数）
        dist_ap_ue_cross = np.full(
            (self.NUM_AP, self.NUM_AP, max_ue_count, max_ue_count),
            np.inf,
            dtype=float,
        )
        #计算AP内的ue距离
        for ap_idx, ue_indices in enumerate(ap_ue_indices):
            count = len(ue_indices)
            if count == 0:
                continue

            local_positions = global_ue_positions[ue_indices]
            ue_positions[ap_idx, :count] = local_positions
            dist_ap_ue[ap_idx, :count] = np.linalg.norm(local_positions - ap_positions[ap_idx], axis=1)

        #计算不同接入点（AP）下，所有用户设备（UE）两两之间的距离计算
        for ap_i in range(self.NUM_AP):
            count_i = ue_counts_per_ap[ap_i]
            if count_i == 0:
                continue
            pos_i = ue_positions[ap_i, :count_i]
            for ap_k in range(self.NUM_AP):
                if ap_i == ap_k:
                    continue
                count_k = ue_counts_per_ap[ap_k]
                if count_k == 0:
                    continue
                pos_k = ue_positions[ap_k, :count_k]
                cross = np.linalg.norm(pos_i[:, np.newaxis, :] - pos_k[np.newaxis, :, :], axis=2)
                dist_ap_ue_cross[ap_i, ap_k, :count_i, :count_k] = cross

        self.GLOBAL_UE_POSITIONS = global_ue_positions
        self.UE_TO_AP = ue_to_ap
        self.AP_UE_INDICES = ap_ue_indices
        self.UE_COUNTS_PER_AP = ue_counts_per_ap
        self.MAX_UE_COUNT = max_ue_count
        return dist_ap_ue, dist_ap_ue_cross, ap_positions, ue_positions



    # ==================== 物理层MAC配置 ====================
    def path_loss_db(self, d_m: float) -> float:
        """
        计算路径损耗(dB)，基于自由空间路径损耗模型（含频率修正）
        参数：d_m - 传播距离(m)
        返回：路径损耗值(dB)（负数，值越小损耗越大）
        """
        d_safe = max(float(d_m), 1e-6)
        f_mhz = self.FREQUENCY_HZ / 1e6
        pl = 20 * np.log10(f_mhz) + 10 * self.PATH_LOSS_EXPONENT * np.log10(d_safe) - 27.55
        return -pl

    def received_power_dbm(self, tx_power_dbm: float, d_m: float) -> float:
        """
        计算接收功率(dBm)
        参数：
            tx_power_dbm: 发射功率(dBm)
            d_m: 传播距离(m)
        返回：接收功率(dBm)
        """
        return tx_power_dbm + self.path_loss_db(d_m)

    def interference_power_dbm(self, tx_power_dbm: float, dist_list_m: List[float], same_channel: bool) -> float:
        if not same_channel or not dist_list_m:
            return -150.0
        valid_dists = [d for d in dist_list_m if (d.mean() if isinstance(d, np.ndarray) else d) >= self.CARRIER_SENSE_RANGE_M]
        if not valid_dists:
            return -150.0
        powers_dbm = [self.received_power_dbm(tx_power_dbm, d) for d in valid_dists]
        powers_linear = 10.0 ** (np.array(powers_dbm) / 10.0)
        total_power_linear = np.sum(powers_linear)
        return 10.0 * np.log10(total_power_linear) if total_power_linear > 0 else -150.0

    def phy_bit_rate_bps(self, snr_linear: float) -> float:
        """
        修正：添加物理层速率上限（符合WiFi 802.11g 20MHz信道实际）
        """
        max_phy_rate = 54e6
        shannon_rate = self.BANDWIDTH_HZ * np.log2(1.0 + snr_linear)
        return min(shannon_rate, max_phy_rate)

    def snr_linear(self, ps_dbm: float, pi_dbm: float, noise_dbm_hz: float) -> float:
        """
        计算线性域信噪比（SNR = 信号功率 / (干扰功率 + 噪声功率)）
        参数：
            ps_dbm: 有用信号功率(dBm)
            pi_dbm: 干扰功率(dBm)
            noise_dbm_hz: 噪声功率谱密度(dBm/Hz)
        返回：线性域信噪比
        """
        ps_w = 10.0 ** ((ps_dbm - 30.0) / 10.0)
        pi_w = 10.0 ** ((pi_dbm - 30.0) / 10.0)
        noise_w_hz = 10.0 ** ((noise_dbm_hz - 30.0) / 10.0)
        total_noise_w = noise_w_hz * self.BANDWIDTH_HZ
        return ps_w / (pi_w + total_noise_w)

    def mac_payload_ratio(self) -> float:
        """计算MAC层有效载荷占比（有效数据 / 总传输数据）"""
        total_bits = self.PHY_HEADER_LEN_BITS + self.MAC_HEADER_LEN_BITS + self.PAYLOAD_LEN_BITS
        return self.PAYLOAD_LEN_BITS / total_bits

    def transmission_time_success_bps(self, r_bps: float) -> float:
        """
        计算成功传输的总时间(秒)（含头部、ACK、帧间隔）
        参数：r_bps - 物理层传输速率(bps)
        返回：成功传输时间(秒)
        """
        bits_time_s = (self.PHY_HEADER_LEN_BITS + self.MAC_HEADER_LEN_BITS + self.PAYLOAD_LEN_BITS) / r_bps
        return (bits_time_s + self.SIFS_US * 1e-6 + self.DIFS_US * 1e-6 + self.ACK_LEN_BITS / r_bps + 2 * self.PROP_DELAY_US * 1e-6)

    def transmission_time_collision_bps(self, r_bps: float) -> float:
        """
        计算碰撞后的传输时间(秒)（无ACK，仅数据传输+帧间隔）
        参数：r_bps - 物理层传输速率(bps)
        返回：碰撞传输时间(秒)
        """
        bits_time_s = (self.PHY_HEADER_LEN_BITS + self.MAC_HEADER_LEN_BITS + self.PAYLOAD_LEN_BITS) / r_bps
        return bits_time_s + self.DIFS_US * 1e-6 + self.PROP_DELAY_US * 1e-6

    def compute_nijkla(self, i: int, j: int, k: int, l: int, a: int) -> int:
        """
        计算竞争用户数n_ijkla（用于CSMA/CA退避概率计算）
        参数：
            i,k: AP编号
            j,l: 信道编号
            a: UE编号（属于AP i）
            dist_ap_ue: AP-UE距离矩阵
            dist_ap_ue_cross: 跨AP UE距离矩阵
        返回：竞争用户数量
        """
        base = int(self.UE_COUNTS_PER_AP[i])
        if base == 0:
            return 0
        if j != l: return base
        count_k = int(self.UE_COUNTS_PER_AP[k])
        if count_k == 0:
            return base

        dists_to_ue_a = self.DIST_AP_UE_CROSS[i, k, a, :count_k]
        within_rsc = np.sum(dists_to_ue_a <= self.CARRIER_SENSE_RANGE_M)
        return int(base + within_rsc)

    def compute_tau(self, n: int) -> float:
        """
        迭代计算CSMA/CA的传输概率τ（基于退避窗口模型）
        参数：n - 竞争用户数
        返回：传输概率τ（0~1）
        """
        if n == 0: return 0.0
        tau = 0.01
        for _ in range(10):
            p = 1.0 - (1.0 - tau) ** (n - 1) if n > 1 else 0.0
            denom = (1.0 - 2.0 * p) * (self.MIN_CONTENTION_WINDOW + 1) + p * self.MIN_CONTENTION_WINDOW * (1.0 - (2.0 * p) ** self.BACKOFF_STAGE_MAX)
            if abs(denom) < 1e-12: break
            tau_new = 2.0 * (1.0 - 2.0 * p) / denom
            if abs(tau_new - tau) < 1e-6: break
            tau = tau_new
        return max(1e-6, min(0.99, tau))

    def compute_ptr_ps(self, n: int) -> Tuple[float, float]:
        """
        计算传输概率P_tr和成功传输概率P_s
        参数：n - 竞争用户数
        返回：(P_tr, P_s)
        """
        tau = self.compute_tau(n)
        ptr = 1.0 - (1.0 - tau) ** n
        if ptr < 1e-12: return 0.0, 0.0
        ps = (n * tau * (1.0 - tau) ** (n - 1)) / ptr
        return ptr, ps

    def compute_Tsuccess_Tfail(self, n: int, r_bps: float) -> Tuple[float, float]:
        """
        计算成功/失败传输的有效时间
        参数：
            n - 竞争用户数
            r_bps - 物理层传输速率(bps)
        返回：(T_success, T_fail) - 成功/失败传输时间(秒)
        """
        Ts = self.transmission_time_success_bps(r_bps)
        Tc = self.transmission_time_collision_bps(r_bps)
        ptr, ps = self.compute_ptr_ps(n)
        Tsuccess = ptr * ps * Ts
        Tfail = (1.0 - ptr) * (self.SLOT_TIME_US * 1e-6) + ptr * (1.0 - ps) * Tc
        return Tsuccess, Tfail

    def compute_Sijkla(self, i: int, j: int, k: int, l: int, a: int, b: int) -> float:
        """
        计算吞吐量项S_ijkla（AP i(信道j)的UE a 与 AP k(信道l)的UE b 之间的吞吐量贡献）
        参数：
            i,k: AP编号
            j,l: 信道编号
            a: AP i的UE编号
            b: AP k的UE编号
            dist_ap_ue: AP-UE距离矩阵
            dist_ap_ue_cross: 跨AP UE距离矩阵
        返回：吞吐量贡献值
        """
        count_i = int(self.UE_COUNTS_PER_AP[i])
        count_k = int(self.UE_COUNTS_PER_AP[k])
        if a >= count_i or b >= count_k:
            return 0.0
        # 计算有用信号功率
        ps_dbm = self.received_power_dbm(self.TX_POWER_DBM, self.DIST_AP_UE[i, a])
        # 计算干扰功率（仅同信道）
        #dists_to_b = self.DIST_AP_UE_CROSS[i, k, a, :count_k].tolist()
        #pi_dbm = self.interference_power_dbm(self.TX_POWER_DBM, dists_to_b, j == l)

        # 不要使用 DIST_AP_UE_CROSS，改用 AP k 到 UE a 的距离
        dist_ap_k_ue_a = np.linalg.norm(self.AP_POSITIONS[k] - self.UE_POSITIONS[i, a])
        pi_dbm = self.interference_power_dbm(self.TX_POWER_DBM, [dist_ap_k_ue_a], j == l)

        # 计算信噪比和传输速率
        snr_lin = self.snr_linear(ps_dbm, pi_dbm, self.NOISE_POWER_DBM_HZ)
        r_bps = self.phy_bit_rate_bps(snr_lin)
        # 计算竞争用户数和有效传输时间
        n = self.compute_nijkla(i, j, k, l, a)
        Tsuccess, Tfail = self.compute_Tsuccess_Tfail(n, r_bps)
        # 有效吞吐量 = 载荷占比 × 成功时间 / 总时间（成功+失败）
        Rpayload = self.mac_payload_ratio()
        if Tsuccess + Tfail < 1e-12 or n <= 0:
            return 0.0
        return Rpayload * Tsuccess / (Tsuccess + Tfail) / n

    #=========================================================
    def _build_throughput_aware_J_h(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建伊辛模型的耦合矩阵J和场向量h（核心函数）
        参数：
            dist_ap_ue: AP-UE距离矩阵
            dist_ap_ue_cross: 跨AP UE距离矩阵
            v1: 跨AP吞吐量项缩放因子
            v2: 同AP信道独占项缩放因子
            num_ap: AP数量
            num_ch: 信道数量
            num_ue_per_ap: 每个AP的UE数量
        返回：
            J_matrix: (N×N) 耦合矩阵（N=num_ap×num_ch）
            h_vector: (N,) 场向量
        """
        N = self.NUM_AP * self.NUM_CH
        J_matrix = np.zeros((N, N))
        h_vector = np.zeros(N)

        for i in range(self.NUM_AP):
            count_i = int(self.UE_COUNTS_PER_AP[i])
            for j in range(self.NUM_CH):
                idx = i * self.NUM_CH + j # 当前神经元索引（AP i + 信道 j）
                for k in range(self.NUM_AP):
                    count_k = int(self.UE_COUNTS_PER_AP[k])
                    for l in range(self.NUM_CH):
                        jdx = k * self.NUM_CH + l # 目标神经元索引（AP k + 信道 l）
                        delta_ik = 1.0 if i == k else 0.0
                        delta_jl = 1.0 if j == l else 0.0

                        # 计算跨AP吞吐量和（仅i≠k时）
                        sum_S = 0.0
                        if i != k and count_i > 0 and count_k > 0:
                            for a in range(count_i):
                                for b in range(count_k):
                                    s1 = self.compute_Sijkla(i, j, k, l, a, b)
                                    s2 = self.compute_Sijkla(k, l, i, j, b, a)
                                    sum_S += (s1 + s2)

                        # 计算耦合权重W_ijkl（论文Eq.14）
                        wij = -2.0 * (self.V2_SCALE * delta_ik * (1.0 - delta_jl) + self.V1_SCALE * (1.0 - delta_ik) * delta_jl * sum_S)
                        J_matrix[idx, jdx] = wij / 2.0
                        J_matrix[jdx, idx] = J_matrix[idx, jdx]

                # 计算场向量h
                theta_ij = -self.V2_SCALE
                sum_w_over_kl = 0.0
                for k in range(self.NUM_AP):
                    for l in range(self.NUM_CH):
                        jdx = k * self.NUM_CH + l
                        sum_w_over_kl += J_matrix[idx, jdx]
                lambda_ij = theta_ij - sum_w_over_kl
                h_vector[idx] = lambda_ij

        return J_matrix, h_vector
    def build_J_h(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._build_throughput_aware_J_h()

    def _contention_count_for_ue(self, ap_i: int, ch_j: int, ue_a: int, X: np.ndarray) -> int:
        count_i = int(self.UE_COUNTS_PER_AP[ap_i])
        if count_i == 0:
            return 0

        n = count_i
        ue_pos = self.UE_POSITIONS[ap_i, ue_a]
        for ap_k in range(self.NUM_AP):
            if ap_k == ap_i:
                continue
            ch_k = np.argmax(X[ap_k * self.NUM_CH: (ap_k + 1) * self.NUM_CH])
            if ch_k != ch_j:
                continue
            count_k = int(self.UE_COUNTS_PER_AP[ap_k])
            if count_k == 0:
                continue
            for ue_b in range(count_k):
                dist = np.linalg.norm(ue_pos - self.UE_POSITIONS[ap_k, ue_b])
                if dist <= self.CARRIER_SENSE_RANGE_M:
                    n += 1
        return n

    def compute_true_throughput(self, X: np.ndarray) -> float:
        total_tp_mbps = 0.0
        R_payload = self.mac_payload_ratio()

        for ap_i in range(self.NUM_AP):
            count_i = int(self.UE_COUNTS_PER_AP[ap_i])
            if count_i == 0:
                continue

            ch_j = np.argmax(X[ap_i * self.NUM_CH: (ap_i + 1) * self.NUM_CH])
            ap_throughput_bps = 0.0

            for ue_a in range(count_i):
                ps_dbm = self.received_power_dbm(self.TX_POWER_DBM, self.DIST_AP_UE[ap_i, ue_a])
                interferers = []
                ue_pos = self.UE_POSITIONS[ap_i, ue_a]

                for ap_k in range(self.NUM_AP):
                    if ap_k == ap_i:
                        continue
                    ch_l = np.argmax(X[ap_k * self.NUM_CH: (ap_k + 1) * self.NUM_CH])
                    if ch_l == ch_j:
                        d = float(np.linalg.norm(self.AP_POSITIONS[ap_k] - ue_pos))
                        interferers.append(d)

                pi_dbm = self.interference_power_dbm(self.TX_POWER_DBM, interferers, True)
                snr = self.snr_linear(ps_dbm, pi_dbm, self.NOISE_POWER_DBM_HZ)
                phy_rate_bps = self.phy_bit_rate_bps(snr)

                n = self._contention_count_for_ue(ap_i, ch_j, ue_a, X)
                Ts, Tf = self.compute_Tsuccess_Tfail(n, phy_rate_bps)
                utilization = Ts / (Ts + Tf) if (Ts + Tf) > 1e-12 else 0.0
                ap_throughput_bps += R_payload * utilization * phy_rate_bps / max(n, 1)

            total_tp_mbps += ap_throughput_bps / 1e6
        return total_tp_mbps


    def compute_true_theoretical_max(self) -> float:
        """
        理论最大吞吐量（无干扰、无竞争）
        """
        max_tp_mbps = 0.0
        R_payload = self.mac_payload_ratio()

        for ap_i in range(self.NUM_AP):
            count_i = int(self.UE_COUNTS_PER_AP[ap_i])
            if count_i == 0:
                continue

            ap_throughput_bps = 0.0
            for ue_a in range(count_i):
                ps_dbm = self.received_power_dbm(self.TX_POWER_DBM, self.DIST_AP_UE[ap_i, ue_a])
                snr = self.snr_linear(ps_dbm, -150.0, self.NOISE_POWER_DBM_HZ)
                phy_rate_bps = self.phy_bit_rate_bps(snr)
                ap_throughput_bps += R_payload * phy_rate_bps / count_i
            max_tp_mbps += ap_throughput_bps / 1e6
        return max_tp_mbps

    def normalize_and_export(self, J, h, output_dir="./hardware_output", prefix="wlan_cim_norm"):
        max_val = max(np.abs(J).max(), np.abs(h).max())
        if max_val == 0: return J, h
        J_norm = J / max_val
        h_norm = h / max_val
        return J_norm, h_norm


