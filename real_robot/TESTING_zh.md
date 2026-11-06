# Panda 真机测试流程（Panda System 4.2.x）

有两个独立的测试：

- **A. Lobby 遥操**：Quest 3 网页里拖 IK marker，真机跟随。不需要 ROS。
- **B. giskard demo**：giskardpy 通过 ros2_control 控制真机。需要 ROS 2 Jazzy。

两个测试不要同时跑。FCI 同一时间只接受一个控制程序。

```
A:  Quest 3 浏览器 ──Wi-Fi──► Franka 电脑: python -m real_robot (9090) ──网线──► Panda
B:  giskard → ros2_control → python3 -m real_robot.ros2 ──网线──► Panda   （全部在 Franka 电脑上）
```

## 0. Franka 电脑（两个测试共用）

### 硬件和系统

- **网络**
  - 一根网线接 Panda **控制柜**的网口，不是机械臂底座上的网口。
  - 这个网口配静态 IP `172.16.0.1/24`，Panda 默认是 `172.16.0.2`。
  - 另一块网卡或 Wi-Fi 接 Quest 所在的局域网。
- **PREEMPT_RT 内核**，libfranka 要求。
  - 检查：`uname -v` 里应该有 `PREEMPT_RT`。
  - 用户要有实时优先级：在 `/etc/security/limits.conf` 里给用户加 `rtprio 99` 和 `memlock unlimited`，或者把用户加进 `realtime` 组。
  - 检查：`ulimit -r` 应该是 `99`。
- **CPU 调频**设成 performance：`sudo cpupower frequency-set -g performance`。
- **Desk**：浏览器打开 `https://172.16.0.2`。
  1. 解锁关节（unlock joints）。
  2. 激活 FCI（Activate FCI）。
  3. 松开外部急停（用户停止按钮），**放在手边**。

### 代码

```bash
git clone -b real-robot-bridge https://github.com/yxzhan/robot-lobby.git ~/robot-lobby
git clone -b dev --recursive https://github.com/yxzhan/ijcai_2026_workshop.git ~/ijcai_2026_workshop   # 只有 B 需要
```

这两个分支要先从开发机 push 上去。

### franky（libfranka 0.9.2 版本）

Panda System 4.2.x 只能用 libfranka 0.9.x。PyPI 上的 franky 是给 FR3 用的新版 libfranka，**不能用**。

从 <https://github.com/TimSchneider42/franky/releases> 下载 v2.0.0 中名字带 `libfranka.0.9.2` 的 wheel：

- `cpXY` 要和你的 Python 版本一致。
- `manylinux_2_34` 需要 Ubuntu 22.04 或更新；Ubuntu 20.04 用 `manylinux_2_28`。

例如 Python 3.12：

```bash
pip install "https://github.com/TimSchneider42/franky/releases/download/v2.0.0/franky_control-2.0.0+libfranka.0.9.2-cp312-cp312-manylinux_2_34_x86_64.whl"
```

franky 1.1.4 也能用，驱动兼容这两个版本。

### 第一次连接：只读，机器人不动

```bash
python3 -c "import franky; r = franky.Robot('172.16.0.2'); print(r.state.q)"
```

- 能打印出 7 个关节角，说明网络、FCI、libfranka 版本都对。
- 报版本不兼容（incompatible library version），说明 franky 的 libfranka 版本装错了。
- 报实时调度相关的错误（realtime / scheduling），说明内核或 rtprio 没配好。

---

## A. Lobby 遥操

### Franka 电脑要装的东西

- Python 3，加 `websockets` 和上面的 franky。不需要 ROS。
- 如果 Lobby 网页也放在这台电脑上：Node 18+（`npm install` 时用）。
  - 也可以放在局域网里另一台电脑上。物理仿真在 Quest 的浏览器里跑，`?local` 模式下 Franka 电脑不跑仿真。

```bash
cd ~/robot-lobby
python3 -m venv .venv && .venv/bin/pip install websockets "<上面的 franky wheel>"
```

### 网页

- `mjcf_assets/` 不在 git 里，要从开发机拷过去，放在 `package.json` 旁边。
- 然后运行：

```bash
npm install
npm start          # http://<这台电脑的局域网 IP>:4007
```

- Quest 浏览器打开 `chrome://flags`，在 "Insecure origins treated as secure" 里填 `http://<IP>:4007`，然后重启浏览器。不做这一步进不了 WebXR。

### 步骤

1. **假机器人先走一遍**，确认网络和网页都正常：
   ```bash
   .venv/bin/python -m real_robot --robot panda --driver fake
   ```
   - Quest 打开 `http://<IP>:4007/?local`，生成 Franka Emika Panda。
   - ROS 面板里 Bridge 填 `<IP>:9090`，勾选这台 Panda。
   - Real robot 一栏：Enable 之后影子变绿，拖 marker，影子会跟着走。
2. **接真机，先在电脑浏览器上测，不戴头显**：
   ```bash
   .venv/bin/python -m real_robot --robot panda --ip 172.16.0.2 --speed 0.2 --dynamics 0.1
   ```
   1. 影子是蓝色，显示的是真机当前的位置。
   2. 按 **Sync**，仿真里的机器人会跳到真机的姿态。
   3. 按 **Enable**，影子变绿。
   4. 用鼠标小幅度拖 marker，真机应该慢慢跟随。
   5. 试夹爪：Panda 的夹爪只有开和合两种状态。
3. **测安全机制**：
   - 关掉网页或断开 Wi-Fi：真机 0.3 s 内停住；最后一个客户端断开时自动失能。
   - 按外部急停：状态变成 fault（影子变红）。松开急停后，在网页上再按 Enable 就能恢复。
   - Disable：真机停在当前位置。
4. **戴上 Quest**：
   - 进入 VR，按住 Y 打开菜单，进 Real robot → Enable。
   - 抓住 marker 移动，扣扳机控制夹爪。
   - 熟悉之后再把 `--speed` 调到 0.3–0.5。

---

## B. giskard demo

### Franka 电脑要装的东西

- ROS 2 Jazzy。它要求 **Ubuntu 24.04**。
- ros2_control、ros2_controllers、ros2controlcli（ros2 control 命令行工具）。
- 从源码编译 `topic_based_ros2_control`。
- giskardpy（CRAM 的 venv）。
- franky 要装进 **ROS 用的 Python**（`/usr/bin/python3`）。

这些手动装比较麻烦，推荐用下面的 Docker。

### 用 Docker（推荐）

可以用 Docker。workshop 自己的镜像（`binder/Dockerfile`）里已经有 ROS Jazzy、CRAM/giskardpy、ros2_control 和编译好的 `topic_based_ros2_control`，只要再装 franky。

实时性由**宿主机**的 RT 内核提供，容器和宿主机用的是同一个内核，只需要给容器实时调度权限。所以：

- **RT 内核装在宿主机上**，宿主机是 Ubuntu 20.04 或 22.04 都行。
- 容器里是 Ubuntu 24.04 加 Jazzy。

```bash
cd ~/ijcai_2026_workshop
docker compose -f binder/docker-compose.yml build          # 生成镜像 cram-vrb-lab:main

docker run -it --rm --name panda \
  --network host --ipc host \
  --cap-add SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1 \
  -v ~/robot-lobby:/home/jovyan/robot-lobby \
  -e LOBBY_DIR=/home/jovyan/robot-lobby \
  cram-vrb-lab:main bash
```

进容器后，把 franky 装进 ROS 的 Python（cp312）：

```bash
sudo /usr/bin/python3 -m pip install --break-system-packages \
  "https://github.com/TimSchneider42/franky/releases/download/v2.0.0/franky_control-2.0.0+libfranka.0.9.2-cp312-cp312-manylinux_2_34_x86_64.whl"
```

几点说明：

- 跑 demo 要用镜像里自带的那份 workshop（`~/ijcai_2026_workshop`，build 时拷进去的，`ros2_ws` 已经编译好），不要把宿主机上的 workshop 挂载进去替换它。改了 demo 就重新 build 一次镜像。
- 如果 `/usr/bin/python3 -m pip` 提示没有 pip，先运行 `sudo apt-get update && sudo apt-get install -y python3-pip`。
- `--network host`：容器直接用宿主机的网卡访问 172.16.0.2。libfranka 走 UDP，不要用端口映射。
- `docker-compose.yml` 里的 GPU 设置和 Jupyter 是给仿真用的，真机测试用不到，直接用上面的 `docker run`。
- 想省掉每次装 franky：可以写个两行的 Dockerfile `FROM cram-vrb-lab:main` 加上面的 pip 命令，或者用 `docker commit`。
- 遥操测试 A 也可以在这个容器里跑：先 `pip install websockets`，再执行 `python3 -m real_robot`。

### 步骤（在容器里，`cd ~/ijcai_2026_workshop`）

1. **假机器人**：
   ```bash
   DRIVER=fake demo/panda_giskard/run_panda_giskard_demo.sh
   ```
   最后打印 `demo exited (0)`，说明这套环境是好的。
   - 如果 ros2_control_node 段错误：ros2_control 的包版本不一致，处理方法见 workshop README。
2. **可选：先在 Lobby 里的 Panda 上跑一遍**：`DRIVER=lobby`。Lobby 那边的设置见 workshop README。
3. **真机**：
   1. 先在 Desk 或者用手引导，把机器人放到 ready 姿态附近。ready 姿态是 `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]`。
   2. 清空工作区：demo 会让手先往前移 15 cm、再往下 10 cm。
   3. 运行：
      ```bash
      DRIVER=real IP=172.16.0.2 SPEED=0.2 DYNAMICS=0.1 demo/panda_giskard/run_panda_giskard_demo.sh
      ```
   4. 脚本会先让你确认，然后才使能机器人。整个过程：回 ready → 往前往下移动 → 夹爪闭合 → 张开 → 回 ready。
   5. 日志在 `~/.panda_giskard_demo/`，实测关节在 `/panda/hw/measured_joint_states`。
4. Ctrl-C 会停掉所有进程，机器人停在原地。

### 注意

- `panda.urdf` 只有运动学模型，没有碰撞模型，所以 giskard **不会避碰**，包括桌面。目标点要自己确认是安全的。
- 以下情况会让机器人停下并报 fault，在 `/panda/hw/status` 能看到原因：
  - 真机落后指令超过 0.5 rad；
  - 触发了碰撞反射（reflex）；
  - 0.3 s 没有收到速度指令。
- 报 fault 后，再调用一次 enable 就能恢复：
  ```bash
  ros2 service call /panda/hw/enable std_srvs/srv/SetBool "{data: true}"
  ```
