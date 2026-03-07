---
name: deploy-xllm-npu
description: 在本项目中指导在 NPU 环境下部署与启动 xLLM 推理服务，包含 NPU Docker 镜像拉取、容器启动、xLLM 编译和以 Qwen3 为例的启动脚本。当用户在本项目中提到在 NPU 上部署、安装、启动或重启 xLLM 服务时使用。
---

# 在 NPU 上部署与启动 xLLM

## 使用场景

在本仓库中，当用户需要在 **NPU（A2/A3 等）** 环境部署或启动 xLLM 时，使用本 Skill：

- 拉取并启动 NPU 开发镜像
- 在容器内编译 xLLM（非 release 镜像）
- 以 Qwen3 为例启动 xLLM 服务（单机单卡 / 单机多卡）

官方文档参考：`docs/zh/getting_started/quick_start.md` 与 `docs/zh/getting_started/launch_xllm.md`。  
更详细的多机、多模型说明见本 Skill 目录下的 `reference.md`。

---

## 一、NPU 环境 Docker 准备

所有镜像位于：

- `https://quay.io/repository/jd_xllm/xllm-ai?tab=tags`

### 1. 拉取 NPU 开发镜像

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-arm
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-hc-rc2-arm
```

### 2. 启动 NPU 开发容器

```bash
docker run -it \
  --ipc=host \
  -u 0 \
  --name xllm-npu \
  --privileged \
  --network=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /usr/local/sbin/:/usr/local/sbin/ \
  -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
  -v /var/log/npu/slog/:/var/log/npu/slog \
  -v /var/log/npu/profiling/:/var/log/npu/profiling \
  -v /var/log/npu/dump/:/var/log/npu/dump \
  -v $HOME:$HOME \
  -w $HOME \
  <docker_image_name> \
  /bin/bash
```

**注意事项：**

- `<docker_image_name>` 替换为上一步拉取的镜像 tag
- Ascend 驱动和日志相关路径需与宿主机实际安装路径一致

---

## 二、容器内编译 xLLM（二进制）

> 如果使用的是 **release 镜像（tag 中带版本号）**，可以跳过编译步骤，镜像中通常已包含 `/usr/local/bin/xllm`。

### 1. 克隆仓库与依赖

在容器内执行：

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm

# 第一次需要进行 pre-commit 安装
pip install pre-commit
pre-commit install

git submodule update --init
```

### 2. 编译 xLLM

```bash
python setup.py build
```

- 默认生成的二进制位于：`/path/to/xllm/build/xllm/core/server/xllm`
- 在 A3 机器上需要根据官方说明添加 `--device a3` 等编译参数（如有需要时再查文档细化）

---

## 三、在 NPU 上启动 xLLM（Qwen3 示例）

> 以下内容来自 `docs/zh/getting_started/launch_xllm.md` 中的 NPU 示例，适用于单机单卡和单机多卡。  
> 多卡时需修改 `NNODES` 和 `ASCEND_RT_VISIBLE_DEVICES`。

### 1. 推荐：使用脚本 `scripts/run_xllm_npu.sh`

在本 Skill 目录下提供了一个示例脚本：

- 路径：`scripts/run_xllm_npu.sh`
- 用途：在 NPU 上以 Qwen3 为例启动 xLLM（单机单卡/多卡）

**使用方式：**

1. 根据实际情况修改脚本中的关键变量：
   - `MODEL_PATH`
   - `MASTER_NODE_ADDR`
   - `START_PORT`
   - `XLLM_BIN`（xllm 可执行文件路径）
2. 赋予执行权限：

   ```bash
   chmod +x scripts/run_xllm_npu.sh
   ```

3. 启动服务：

   ```bash
   ./scripts/run_xllm_npu.sh
   ```

如需查看脚本的全部参数说明与多机扩展建议，可参考本目录下的 `reference.md`。

---

## 四、答复时的操作流程建议

当用户在本项目中说「在 NPU 上部署/启动 xLLM」时，按以下顺序回答并给命令：

1. **确认环境信息**
   - 是否已有 NPU 驱动和 Ascend 相关依赖
   - 是否允许使用 Docker / 已选定的镜像 tag
2. **给出完整 NPU Docker+容器启动命令**
   - 使用上文「拉取镜像」+「启动容器」命令
3. **视情况决定是否需要编译**
   - 如果明确使用 release 镜像，说明可以直接用 `/usr/local/bin/xllm`
   - 否则给出「克隆 + 编译」命令
4. **给出启动脚本**
   - 提供一份可直接保存为 `run_xllm_npu.sh` 的脚本
   - 明确提示需要替换 `/path/to/xllm`、`MODEL_PATH` 等变量
5. **提示检查方式**
   - 查看 `log/node_0.log` 是否有报错
   - 如有必要提示用 `ps` / `netstat` 检查进程与端口

