# main_launcher.py
# ------------------------------------------------------------
# 一键切换“单模块”运行（TTS / ASR / 测评 / 微调 / 数据清洗 / 语音桥接）
# 设计目标（按用户需求实现）：
#   1) 同一时刻仅允许一个功能模块进程存活（防止显存/内存爆）
#   2) 主控台提供导航按钮：点击 → 关闭当前模块 → 启动目标模块 → 自动跳转到该模块页面
#   3) 保留原有：端口占用预检 / 健康检查 / 日志落盘 / 异常重启 / 优雅关闭
# ------------------------------------------------------------

import os
import sys
import time
import socket
import signal
import atexit
import subprocess
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import datetime
from typing import List, Optional
from my_config import BASE_DIR

# [ADD] 新增：主控台（Gradio）作为“切换中心”
try:
    import gradio as gr  # pip install gradio
except Exception:
    gr = None  # 若没有安装，也不影响命令行使用（但没有Web主控台）

# ============【配置区：按你的实际环境修改】===================================

LAUNCH_CONFIG = [
    {
        "name": "TTS",
        "cmd": ["python", "gradio_tts.py"],  # 例如：["python", "webui.py", "--port", "9872"]
        "cwd": BASE_DIR,  # [KEEP] 按你现有路径；后续可改相对路径
        "url": "http://127.0.0.1:9872/",
        "enabled": True,
        "auto_restart": True,
        "max_restarts": 1,
        "health_timeout_sec": 120,
        "health_interval_sec": 2,
    },
    {
        "name": "ASR",
        "cmd": ["python", "gradio_asr.py"],  # 你现有的 ASR 页面脚本
        "cwd": BASE_DIR,
        "url": "http://127.0.0.1:9874/",
        "enabled": True,
        "auto_restart": True,
        "max_restarts": 1,
        "health_timeout_sec": 60,
        "health_interval_sec": 2,
    },
    {
        "name": "EVAL",  # 测评占位；未实现可保持 disabled
        "cmd": ["python", "gradio_eval.py"],
        "cwd": BASE_DIR,
        "url": "http://127.0.0.1:9881/",
        "enabled": True,
        "auto_restart": True,
        "max_restarts": 1,
        "health_timeout_sec": 40,
        "health_interval_sec": 2,
    },
    {
        "name": "FINETUNE",  # 微调占位
        "cmd": ["python", "gradio_finetune.py"],
        "cwd": BASE_DIR,
        "url": "http://127.0.0.1:9882/",
        "enabled": True,
        "auto_restart": True,
        "max_restarts": 1,
        "health_timeout_sec": 40,
        "health_interval_sec": 2,
    },
    {
        "name": "CLEAN",  # 数据清洗占位
        "cmd": ["python", "clean_app.py"],
        "cwd": BASE_DIR,
        "url": "http://127.0.0.1:9883/",
        "enabled": False,
        "auto_restart": True,
        "max_restarts": 1,
        "health_timeout_sec": 30,
        "health_interval_sec": 2,
    },
    {
        "name": "BRIDGE",  # 语音桥接占位
        "cmd": ["python", "bridge_app.py"],
        "cwd": BASE_DIR,
        "url": "http://127.0.0.1:9884/",
        "enabled": False,
        "auto_restart": True,
        "max_restarts": 1,
        "health_timeout_sec": 40,
        "health_interval_sec": 2,
    },
]

LOG_DIR = "logs"  # [KEEP] 日志目录

# [MOD] 默认打开的模块（满足“首次即进入 TTS”）
DEFAULT_MODULE = "TTS"

# [ADD] 主控台端口（可改）
CONTROL_PORT = 9900

# [MOD] 首次启动默认跳转到 DEFAULT_MODULE
OPEN_BROWSER_TO = None  # 稍后在 __main__ 里根据 DEFAULT_MODULE 动态设置


# ===================【实现逻辑：原有 + 新增“单模块切换”】======================

class ModuleProc:
    """单个模块的启动/健康检查/监督管理封装"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.log_fp = None
        self._parse_url()

    def _parse_url(self):
        u = urlparse(self.cfg["url"])
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port  # 必须在 url 中显式包含端口

    @property
    def name(self) -> str:
        return self.cfg["name"]

    def log_path(self) -> str:
        return os.path.join(LOG_DIR, f"{self.name}.log")

    def open_log(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_fp = open(self.log_path(), "a", encoding="utf-8", buffering=1)
        header = f"\n\n===== [{self.name}] START @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        self.log_fp.write(header)

    def close_log(self):
        try:
            if self.log_fp:
                self.log_fp.flush()
                self.log_fp.close()
        except Exception:
            pass
        finally:
            self.log_fp = None

    def is_port_in_use(self) -> bool:
        if not self.port:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                return s.connect_ex((self.host, int(self.port))) == 0
            except Exception:
                return False

    def start(self) -> bool:
        """启动子进程；端口被占用则跳过启动（认为“可能已在运行”）"""
        if not self.cfg.get("enabled", True):
            print(f"[{self.name}] disabled → 跳过。")
            return False

        if self.is_port_in_use():
            print(f"[{self.name}] 端口 {self.port} 已占用：假定服务已在运行，跳过启动。URL={self.cfg['url']}")
            return True

        # 若是 python 脚本形式，先校验脚本存在
        if len(self.cfg["cmd"]) >= 2 and str(self.cfg["cmd"][0]).lower().startswith("python"):
            script_path = os.path.join(self.cfg["cwd"], self.cfg["cmd"][1])
            if not os.path.exists(script_path):
                print(f"[{self.name}] 找不到脚本：{script_path} —— 请检查路径或将 enabled 设为 False（占位）。")
                return False

        self.open_log()
        # 子进程标注输出给了日志文件记录，不再输出到cmd
        stdout = self.log_fp
        stderr = subprocess.STDOUT

        creationflags = 0
        if os.name == "nt":
            # Windows：独立进程组，便于 Ctrl+C 统一关闭
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore

        print(f"[{self.name}] 启动中：{' '.join(self.cfg['cmd'])}  (cwd={self.cfg['cwd']})")
        try:
            self.proc = subprocess.Popen(
                self.cfg["cmd"],
                cwd=self.cfg["cwd"],
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags
            )
            return True
        except Exception as e:
            print(f"[{self.name}] 启动失败：{e}")
            self.close_log()
            return False

    def is_running(self) -> bool:
        return (self.proc is not None) and (self.proc.poll() is None)

    def wait_healthy(self) -> bool:
        """通过 HTTP GET 轮询健康状态（200~399 视为健康）"""
        timeout = int(self.cfg.get("health_timeout_sec", 60))
        interval = float(self.cfg.get("health_interval_sec", 2))
        deadline = time.time() + timeout
        url = self.cfg["url"]

        while time.time() < deadline:
            if self._http_ok(url):
                print(f"[{self.name}] 健康检查通过：{url}")
                return True
            # 端口被外部进程占用 + 本进程未运行 → 也视作“已上线”
            if self.is_port_in_use() and not self.is_running():
                print(f"[{self.name}] 端口被占用但本进程未运行，可能为外部服务。视为已上线：{url}")
                return True
            time.sleep(interval)

        print(f"[{self.name}] 健康检查超时（{timeout}s）：{url}")
        return False

    @staticmethod
    def _http_ok(url: str) -> bool:
        try:
            req = Request(url, headers={"User-Agent": "launcher/1.0"})
            with urlopen(req, timeout=2) as resp:
                return 200 <= resp.status < 400
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError):
            return False

    def terminate(self):
        """优雅终止子进程"""
        if self.proc is None:
            return
        if self.is_running():
            print(f"[{self.name}] 尝试优雅终止 (pid={self.proc.pid}) ...")
            try:
                if os.name == "nt":
                    # Windows：给子进程组发 CTRL_BREAK_EVENT，再调用 terminate 兜底
                    try:
                        self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore
                        time.sleep(1.0)
                    except Exception:
                        pass
                self.proc.terminate()
            except Exception:
                pass

    def kill(self):
        """强制结束"""
        if self.proc and self.is_running():
            print(f"[{self.name}] 强制结束 (pid={self.proc.pid})")
            try:
                self.proc.kill()
            except Exception:
                pass

    def supervise(self):
        """监督：若异常退出且允许重启，则在限制次数内自动重启"""
        if self.proc is None:
            return
        ret = self.proc.poll()
        if ret is None:
            return  # 仍在运行
        if ret != 0 and self.cfg.get("auto_restart", True) and self.restarts < int(self.cfg.get("max_restarts", 1)):
            self.restarts += 1
            print(f"[{self.name}] 异常退出（code={ret}），尝试第 {self.restarts} 次重启 ...")
            if self.start():
                self.wait_healthy()


# ----------------------- 管理与切换（新增） -----------------------------

PROCS: List[ModuleProc] = []
NAME2MOD = {}


def _build_registry():
    """[ADD] 初始化模块对象表"""
    global PROCS, NAME2MOD
    PROCS = [ModuleProc(cfg) for cfg in LAUNCH_CONFIG]
    NAME2MOD = {m.name: m for m in PROCS}


def shutdown_all():
    """统一关闭与清理"""
    for m in PROCS:
        m.terminate()
    time.sleep(2)
    for m in PROCS:
        if m.is_running():
            m.kill()
        m.close_log()


def _running_mod() -> Optional[ModuleProc]:
    """[ADD] 返回当前正在运行的模块（按进程状态或端口检查近似判断）"""
    for m in PROCS:
        if m.is_running():
            return m
    # 兜底：如果某个端口可用，也认为它在运行（可能是外部启动）
    for m in PROCS:
        if m._http_ok(m.cfg["url"]):
            return m
    return None


def switch_to(name: str) -> (str, str):
    """
    [ADD] 单击切换核心逻辑：
      1) 关闭所有其它模块
      2) 启动目标模块
      3) 等健康后返回状态与“重定向脚本HTML”
    返回：(status_markdown, redirect_html)
    """
    if name not in NAME2MOD:
        return f"❌ 未找到模块：{name}", "<p>无效目标</p>"

    target = NAME2MOD[name]
    # 1) 关闭其他模块
    for m in PROCS:
        if m.name != name:
            m.terminate()
    time.sleep(1.0)
    for m in PROCS:
        if m.name != name and m.is_running():
            m.kill()

    # 2) 启动目标（若端口已被占用，视作外部已运行）
    started = target.start()
    healthy = target.wait_healthy()

    url = target.cfg["url"]
    status_lines = [
        f"### 切换到：**{name}**",
        f"- 启动动作: {'执行' if started else '跳过(端口占用或未启用)'}",
        f"- 健康检查: {'通过 ✅' if healthy else '超时/未知 ⚠️'}",
        f"- 目标地址: {url}",
        "",
        "（如果页面未自动跳转，可手动点击下方链接）",
    ]
    status = "\n".join(status_lines)

    # 3) 自动跳转脚本（同页跳转）
    redirect_html = f"""
    <p>即将跳转到 <a href="{url}">{url}</a> …</p>
    <script>
      setTimeout(function() {{
        window.location.href = "{url}";
      }}, 600);
    </script>
    """
    return status, redirect_html


def running_summary() -> str:
    """[ADD] 展示当前运行状态"""
    lines = ["### 运行状态"]
    for m in PROCS:
        state = "RUNNING" if (m.is_running() or m._http_ok(m.cfg["url"])) else (
            "DISABLED" if not m.cfg.get("enabled", True) else "STOPPED")
        lines.append(f"- {m.name:9s} : {state} → {m.cfg['url']}")
    return "\n".join(lines)


def supervise_loop():
    """主监督循环：定期检查子进程存活并按需重启"""
    try:
        while True:
            for m in PROCS:
                m.supervise()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备关闭全部模块 ...")
        shutdown_all()


# ----------------------- 主控台（Gradio） -------------------------------

def launch_control_plane():
    """[ADD] 启动主控台 UI：按钮切换模块 + 自动跳转"""
    if gr is None:
        print("[WARN] 未安装 gradio，无法提供Web主控台。可运行：pip install gradio")
        return

    with gr.Blocks(title="多模块主控台 · 单实例切换") as app:
        gr.Markdown("# 🧭 多模块主控台（单实例切换）")
        status_md = gr.Markdown(running_summary())
        redirect_html = gr.HTML("")

        with gr.Row():
            btns = []
            for m in PROCS:
                # 对 disabled 的也给按钮，但提示未启用
                label = f"切到 {m.name}"
                btns.append(gr.Button(label, variant="primary" if m.name == DEFAULT_MODULE else "secondary",
                                      elem_id=f"btn_switch_{m.name}"))
            stop_btn = gr.Button("■ 停止全部模块", variant="stop")

        # 点击绑定：逐个按钮绑定到 switch_to
        # ===== [MOD] 交互绑定：后端切换完成后，前端自动整页跳转 =====
        for i, m in enumerate(PROCS):
            mod_name = m.name
            mod_url = m.cfg["url"]

            def make_fn(mod_name=mod_name):
                def _fn():
                    # 切换到目标模块（会关掉其它）+ 刷新状态
                    s, _ = switch_to(mod_name)  # 不再用 redirect_html
                    return s, running_summary()

                return _fn

            # 1) 先执行后端切换（只输出状态：两个 Markdown）
            handle = btns[i].click(make_fn(), inputs=None, outputs=[status_md, status_md])

            # 2) 再做前端跳转（等待上一步完成），兼容新旧 gradio + iframe 顶层跳转
            _redir_js = f"(x)=>{{ window.top.location.href = '{mod_url}'; return []; }}"
            try:
                handle.then(None, [], [], js=_redir_js)  # gradio v4+
            except TypeError:
                handle.then(None, [], [], _js=_redir_js)  # gradio v3.x

        def _stop_all():
            shutdown_all()
            return "### 已停止全部模块。", "", running_summary()

        stop_btn.click(_stop_all, inputs=None, outputs=[status_md, redirect_html, status_md])

        # [ADD] 页面加载自动切换：根据 ?switch=模块名 自动点击对应“切到 XXX”按钮
        _auto_js = """
        (x)=>{
          try{
            const url = new URL(window.location.href);
            const mod = url.searchParams.get('switch');
            if(!mod) return [];
            const id = 'btn_switch_' + mod;  // 例如 btn_switch_ASR

            // 防止循环：清掉查询参数
            url.searchParams.delete('switch');
            window.history.replaceState({}, '', url.toString());

            function tryClick(n=0){
              // Gradio 把 elem_id 挂在外层容器，要点容器里的 <button>
              const host = document.getElementById(id);
              const btn  = host && (host.querySelector('button') || host);
              if(btn){ btn.click(); return []; }
              if(n < 100){ setTimeout(()=>tryClick(n+1), 80); }
              return [];
            }
            tryClick();
            return [];
          }catch(e){ return []; }
        }
        """
        try:
            app.load(None, [], [], js=_auto_js)      # Gradio v4+
        except TypeError:
            app.load(None, [], [], _js=_auto_js)     # Gradio v3.x 兼容

        gr.Markdown(
            f"> 建议将本页固定在浏览器；如需返回主控台，访问：**http://127.0.0.1:{CONTROL_PORT}/**。"
        )

    # 注意：主控台自身很轻，几乎不占资源
    app.launch(server_name="0.0.0.0", server_port=CONTROL_PORT, inbrowser=False)


# ----------------------- 程序入口 --------------------------------------

if __name__ == "__main__":
    _build_registry()  # [ADD] 初始化模块表

    # [MOD] 首次仅启动默认模块（避免“一次性全拉起”导致显存爆）
    if DEFAULT_MODULE in NAME2MOD and NAME2MOD[DEFAULT_MODULE].cfg.get("enabled", True):
        print(f"\n=== 首次启动默认模块：{DEFAULT_MODULE} ===")
        switch_to(DEFAULT_MODULE)  # 内部会先停其它（此时也没有其它），再启默认模块
        OPEN_BROWSER_TO = NAME2MOD[DEFAULT_MODULE].cfg["url"]
    else:
        OPEN_BROWSER_TO = None

    # [ADD] 启动主控台 UI（如未安装 gradio，会给出提示并跳过）
    if gr is not None:
        # 主控台与监督循环并行各自运行：建议在不同终端运行这个脚本一次即可
        # 你也可以把 supervise_loop 放到后台线程，这里保持简单直接。
        # 为了便于 Ctrl+C 统一退出，我们直接串行：先起主控台，另开一个终端再起一份此脚本以做纯监督亦可。
        # 实际上 supervise_loop 不一定需要，默认重启策略也保留。
        try:
            # 打开默认模块页面（可选）
            if OPEN_BROWSER_TO:
                try:
                    import webbrowser

                    webbrowser.open(OPEN_BROWSER_TO)
                except Exception:
                    pass

            # 启动主控台（阻塞）
            launch_control_plane()
        finally:
            # 退出主控台时，确保清理
            shutdown_all()
    else:
        # 纯命令行模式：打开默认模块并进入监督循环
        print("[INFO] 以命令行模式运行（未启主控台 UI）")
        supervise_loop()
