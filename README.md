# 🧩 星TAP 智能拼图大师 | Grid Puzzle Master (全RUST内核)

![Version](https://img.shields.io/badge/version-3.5.0-blue)
![Language](https://img.shields.io/badge/language-Rust-orange)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey)

> **极致性能，极简生活。** 星TAP 实验室出品，专为摄影爱好者与自媒体人打造的“单一 EXE”独立运行版智能拼图工具。

---

## 📸 界面预览 (UI Preview)

![星TAP拼图大师V3.5 界面展示](https://raw.githubusercontent.com/cscb603/Grid-Puzzle-Pro/main/assets/preview_ui.png)

## 🎨 成品展示 (Sample Output)

![星TAP拼图大师V3.5 成品展示](https://raw.githubusercontent.com/cscb603/Grid-Puzzle-Pro/main/assets/sample_output.jpg)

---

## ✨ 小白看这里：为什么它更好用？

如果你厌倦了网页版拼图的广告、手机端拼图的模糊、以及传统软件的一堆依赖文件夹，**星TAP 智能拼图大师 v3.5** 就是为你准备的：

- **🚀 全 Rust 架构，快如闪电**：
  采用工业级 Rust 语言重构，利用你电脑的多核 CPU 进行并行处理。无论是 9 张图还是 90 张图，合成都在瞬息之间。
- **📦 单一文件，零依赖 (Standalone)**：
  **这是我们最大的改进！** Windows 版现在只有一个独立的 `.exe` 文件。没有复杂的 `lib` 文件夹，没有乱七八糟的资源包，图标、字体全部嵌入在程序里。下载即用，用完即走，不留任何系统垃圾。
- **🛡️ 拒绝截断，画质守护**：
  智能画布算法确保每一张图片都 100% 完整显示。告别市面上某些软件“拼完发现少了一截”的尴尬，支持高清原图无损导出。
- **🎨 大师级审美 UI**：
  预览区自动撑满屏幕，大图排版一目了然。支持圆角处理、智能排序（按亮度/色彩）、以及自适应布局。

---

## 🛠️ 极客看这里：底层硬核技术

- **高性能渲染**：基于 `fast_image_resize` 库，提供比传统库快数倍的图像缩放体验。
- **即时 GUI**：使用 `egui` 构建的即时模式界面，GPU 加速渲染，操作零延迟。
- **静态嵌入**：通过 `include_bytes!` 宏实现静态资源编译集成，通过 `embed-resource` 解决 macOS 交叉编译 Windows 资源的兼容性问题。
- **并发控制**：深度集成 `rayon` 库，实现真正的多线程并行流水线。

---

## 🚀 快速开始

### Windows 用户
1. 下载 `星TAP拼图大师_v3.5_Standalone.exe`。
2. 双击直接运行（无需安装）。

### macOS 用户
1. 下载并解压 `星TAP智能拼图大师_V3.5_MacOS.zip`。
2. 拖入“应用程序”文件夹即可。

---

## 📁 项目结构
- `src/main.rs`: 核心 UI 逻辑与图像合成算法。
- `build.rs` & `resources.rc`: 独立版 Windows 资源编译配置。
- `assets/`: 原始视觉资产（图标、预览图）。

---
星TAP 实验室 © 2026 | [cscb603@qq.com](mailto:cscb603@qq.com)
