# 🧩 星TAP 智能拼图大师 | Grid Puzzle Master (全RUST内核)

![Version](https://img.shields.io/badge/version-3.5.0-blue)
![Language](https://img.shields.io/badge/language-Rust-orange)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey)

星TAP 实验室出品的高性能智能拼图工具，旨在解决大批量图片快速排版与高质量导出的痛点。基于 **全RUST内核** 开发，拥有极致的性能与内存安全。

---

## ✨ 小白看这里 (User Guide)

### 为什么选择它？
- **告别截断**：智能计算画布尺寸，确保导出的拼图成品 100% 完整，不再丢失边缘。
- **大屏预览**：预览区域自动撑满窗口，左侧看文件，右侧看大图，排版效果一目了然。
- **傻瓜操作**：直接将图片拖入软件，自动生成缩略图，点击按钮即可完成拼图。
- **极致速度**：无论处理 10 张还是 100 张图，Rust 内核都能在毫秒间完成。

### 如何使用？
1. 打开 `星TAP智能拼图大师.app`。
2. 拖入需要拼合的图片。
3. 在参数面板调整边距或行列（默认已设为最优）。
4. 点击“生成并保存”，成品将自动存放在 `output` 目录。

---

## 🛠️ 极客看这里 (Technical Specs)

### 核心架构
- **UI 框架**：使用 `eframe/egui` 构建的即时模式 GUI，确保极致的交互响应速度。
- **渲染引擎**：基于 `image` 与 `fast_image_resize` 库，支持高质量的双线性/双三次插值缩放。
- **并行处理**：利用 `rayon` 进行多线程缩略图生成与图像合成。
- **字体处理**：集成 `FontMaster`，完美支持中文字符绘制与版权水印（可选）。

### 版本记录 (v3.5)
- **画布算法优化**：替换了硬编码的画布宽度，改为基于 `inner_w + outer_margin` 的动态计算。
- **布局引擎重构**：使用 `horizontal_top` 与 `auto_shrink([false; 2])` 解决了预览区组件高度压缩的问题。
- **跨平台适配**：已接入星TAP实验室通用工作流底座，支持 macOS 原生签名与 Windows 交叉编译。

---

## 📁 目录说明
- `src/main.rs`: 核心 UI 逻辑与图像合成算法。
- `assets/`: 图标与视觉资源。
- `Cargo.toml`: 项目依赖与工作区配置。

---
© 2026 星TAP实验室 <cscb603@qq.com>
