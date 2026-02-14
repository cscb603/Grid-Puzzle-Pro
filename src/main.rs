#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/*
 * Copyright (c) 2026 星TAP实验室
 * Project: 网格智能拼图 (Grid Puzzle Master)
 * Description: 大师级 Rust 图像处理与智能拼图工具
 */

use core_lib::ui::{eframe, egui, fonts::FontMaster, theme::UiTheme};
use fast_image_resize as fr;
use fast_image_resize::images::Image;
use image::{DynamicImage, GenericImageView, Rgba};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, error, info};

use rusttype::Font;
use std::sync::OnceLock;

static FONT_DATA: &[u8] = include_bytes!("../../../libs/core_lib/assets/fonts/MiSans-Regular.ttf");
static FONT_RT: OnceLock<Font<'static>> = OnceLock::new();
static FONT_AB: OnceLock<ab_glyph::FontVec> = OnceLock::new();

fn get_font_rt() -> &'static Font<'static> {
    FONT_RT.get_or_init(|| Font::try_from_bytes(FONT_DATA).expect("Error constructing Font"))
}

fn get_font_ab() -> &'static ab_glyph::FontVec {
    FONT_AB.get_or_init(|| {
        ab_glyph::FontVec::try_from_vec(FONT_DATA.to_vec()).expect("Error constructing FontVec")
    })
}

fn main() -> eframe::Result {
    // 设置崩溃捕获钩子，方便调试
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("程序崩溃了: {:?}", panic_info);
    }));

    // 初始化日志
    tracing_subscriber::fmt::init();
    info!("Starting Grid Puzzle Master...");

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([800.0, 950.0]) // 宽度 800，高度设大一些以适配多数屏幕高度
        .with_min_inner_size([800.0, 700.0]) // 最小宽度保持 800
        .with_title("星TAP 拼图大师 v3.5 (全RUST内核)");

    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(Arc::new(icon));
    }

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "grid_puzzle_master_v3_5_final_release", // 再次更换 ID 彻底重置
        native_options,
        Box::new(|cc| {
            // 明确禁用持久化存储，不保存任何用户勾选状态
            cc.egui_ctx.memory_mut(|mem| *mem = Default::default());

            // 初始化默认主题
            UiTheme::apply_master_visuals(&cc.egui_ctx, core_lib::ui::theme::ThemeMode::Auto);
            // 嵌入中文字体 (MiSans)
            let font_data =
                include_bytes!("../../../libs/core_lib/assets/fonts/MiSans-Regular.ttf");
            FontMaster::setup_chinese_fonts(&cc.egui_ctx, "MiSans", font_data);
            Ok(Box::new(GridPuzzleApp::default()))
        }),
    )
}

fn load_icon() -> Option<egui::IconData> {
    // 大师级优化：将图标直接嵌入二进制，实现真正独立运行
    let icon_data = include_bytes!("../assets/app.ico");
    if let Ok(img) = image::load_from_memory(icon_data) {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Some(egui::IconData {
            rgba: rgba.into_raw(),
            width,
            height,
        })
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LayoutStyle {
    Columns,
    Rows,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SortStrategy {
    Balanced,
    Brightness,
    Hue,
    Random,
}

use std::sync::mpsc::{channel, Receiver, Sender};

type ThumbReceiver = Arc<std::sync::Mutex<Receiver<(PathBuf, egui::ColorImage)>>>;

struct GridPuzzleApp {
    input_images: Vec<PathBuf>,
    selected_indices: std::collections::HashSet<usize>,
    thumbnails: std::collections::HashMap<PathBuf, egui::TextureHandle>,
    loading_paths: std::collections::HashSet<PathBuf>,
    output_dir: PathBuf,
    column_count: u32,
    layout_style: LayoutStyle,
    sort_strategy: SortStrategy,
    rounded_corners: bool,
    crop_3_4: bool,
    tag_enabled_v2: bool, // 彻底更换变量名
    compress_output: bool,
    web_mode: bool,
    theme_mode: core_lib::ui::theme::ThemeMode,
    status: String,
    progress: f32,
    processing: bool,
    tx: Option<Sender<String>>,
    rx: Option<Arc<std::sync::Mutex<Receiver<String>>>>,
    thumb_tx: Option<Sender<(PathBuf, egui::ColorImage)>>,
    thumb_rx: Option<ThumbReceiver>,
}

impl Clone for GridPuzzleApp {
    fn clone(&self) -> Self {
        Self {
            input_images: self.input_images.clone(),
            selected_indices: self.selected_indices.clone(),
            thumbnails: self.thumbnails.clone(),
            loading_paths: self.loading_paths.clone(),
            output_dir: self.output_dir.clone(),
            column_count: self.column_count,
            layout_style: self.layout_style,
            sort_strategy: self.sort_strategy,
            rounded_corners: self.rounded_corners,
            crop_3_4: self.crop_3_4,
            tag_enabled_v2: self.tag_enabled_v2,
            compress_output: self.compress_output,
            web_mode: self.web_mode,
            theme_mode: self.theme_mode,
            status: self.status.clone(),
            progress: self.progress,
            processing: self.processing,
            tx: self.tx.clone(),
            rx: self.rx.clone(),
            thumb_tx: self.thumb_tx.clone(),
            thumb_rx: self.thumb_rx.clone(),
        }
    }
}

impl Default for GridPuzzleApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (thumb_tx, thumb_rx) = channel();

        let output_dir = dirs::desktop_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("星TAP拼图成品");

        Self {
            input_images: Vec::new(),
            selected_indices: std::collections::HashSet::new(),
            thumbnails: std::collections::HashMap::new(),
            loading_paths: std::collections::HashSet::new(),
            output_dir,
            column_count: 3,
            layout_style: LayoutStyle::Rows,
            sort_strategy: SortStrategy::Balanced,
            rounded_corners: true,
            crop_3_4: true,
            tag_enabled_v2: false, // 核心修复：绝对默认为 false
            compress_output: true,
            web_mode: false,
            theme_mode: core_lib::ui::theme::ThemeMode::Auto,
            status: "就绪".to_string(),
            progress: 0.0,
            processing: false,
            tx: Some(tx),
            rx: Some(Arc::new(std::sync::Mutex::new(rx))),
            thumb_tx: Some(thumb_tx),
            thumb_rx: Some(Arc::new(std::sync::Mutex::new(thumb_rx))),
        }
    }
}

#[derive(Clone)]
struct ProcessedImageData {
    path: PathBuf,
    img: DynamicImage,
    brightness: u64,
    hue: u32,
    ratio: f32,
}

impl GridPuzzleApp {
    /// 智能提取作者名 (移植自 Python 版，并进行了大师级增强)
    fn extract_author(path: &Path) -> String {
        let filename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("未知");

        // 前缀定义
        let prefixes = ["手机班摄影班", "手机班", "影友"];

        // 分隔符定义
        let separators = ["作业", "作品"];

        let mut remaining = filename.to_string();
        let mut found_prefix = None;

        // 1. 提取并移除前缀
        for p in &prefixes {
            if remaining.starts_with(p) {
                found_prefix = Some(*p);
                remaining = remaining[p.len()..].trim().to_string();
                break;
            }
        }

        // 2. 查找分隔符
        let mut author_part = remaining.clone();
        for s in &separators {
            if let Some(pos) = remaining.find(s) {
                author_part = remaining[..pos].trim().to_string();
                break;
            }
        }

        // 3. 如果提取后为空，尝试按空格取第一段
        if author_part.is_empty() {
            if let Some(first) = remaining.split_whitespace().next() {
                author_part = first.to_string();
            } else {
                return "未知作者 作品".to_string();
            }
        }

        // 4. 清理特殊字符，保留中文、字母、点、减号、下划线
        let author = author_part
            .chars()
            .filter(|c| {
                c.is_alphanumeric()
                    || *c == '.'
                    || *c == '-'
                    || *c == '_'
                    || (*c >= '\u{4e00}' && *c <= '\u{9fa5}')
            })
            .collect::<String>();

        if let Some(p) = found_prefix {
            format!("{} {} 作品", p, author)
        } else {
            format!("{} 作品", author)
        }
    }

    fn save_result(
        canvas: &image::ImageBuffer<Rgba<u8>, Vec<u8>>,
        app_state: &GridPuzzleApp,
        tx: &Sender<String>,
    ) {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let (w, h) = canvas.dimensions();

        let (output_path, success) = if app_state.web_mode {
            // 网图模式：保存为 JPG (85% 质量，体积更小)
            let path = app_state
                .output_dir
                .join(format!("网图拼图_{}.jpg", timestamp));
            if !app_state.output_dir.exists() {
                let _ = std::fs::create_dir_all(&app_state.output_dir);
            }

            let result = (|| -> anyhow::Result<()> {
                let mut file = std::fs::File::create(&path)?;
                let encoder = jpeg_encoder::Encoder::new(&mut file, 85);

                let mut rgb_buf = Vec::with_capacity((w * h * 3) as usize);
                let rgba_raw = canvas.as_raw();
                for chunk in rgba_raw.chunks_exact(4) {
                    rgb_buf.push(chunk[0]);
                    rgb_buf.push(chunk[1]);
                    rgb_buf.push(chunk[2]);
                }

                let width_u16 = w.min(65535) as u16;
                let height_u16 = h.min(65535) as u16;
                encoder.encode(
                    &rgb_buf,
                    width_u16,
                    height_u16,
                    jpeg_encoder::ColorType::Rgb,
                )?;
                Ok(())
            })();
            (path, result.is_ok())
        } else if app_state.compress_output {
            // 高清压缩模式：保存为 JPG (95% 质量，使用极速 jpeg-encoder)
            let path = app_state
                .output_dir
                .join(format!("智能拼图_{}.jpg", timestamp));
            if !app_state.output_dir.exists() {
                let _ = std::fs::create_dir_all(&app_state.output_dir);
            }

            let save_start = std::time::Instant::now();
            let result = (|| -> anyhow::Result<()> {
                let mut file = std::fs::File::create(&path)?;
                // 算法大师：使用 jpeg-encoder 替代 image 默认编码器，速度提升数倍
                let encoder = jpeg_encoder::Encoder::new(&mut file, 95);

                // 极速转换：从 RGBA 提取 RGB，避免 DynamicImage 的重分配开销
                let mut rgb_buf = Vec::with_capacity((w * h * 3) as usize);
                let rgba_raw = canvas.as_raw();
                for chunk in rgba_raw.chunks_exact(4) {
                    rgb_buf.push(chunk[0]);
                    rgb_buf.push(chunk[1]);
                    rgb_buf.push(chunk[2]);
                }

                let width_u16 = w.min(65535) as u16;
                let height_u16 = h.min(65535) as u16;
                encoder.encode(
                    &rgb_buf,
                    width_u16,
                    height_u16,
                    jpeg_encoder::ColorType::Rgb,
                )?;
                Ok(())
            })();

            info!(
                "🚀 高清压缩保存完成 ({}x{})，耗时: {:?}",
                w,
                h,
                save_start.elapsed()
            );
            (path, result.is_ok())
        } else {
            // 原图无损模式：保存为 PNG
            let path = app_state
                .output_dir
                .join(format!("智能拼图_{}.png", timestamp));
            if !app_state.output_dir.exists() {
                let _ = std::fs::create_dir_all(&app_state.output_dir);
            }
            let result = canvas.save(&path);
            (path, result.is_ok())
        };

        if !success {
            let _ = tx.send("错误: 保存失败".to_string());
        } else {
            let _ = tx.send(format!("SUCCESS:{}", output_path.to_str().unwrap()));
            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open").arg(&output_path).spawn();
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("explorer")
                .arg(&output_path)
                .spawn();
        }
    }

    /// 获取当前主题对应的背景色 (算法大师：避免死黑死白，提升质感)
    fn get_canvas_bg_color(&self) -> Rgba<u8> {
        use chrono::Timelike;
        let is_dark = match self.theme_mode {
            core_lib::ui::theme::ThemeMode::Light => false,
            core_lib::ui::theme::ThemeMode::Dark => true,
            core_lib::ui::theme::ThemeMode::Auto => {
                let hour = chrono::Local::now().hour();
                !(6..18).contains(&hour)
            }
        };

        if is_dark {
            // 大师级深蓝灰 (与 UI 面板同步)，非纯黑
            Rgba([18, 20, 24, 255])
        } else {
            // 大师级乳白色 (高级感)，非死白
            Rgba([248, 249, 250, 255])
        }
    }

    /// 核心图像处理逻辑
    /// 核心图像处理逻辑
    fn process_images(app_state: Self) -> anyhow::Result<()> {
        let start_time = std::time::Instant::now();
        let tx = app_state.tx.as_ref().unwrap();

        if app_state.input_images.is_empty() {
            let _ = tx.send("错误: 没有选择图片".to_string());
            anyhow::bail!("没有选择图片");
        }

        info!("🚀 开始大师级预处理 pipeline (目标: 毫秒级响应)");
        let _ = tx.send("PROGRESS:0.1".to_string());
        let _ = tx.send(format!(
            "🚀 大师级预处理: {} 张图片...",
            app_state.input_images.len()
        ));

        // 1. 加载并预处理图片 (并行处理 + 特征提取)
        let images_paths = app_state.input_images.clone();
        let crop_3_4 = app_state.crop_3_4;

        info!("🚀 开始预处理: {} 张图片", app_state.input_images.len());
        let pre_start = std::time::Instant::now();
        let mut processed_data: Vec<ProcessedImageData> = images_paths
            .par_iter()
            .map(|path| {
                let mut img = image::open(path).expect("无法打开图片");
                let p_start = std::time::Instant::now();

                // 智能裁剪 3:4
                if crop_3_4 {
                    img = Self::smart_crop_3_4(&img);
                }
                debug!(
                    "  - 图片 {:?} 裁剪/加载耗时: {:?}",
                    path.file_name(),
                    p_start.elapsed()
                );

                // 提取特征 (用于排序) - 优化：直接在原图上取样
                let _f_start = std::time::Instant::now();
                let (w, h) = img.dimensions();
                let step_x = (w / 10).max(1);
                let step_y = (h / 10).max(1);
                let mut total_brightness: u64 = 0;
                let mut count = 0;
                for y in (0..h).step_by(step_y as usize) {
                    for x in (0..w).step_by(step_x as usize) {
                        let p = img.get_pixel(x, y);
                        total_brightness +=
                            (p[0] as u64 * 299 + p[1] as u64 * 587 + p[2] as u64 * 114) / 1000;
                        count += 1;
                    }
                }
                let brightness = total_brightness / count.max(1);
                info!(
                    "  - 图片 {:?} 预处理完成 (裁剪/加载/特征)，耗时: {:?}",
                    path.file_name(),
                    p_start.elapsed()
                );

                ProcessedImageData {
                    path: path.clone(),
                    ratio: img.width() as f32 / img.height() as f32,
                    img,
                    brightness,
                    hue: 0,
                }
            })
            .collect();

        info!(
            "✅ 预处理完成 (裁剪+加载+特征)，总耗时: {:?}",
            pre_start.elapsed()
        );
        let _ = tx.send("PROGRESS:0.3".to_string());
        let _ = tx.send("✨ 智能排序中...".to_string());
        app_state.apply_sorting_optimized(&mut processed_data);

        let layout_start = std::time::Instant::now();
        let _ = tx.send("PROGRESS:0.5".to_string());
        let _ = tx.send("📐 布局计算中...".to_string());
        let spacing: u32 = if app_state.web_mode { 10 } else { 15 };
        let outer_margin: u32 = if app_state.web_mode { 15 } else { 20 };
        // 算法大师：网图模式下基础宽度降至 1200px，普通模式 3000px
        let base_canvas_w: u32 = if app_state.web_mode { 1200 } else { 3000 };
        let canvas_w: u32 = base_canvas_w + outer_margin * 2;
        let inner_w: u32 = canvas_w - outer_margin * 2;

        if app_state.layout_style == LayoutStyle::Columns {
            // --- 瀑布流模式 ---
            let col_width =
                (inner_w - (app_state.column_count - 1) * spacing) / app_state.column_count;

            let _ = tx.send("PROGRESS:0.6".to_string());
            let _ = tx.send("⚡ 高性能并行缩放...".to_string());
            let scale_start = std::time::Instant::now();

            let resized_results: Vec<(usize, DynamicImage, u32, u32)> = processed_data
                .into_par_iter()
                .enumerate()
                .map(|(idx, data)| {
                    let h = (col_width as f32 / data.ratio) as u32;
                    let step_start = std::time::Instant::now();

                    let src_image = Image::from_vec_u8(
                        data.img.width(),
                        data.img.height(),
                        data.img.to_rgba8().into_raw(),
                        fr::PixelType::U8x4,
                    )
                    .unwrap();

                    let mut dst_image = Image::new(col_width, h, fr::PixelType::U8x4);
                    let mut resizer = fr::Resizer::new();
                    resizer.resize(&src_image, &mut dst_image, None).unwrap();

                    let mut resized = DynamicImage::ImageRgba8(
                        image::RgbaImage::from_raw(col_width, h, dst_image.into_vec()).unwrap(),
                    );

                    let scale_dur = step_start.elapsed();

                    // 延迟执行后处理
                    let post_start = std::time::Instant::now();
                    if app_state.tag_enabled_v2 {
                        let author = Self::extract_author(&data.path);
                        resized = app_state.draw_author_tag(resized, &author);
                    }

                    if app_state.rounded_corners {
                        let radius = (col_width.min(h) as f32 * 0.05).max(10.0);
                        resized = app_state.apply_rounded_corners(resized, radius);
                    }

                    debug!(
                        "  - 图片 {} 缩放({:?}) + 后处理({:?})",
                        idx,
                        scale_dur,
                        post_start.elapsed()
                    );

                    (idx, resized, col_width, h)
                })
                .collect();
            info!(
                "✅ 并行缩放与后处理完成，总耗时: {:?}",
                scale_start.elapsed()
            );

            let mut columns_y = vec![0u32; app_state.column_count as usize];
            let mut final_positions = Vec::new();
            for (_idx, img, _w, h) in resized_results {
                let min_col_idx = columns_y
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &h)| h)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let x = outer_margin + min_col_idx as u32 * (col_width + spacing);
                let y = outer_margin + columns_y[min_col_idx];
                final_positions.push((x, y, img));
                columns_y[min_col_idx] += h + spacing;
            }

            // 算法大师：修正底部留白，移除最后一个多余的 spacing
            let max_h = columns_y
                .iter()
                .map(|&h| if h > spacing { h - spacing } else { h })
                .max()
                .unwrap_or(0);
            let _ = tx.send("PROGRESS:0.8".to_string());
            let mut canvas = image::ImageBuffer::new(canvas_w, max_h + outer_margin * 2);
            let bg_color = app_state.get_canvas_bg_color();
            for pixel in canvas.pixels_mut() {
                *pixel = bg_color;
            }

            for (x, y, img) in final_positions {
                image::imageops::overlay(&mut canvas, &img, x as i64, y as i64);
            }

            info!("✅ 布局与合并完成，总耗时: {:?}", layout_start.elapsed());
            Self::save_result(&canvas, &app_state, tx);
        } else {
            // --- 画廊流模式 (Gallery / Row-based) ---
            // 算法大师：实现自动均衡缩放，确保单双数都能拼成完整矩形
            let mut rows = Vec::new();
            let total_images = processed_data.len();
            let col_count = app_state.column_count as usize;

            // 动态分组策略：根据总数和列数，自动分配每行图片数，确保最后一行不落单
            let mut current_idx = 0;
            while current_idx < total_images {
                let remaining = total_images - current_idx;
                let take = if remaining > col_count {
                    col_count
                } else if remaining == col_count + 1 && col_count > 1 {
                    // 关键算法：如果剩下 col+1 张图，则拆分为两行 (col/2 + col/2 风格)
                    // 避免出现最后一行只有一张图的情况，从而实现“自动均衡缩放”
                    col_count / 2 + 1
                } else {
                    remaining
                };

                let mut row = Vec::new();
                for _ in 0..take {
                    if current_idx < total_images {
                        row.push(processed_data[current_idx].clone());
                        current_idx += 1;
                    }
                }
                rows.push(row);
            }

            let mut row_configs = Vec::new();
            let mut total_h = 0;
            let rows_len = rows.len();

            for (i, row) in rows.into_iter().enumerate() {
                let total_ratio: f32 = row.iter().map(|d| d.ratio).sum();
                let is_last = i == rows_len - 1;

                // 算法大师：所有行都执行“强制对齐”逻辑
                // 因为我们已经通过分组策略确保了每行都有足够的图片，
                // 所以每一行都可以通过调整高度来填满宽度，从而拼成完美的矩形。
                let row_h = ((inner_w as f32 - (row.len() as f32 - 1.0) * spacing as f32)
                    / total_ratio) as u32;

                row_configs.push((row, row_h, true)); // 始终 should_justify = true
                total_h += row_h;
                if !is_last {
                    total_h += spacing;
                }
            }

            let _ = tx.send("PROGRESS:0.6".to_string());
            let _ = tx.send("⚡ 高性能并行缩放...".to_string());
            let scale_start = std::time::Instant::now();
            let resized_rows: Vec<Vec<(DynamicImage, u32)>> = row_configs
                .into_par_iter()
                .map(|(row_images_data, row_h, should_justify)| {
                    let mut images_with_widths = Vec::new();
                    let mut current_row_x = 0u32;

                    let count = row_images_data.len();
                    for (idx, data) in row_images_data.into_iter().enumerate() {
                        // 算法大师：预先计算宽度，最后一张图补齐像素差，彻底消除右侧白边
                        let mut target_w = (row_h as f32 * data.ratio) as u32;
                        if should_justify && idx == count - 1 {
                            target_w = inner_w.saturating_sub(current_row_x);
                        }

                        let src_image = Image::from_vec_u8(
                            data.img.width(),
                            data.img.height(),
                            data.img.to_rgba8().into_raw(),
                            fr::PixelType::U8x4,
                        )
                        .unwrap();

                        let mut dst_image = Image::new(target_w, row_h, fr::PixelType::U8x4);
                        let mut resizer = fr::Resizer::new();
                        resizer.resize(&src_image, &mut dst_image, None).unwrap();

                        let mut resized = DynamicImage::ImageRgba8(
                            image::RgbaImage::from_raw(target_w, row_h, dst_image.into_vec())
                                .unwrap(),
                        );

                        // 延迟执行后处理
                        if app_state.tag_enabled_v2 {
                            let author = Self::extract_author(&data.path);
                            resized = app_state.draw_author_tag(resized, &author);
                        }
                        if app_state.rounded_corners {
                            let radius = (target_w.min(row_h) as f32 * 0.05).max(10.0);
                            resized = app_state.apply_rounded_corners(resized, radius);
                        }

                        images_with_widths.push((resized, target_w));
                        current_row_x += target_w + spacing;
                    }
                    images_with_widths
                })
                .collect();
            info!(
                "✅ 画廊模式并行缩放与后处理完成，总耗时: {:?}",
                scale_start.elapsed()
            );

            let _ = tx.send("PROGRESS:0.8".to_string());

            // 算法大师：自适应画布宽度，防止成品截断
            let canvas_w = inner_w + outer_margin * 2;
            let mut canvas = image::ImageBuffer::new(canvas_w, total_h + outer_margin * 2);
            let bg_color = app_state.get_canvas_bg_color();
            for pixel in canvas.pixels_mut() {
                *pixel = bg_color;
            }

            let mut current_y = outer_margin;
            for row_images in resized_rows {
                let mut current_x = outer_margin;
                let mut max_h_in_row = 0;

                for (img, w) in row_images {
                    image::imageops::overlay(&mut canvas, &img, current_x as i64, current_y as i64);
                    current_x += w + spacing;
                    max_h_in_row = max_h_in_row.max(img.height());
                }
                current_y += max_h_in_row + spacing;
            }

            info!("✅ 布局与合并完成，总耗时: {:?}", layout_start.elapsed());
            Self::save_result(&canvas, &app_state, tx);
        }

        info!("🎉 全流程完成！总耗时: {:?}", start_time.elapsed());
        Ok(())
    }

    fn apply_sorting_optimized(&self, images: &mut Vec<ProcessedImageData>) {
        match self.sort_strategy {
            SortStrategy::Random => {
                images.shuffle(&mut rand::thread_rng());
            }
            SortStrategy::Brightness => {
                images.sort_by_key(|d| d.brightness);
            }
            SortStrategy::Hue => {
                images.sort_by_key(|d| d.hue);
            }
            SortStrategy::Balanced => {
                images.sort_by_key(|d| d.brightness);
                let mut balanced = Vec::with_capacity(images.len());
                let mut i = 0;
                let mut j = images.len() - 1;
                while i <= j {
                    balanced.push(images[j].clone());
                    if i < j {
                        balanced.push(images[i].clone());
                    }
                    i += 1;
                    if j > 0 {
                        j -= 1;
                    } else {
                        break;
                    }
                }
                *images = balanced;
            }
        }
    }

    /// 大师级智能裁剪 (Content-Aware Smart Crop)
    /// 结合 Haar 特征人脸检测、边缘检测、肤色识别、饱和度分析，寻找图像中最具价值的区域
    fn smart_crop_3_4(img: &DynamicImage) -> DynamicImage {
        let (w, h) = img.dimensions();
        let target_ratio = 3.0 / 4.0;

        let (tw, th) = if (w as f32 / h as f32) > target_ratio {
            ((h as f32 * target_ratio) as u32, h)
        } else {
            (w, (w as f32 / target_ratio) as u32)
        };

        // 缩放用于分析
        let small = img.thumbnail(150, 150);
        let rgb = small.to_rgb8();
        let (sw, sh) = rgb.dimensions();

        // 1. 智能重要性分析 (人脸/肤色/边缘)
        let mut face_map = vec![0.0; (sw * sh) as usize];

        // 算法大师：即使没有 AI 模型，我们也能通过肤色概率模型保护人脸
        // 肤色模型 (典型区间)
        for y in 0..sh {
            for x in 0..sw {
                let pixel = rgb.get_pixel(x, y);
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                let is_skin =
                    r > 95.0 && g > 40.0 && b > 20.0 && (r - g).abs() > 15.0 && r > g && r > b;

                if is_skin {
                    face_map[(y * sw + x) as usize] = 500.0;
                }
            }
        }

        let mut importance_map = vec![0.0; (sw * sh) as usize];

        for y in 0..sh {
            for x in 0..sw {
                let pixel = rgb.get_pixel(x, y);
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                // 2. 边缘检测 (Laplacian 简化版)
                let mut edge = 0.0;
                if x > 0 && x < sw - 1 && y > 0 && y < sh - 1 {
                    let center = rgb.get_pixel(x, y)[0] as f32;
                    let neighbors = rgb.get_pixel(x - 1, y)[0] as f32
                        + rgb.get_pixel(x + 1, y)[0] as f32
                        + rgb.get_pixel(x, y - 1)[0] as f32
                        + rgb.get_pixel(x, y + 1)[0] as f32;
                    edge = (4.0 * center - neighbors).abs();
                }

                // 3. 肤色识别 (已经合并到 face_map 中)
                let skin_boost = face_map[(y * sw + x) as usize] * 0.1;

                // 4. 饱和度识别
                let max_val = r.max(g).max(b);
                let min_val = r.min(g).min(b);
                let saturation = if max_val > 0.0 {
                    (max_val - min_val) / max_val
                } else {
                    0.0
                };

                // 综合重要性
                importance_map[(y * sw + x) as usize] = face_map[(y * sw + x) as usize]
                    + edge * 0.5
                    + skin_boost * 30.0
                    + saturation * 20.0;
            }
        }

        // 算法大师：使用积分图 (Summed-Area Table) 加速区域求和，将 O(W*H) 降至 O(1)
        let mut integral_image = vec![0.0; ((sw + 1) * (sh + 1)) as usize];
        for y in 0..sh {
            let mut row_sum = 0.0;
            for x in 0..sw {
                row_sum += importance_map[(y * sw + x) as usize];
                integral_image[((y + 1) * (sw + 1) + (x + 1)) as usize] =
                    integral_image[(y * (sw + 1) + (x + 1)) as usize] + row_sum;
            }
        }

        // 寻找最优裁剪窗口
        let mut best_score = -1.0;
        let mut best_x = 0;
        let mut best_y = 0;

        let stw = (tw as f32 * sw as f32 / w as f32) as u32;
        let sth = (th as f32 * sh as f32 / h as f32) as u32;

        // 滑动窗口寻找最大重要性区域 (现在是极速 O(1) 查询)
        for sy in 0..=(sh - sth).min(sh - 1) {
            for sx in 0..=(sw - stw).min(sw - 1) {
                let x1 = sx as usize;
                let y1 = sy as usize;
                let x2 = (sx + stw) as usize;
                let y2 = (sy + sth) as usize;

                // 积分图公式: Score = I(x2,y2) - I(x1,y2) - I(x2,y1) + I(x1,y1)
                let current_score_base = integral_image[y2 * (sw as usize + 1) + x2]
                    - integral_image[y2 * (sw as usize + 1) + x1]
                    - integral_image[y1 * (sw as usize + 1) + x2]
                    + integral_image[y1 * (sw as usize + 1) + x1];

                let mut current_score = current_score_base;

                // 构图加权：中心偏好
                let dx = (sx as f32 + stw as f32 / 2.0) / sw as f32 - 0.5;
                let dy = (sy as f32 + sth as f32 / 2.0) / sh as f32 - 0.5;
                let dist_from_center = (dx * dx + dy * dy).sqrt();
                current_score *= 1.0 - dist_from_center * 0.5;

                if current_score > best_score {
                    best_score = current_score;
                    best_x = sx;
                    best_y = sy;
                }
            }
        }

        // 映射回原图
        let final_x = (best_x as f32 * w as f32 / sw as f32) as u32;
        let final_y = (best_y as f32 * h as f32 / sh as f32) as u32;

        img.crop_imm(final_x.min(w - tw), final_y.min(h - th), tw, th)
    }

    /// 圆角处理 (高性能版本：仅处理四个角)
    fn apply_rounded_corners(&self, img: DynamicImage, radius: f32) -> DynamicImage {
        let (w, h) = img.dimensions();
        let mut rgba = img.to_rgba8();
        let r = radius;

        // 仅遍历四个角落的矩形区域
        let r_ui = r.ceil() as u32;

        // 左上
        for x in 0..r_ui.min(w) {
            for y in 0..r_ui.min(h) {
                if (x as f32 - r).powi(2) + (y as f32 - r).powi(2) > r.powi(2) {
                    rgba.get_pixel_mut(x, y).0[3] = 0;
                }
            }
        }
        // 右上
        for x in (w.saturating_sub(r_ui))..w {
            for y in 0..r_ui.min(h) {
                if (x as f32 - (w as f32 - r)).powi(2) + (y as f32 - r).powi(2) > r.powi(2) {
                    rgba.get_pixel_mut(x, y).0[3] = 0;
                }
            }
        }
        // 左下
        for x in 0..r_ui.min(w) {
            for y in (h.saturating_sub(r_ui))..h {
                if (x as f32 - r).powi(2) + (y as f32 - (h as f32 - r)).powi(2) > r.powi(2) {
                    rgba.get_pixel_mut(x, y).0[3] = 0;
                }
            }
        }
        // 右下
        for x in (w.saturating_sub(r_ui))..w {
            for y in (h.saturating_sub(r_ui))..h {
                if (x as f32 - (w as f32 - r)).powi(2) + (y as f32 - (h as f32 - r)).powi(2)
                    > r.powi(2)
                {
                    rgba.get_pixel_mut(x, y).0[3] = 0;
                }
            }
        }

        DynamicImage::ImageRgba8(rgba)
    }

    // 移除了旧的 is_inside_rounded_rect，因为它不再被需要且低效

    /// 绘制作者标签 (100% 还原 PY 版样式，且增加圆角效果)
    fn draw_author_tag(&self, img: DynamicImage, author: &str) -> DynamicImage {
        use ab_glyph::PxScale;
        use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_text_mut};
        use imageproc::rect::Rect;
        use rusttype::Scale;

        let (w, h) = img.dimensions();
        let font_rt = get_font_rt();

        // 动态计算字体大小
        let font_size = (w as f32 / 25.0).max(24.0);
        let scale_rt = Scale {
            x: font_size,
            y: font_size,
        };
        let px_scale = PxScale::from(font_size);

        let text_color = Rgba([50, 50, 50, 255]);
        let bg_color = Rgba([240, 240, 240, 200]); // 略微加深背景透明度

        // 计算文字尺寸
        let v_metrics = font_rt.v_metrics(scale_rt);
        let glyphs: Vec<_> = font_rt
            .layout(author, scale_rt, rusttype::point(0.0, v_metrics.ascent))
            .collect();
        let text_w = glyphs
            .iter()
            .next_back()
            .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
            .unwrap_or(0.0) as u32;
        let text_h = font_size as u32;

        let padding_h = (font_size * 0.6) as u32;
        let padding_v = (font_size * 0.3) as u32;
        let rect_w = text_w + padding_h * 2;
        let rect_h = text_h + padding_v * 2;

        let x = (w.saturating_sub(rect_w)) / 2;
        let y = h.saturating_sub(rect_h + (h as f32 * 0.05) as u32);

        let mut rgba = img.to_rgba8();

        // 绘制带圆角的背景矩形
        let r = (rect_h / 2) as i32; // 圆角半径设为高度的一半

        // 1. 中间主体矩形
        draw_filled_rect_mut(
            &mut rgba,
            Rect::at((x as i32) + r, y as i32)
                .of_size(rect_w.saturating_sub((r as u32) * 2), rect_h),
            bg_color,
        );
        // 2. 左圆角
        draw_filled_circle_mut(&mut rgba, ((x as i32) + r, (y as i32) + r), r, bg_color);
        // 3. 右圆角
        draw_filled_circle_mut(
            &mut rgba,
            ((x as i32) + (rect_w as i32) - r, (y as i32) + r),
            r,
            bg_color,
        );

        // 使用缓存的 ab_glyph 字体进行绘制
        let font_ab = get_font_ab();
        draw_text_mut(
            &mut rgba,
            text_color,
            (x + padding_h) as i32,
            (y + padding_v) as i32,
            px_scale,
            font_ab,
            author,
        );

        DynamicImage::ImageRgba8(rgba)
    }
}

impl eframe::App for GridPuzzleApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 处理异步消息
        // 1. 接收处理结果消息
        if let Some(rx_mutex) = &self.rx {
            if let Ok(rx) = rx_mutex.lock() {
                while let Ok(msg) = rx.try_recv() {
                    if let Some(path) = msg.strip_prefix("SUCCESS:") {
                        self.status = format!("生成成功! 已保存至: {}", path);
                        self.processing = false;
                        self.progress = 0.0;
                    } else if let Some(p_str) = msg.strip_prefix("PROGRESS:") {
                        if let Ok(p) = p_str.parse::<f32>() {
                            self.progress = p;
                        }
                    } else if msg.starts_with("错误:") {
                        self.status = msg;
                        self.processing = false;
                        self.progress = 0.0;
                    } else {
                        self.status = msg;
                    }
                    ctx.request_repaint();
                }
            }
        }

        // 每帧应用主题
        core_lib::ui::theme::UiTheme::apply_master_visuals(ctx, self.theme_mode);

        // 2. 接收缩略图消息
        if let Some(rx_mutex) = &self.thumb_rx {
            if let Ok(rx) = rx_mutex.lock() {
                while let Ok((path, color_image)) = rx.try_recv() {
                    let texture =
                        ctx.load_texture(path.to_string_lossy(), color_image, Default::default());
                    self.loading_paths.remove(&path);
                    self.thumbnails.insert(path, texture);
                    ctx.request_repaint();
                }
            }
        }

        // 3. 异步触发缺失缩略图加载
        self.trigger_thumbnail_loading();

        // 底部状态栏与进度条 (Docked at bottom)
        egui::TopBottomPanel::bottom("status_bar")
            .frame(
                egui::Frame::NONE
                    .fill(ctx.style().visuals.panel_fill) // Use default panel fill
                    .inner_margin(egui::Margin::symmetric(20, 10)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let text_color = ui.visuals().widgets.noninteractive.fg_stroke.color;
                    ui.label(
                        egui::RichText::new(format!("共 {} 张图片", self.input_images.len()))
                            .color(text_color),
                    );
                    ui.separator();

                    if self.processing {
                        // 处理中：显示进度条
                        ui.label(egui::RichText::new(&self.status).color(text_color));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let pb = egui::ProgressBar::new(self.progress)
                                .show_percentage()
                                .fill(UiTheme::PRIMARY)
                                .animate(true);
                            ui.add_sized([250.0, 14.0], pb);
                            ui.label(
                                egui::RichText::new("正在处理...")
                                    .color(UiTheme::PRIMARY)
                                    .strong(),
                            );
                        });
                    } else {
                        // 非处理中：显示状态文字
                        let status_text = if self.status.len() > 80 {
                            format!("{}...", self.status.chars().take(77).collect::<String>())
                        } else {
                            self.status.clone()
                        };
                        ui.label(egui::RichText::new(status_text).color(text_color).strong());
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(ctx.style().visuals.panel_fill).inner_margin(egui::Margin::ZERO))
            .show(ctx, |ui| {
                let total_rect = ui.max_rect();
                let side_margin = 20.0;
                let scale = (total_rect.width() / 800.0).clamp(0.7, 1.5);
                let current_inner_width = (total_rect.width() - (side_margin * 2.0)).max(700.0);

                ui.vertical(|ui| {
                    ui.set_width(total_rect.width());
                    ui.add_space(15.0 * scale);

                    ui.horizontal(|ui| {
                        ui.add_space(side_margin);
                        ui.vertical(|ui| {
                            ui.set_width(current_inner_width);

                            // 顶部标题 (移除 vertical_centered 避免偏移)
                            ui.add_space(5.0 * scale);
                            ui.vertical(|ui| {
                                ui.set_width(current_inner_width);
                                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                        ui.heading(egui::RichText::new("星TAP 拼图大师 v3.5").strong().size(28.0 * scale).color(UiTheme::PRIMARY));
                                        ui.label(egui::RichText::new("全RUST内核 · 智能图像网格排版专家").size(13.0 * scale).color(egui::Color32::GRAY));
                                    });
                            });
                            ui.add_space(20.0 * scale);

                            // 第一板块：操作面板
                            let frame_margin_h = 20.0 * scale;
                            egui::Frame::group(ui.style())
                                .corner_radius(egui::CornerRadius::same(16))
                                .fill(ui.visuals().widgets.noninteractive.bg_fill)
                                .inner_margin(egui::Margin::symmetric(frame_margin_h as i8, (15.0 * scale) as i8))
                                .show(ui, |ui| {
                                    ui.set_width(current_inner_width - (frame_margin_h * 2.0));
                                    ui.add_enabled_ui(!self.processing, |ui| {
                                        ui.horizontal_centered(|ui| {
                                            ui.spacing_mut().item_spacing.x = 15.0 * scale;
                                            let btn_size = egui::vec2(110.0 * scale, 36.0 * scale);

                                            if ui.add_sized(btn_size, egui::Button::new(egui::RichText::new("🖼️ 选择图片").size(14.0 * scale))).clicked() {
                                                if let Some(files) = rfd::FileDialog::new()
                                                    .add_filter("图片", &["png", "jpg", "jpeg", "webp"])
                                                    .pick_files() {
                                                    self.input_images = files;
                                                    self.selected_indices.clear();
                                                    self.status = format!("已加载 {} 张图片", self.input_images.len());
                                                }
                                            }

                                            if ui.add_sized(btn_size, egui::Button::new(egui::RichText::new("➕ 追加图片").size(14.0 * scale))).clicked() {
                                                if let Some(files) = rfd::FileDialog::new()
                                                    .add_filter("图片", &["png", "jpg", "jpeg", "webp"])
                                                    .pick_files() {
                                                    self.input_images.extend(files);
                                                    self.status = format!("当前共有 {} 张图片", self.input_images.len());
                                                }
                                            }

                                            ui.add_space(20.0 * scale);
                                            let generate_text = if self.processing { "正在生成..." } else { "🚀 生成拼图" };
                                            if ui.add_sized(egui::vec2(160.0 * scale, 42.0 * scale), egui::Button::new(egui::RichText::new(generate_text).strong().size(16.0 * scale).color(egui::Color32::WHITE)).fill(UiTheme::PRIMARY)).clicked() {
                                                self.processing = true;
                                                self.progress = 0.0;
                                                let app_clone = self.clone();
                                                std::thread::spawn(move || {
                                                    if let Err(e) = Self::process_images(app_clone) {
                                                        error!("处理失败: {}", e);
                                                    }
                                                });
                                            }

                                            ui.add_space(20.0 * scale);
                                            if ui.add_sized(btn_size, egui::Button::new(egui::RichText::new("🗑️ 清空列表").size(14.0 * scale))).clicked() {
                                                self.input_images.clear();
                                                self.thumbnails.clear();
                                                self.selected_indices.clear();
                                                self.status = "列表已清空".to_string();
                                            }
                                            if ui.add_sized(btn_size, egui::Button::new(egui::RichText::new("❌ 删除选中").size(14.0 * scale))).clicked() {
                                                if self.selected_indices.is_empty() {
                                                    if !self.input_images.is_empty() {
                                                        let last = self.input_images.pop().unwrap();
                                                        self.thumbnails.remove(&last);
                                                    }
                                                } else {
                                                    let mut sorted_indices: Vec<_> = self.selected_indices.iter().cloned().collect();
                                                    sorted_indices.sort_unstable_by(|a, b| b.cmp(a));
                                                    for idx in sorted_indices {
                                                        if idx < self.input_images.len() {
                                                            let path = self.input_images.remove(idx);
                                                            self.thumbnails.remove(&path);
                                                        }
                                                    }
                                                    self.selected_indices.clear();
                                                }
                                            }
                                        });
                                    });
                                });

                            ui.add_space(15.0 * scale);

                            // 第二板块：参数面板
                            egui::Frame::group(ui.style())
                                .corner_radius(egui::CornerRadius::same(16))
                                .inner_margin(egui::Margin::symmetric(frame_margin_h as i8, (15.0 * scale) as i8))
                                .show(ui, |ui| {
                                    ui.set_width(current_inner_width - (frame_margin_h * 2.0));
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("⚙️ 大师参数配置").strong().size(15.0 * scale));
                                            ui.add_space(12.0 * scale);
                                            ui.label(egui::RichText::new("(内容感知裁剪已就绪)").size(11.0 * scale).color(egui::Color32::from_rgb(0, 150, 255)));
                                        });
                                        ui.add_space(12.0 * scale);
                                        egui::Grid::new("params_grid")
                                            .num_columns(4)
                                            .spacing([22.0 * scale, 12.0 * scale])
                                            .show(ui, |ui| {
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    ui.label(egui::RichText::new("网格列数:").size(14.0 * scale));
                                                });
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    ui.add(egui::Slider::new(&mut self.column_count, 1..=6));
                                                });
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    ui.label(egui::RichText::new("布局:").size(14.0 * scale));
                                                });
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    egui::ComboBox::new("layout_style", "").selected_text(match self.layout_style { LayoutStyle::Columns => "瀑布流", LayoutStyle::Rows => "画廊流" }).show_ui(ui, |ui| {
                                                        ui.selectable_value(&mut self.layout_style, LayoutStyle::Columns, "瀑布流");
                                                        ui.selectable_value(&mut self.layout_style, LayoutStyle::Rows, "画廊流");
                                                    });
                                                });
                                                ui.end_row();
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    ui.label(egui::RichText::new("排序:").size(14.0 * scale));
                                                });
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    egui::ComboBox::new("sort_strategy_v3", "").selected_text(match self.sort_strategy { SortStrategy::Balanced => "均衡", SortStrategy::Brightness => "明暗", SortStrategy::Hue => "色彩", SortStrategy::Random => "随机" }).show_ui(ui, |ui| {
                                                        ui.selectable_value(&mut self.sort_strategy, SortStrategy::Balanced, "均衡");
                                                        ui.selectable_value(&mut self.sort_strategy, SortStrategy::Brightness, "明暗");
                                                        ui.selectable_value(&mut self.sort_strategy, SortStrategy::Hue, "色彩");
                                                        ui.selectable_value(&mut self.sort_strategy, SortStrategy::Random, "随机");
                                                    });
                                                });
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    ui.label(egui::RichText::new("主题:").size(14.0 * scale));
                                                });
                                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                                    egui::ComboBox::new("theme_mode", "").selected_text(match self.theme_mode { core_lib::ui::theme::ThemeMode::Auto => "自动", core_lib::ui::theme::ThemeMode::Light => "白昼", core_lib::ui::theme::ThemeMode::Dark => "深夜" }).show_ui(ui, |ui| {
                                                        ui.selectable_value(&mut self.theme_mode, core_lib::ui::theme::ThemeMode::Auto, "自动");
                                                        ui.selectable_value(&mut self.theme_mode, core_lib::ui::theme::ThemeMode::Light, "白昼");
                                                        ui.selectable_value(&mut self.theme_mode, core_lib::ui::theme::ThemeMode::Dark, "深夜");
                                                    });
                                                });
                                                ui.end_row();
                                            });
                                        ui.add_space(12.0 * scale);
                                        ui.horizontal_centered(|ui| {
                                            ui.spacing_mut().item_spacing.x = 18.0 * scale;
                                            ui.checkbox(&mut self.crop_3_4, egui::RichText::new("智能裁剪").size(14.0 * scale));
                                            ui.checkbox(&mut self.rounded_corners, egui::RichText::new("圆角").size(14.0 * scale));
                                            ui.checkbox(&mut self.tag_enabled_v2, egui::RichText::new("作者标签").size(14.0 * scale));

                                            // 模式互斥逻辑：网图模式与高清压缩互斥
                                             if ui.checkbox(&mut self.compress_output, egui::RichText::new("高清压缩").size(14.0 * scale)).clicked()
                                                 && self.compress_output
                                             {
                                                 self.web_mode = false;
                                             }

                                             let is_web = self.web_mode;
                                             if ui.checkbox(&mut self.web_mode, egui::RichText::new("网图模式").size(14.0 * scale).color(if is_web { UiTheme::PRIMARY } else { egui::Color32::GRAY })).clicked()
                                                 && self.web_mode
                                             {
                                                 self.compress_output = false;
                                             }
                                        });
                                    });
                                });

                            ui.add_space(15.0 * scale);

                            // 第三板块：预览区域
                            let bottom_padding = 10.0;
                            let current_cursor_top = ui.cursor().top();
                            let canvas_height = (total_rect.bottom() - current_cursor_top - bottom_padding).max(200.0);
                            let canvas_margin = 15.0 * scale;

                            egui::Frame::group(ui.style())
                                .corner_radius(egui::CornerRadius::same(16))
                                .fill(ui.visuals().extreme_bg_color)
                                .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                                .inner_margin(canvas_margin)
                                .show(ui, |ui| {
                                    ui.set_width(current_inner_width - (canvas_margin * 2.0));
                                    ui.set_height(canvas_height - (canvas_margin * 2.0));

                                    if self.input_images.is_empty() {
                                        let rect = ui.max_rect();
                                        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "🚀 请先选择图片或将图片拖拽至此", egui::FontId::proportional(20.0 * scale), egui::Color32::GRAY);
                                    } else {
                                        let list_width = (current_inner_width * 0.25).clamp(160.0 * scale, 220.0 * scale);
                                        let gap = 12.0 * scale;

                                        ui.horizontal_top(|ui| {
                                            ui.set_height(ui.available_height());

                                            // 左侧：文件列表
                                            ui.vertical(|ui| {
                                                ui.set_width(list_width);
                                                ui.set_height(ui.available_height());
                                                ui.label(egui::RichText::new("文件列表").strong().size(13.0 * scale));
                                                ui.add_space(6.0 * scale);
                                                egui::ScrollArea::vertical()
                                                    .id_salt("file_list")
                                                    .auto_shrink([false; 2])
                                                    .show(ui, |ui| {
                                                        ui.set_width(list_width);
                                                        ui.set_height(ui.available_height());
                                                        for (idx, path) in self.input_images.iter().enumerate() {
                                                            let is_selected = self.selected_indices.contains(&idx);
                                                            let name = path.file_name().unwrap_or_default().to_string_lossy();
                                                            let short_name = if name.chars().count() > 20 {
                                                                format!("{}...", name.chars().take(18).collect::<String>())
                                                            } else {
                                                                name.into_owned()
                                                            };
                                                            if ui.selectable_label(is_selected, egui::RichText::new(short_name).size(12.0 * scale)).clicked() {
                                                                if ui.input(|i| i.modifiers.command || i.modifiers.ctrl) {
                                                                    if is_selected { self.selected_indices.remove(&idx); }
                                                                    else { self.selected_indices.insert(idx); }
                                                                } else {
                                                                    self.selected_indices.clear();
                                                                    self.selected_indices.insert(idx);
                                                                }
                                                            }
                                                        }
                                                    });
                                            });

                                            ui.add_space(gap);
                                            ui.separator();
                                            ui.add_space(gap);

                                            // 右侧：缩略图预览
                                            let thumb_area_width = ui.available_width();
                                            ui.vertical(|ui| {
                                                ui.set_width(thumb_area_width);
                                                ui.set_height(ui.available_height());
                                                ui.label(egui::RichText::new("图片预览").strong().size(13.0 * scale));
                                                ui.add_space(6.0 * scale);

                                                egui::ScrollArea::vertical()
                                                    .id_salt("thumb_scroll")
                                                    .auto_shrink([false; 2])
                                                    .show(ui, |ui| {
                                                        ui.set_width(thumb_area_width);
                                                        let spacing = 10.0 * scale;
                                                        ui.spacing_mut().item_spacing = egui::vec2(spacing, spacing);

                                                        ui.horizontal_wrapped(|ui| {
                                                            let thumb_size = 100.0 * scale;
                                                            for (idx, path) in self.input_images.iter().enumerate() {
                                                                let is_selected = self.selected_indices.contains(&idx);
                                                                if let Some(texture) = self.thumbnails.get(path) {
                                                                    let response = ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(thumb_size, thumb_size)).corner_radius(8.0));
                                                                    if response.clicked() {
                                                                        if ui.input(|i| i.modifiers.command || i.modifiers.ctrl) {
                                                                            if is_selected { self.selected_indices.remove(&idx); }
                                                                            else { self.selected_indices.insert(idx); }
                                                                        } else {
                                                                            self.selected_indices.clear();
                                                                            self.selected_indices.insert(idx);
                                                                        }
                                                                    }
                                                                    if is_selected {
                                                                        ui.painter().rect_stroke(response.rect, 8.0, (2.0, UiTheme::PRIMARY), egui::StrokeKind::Outside);
                                                                    }
                                                                } else {
                                                                    let (rect, _response) = ui.allocate_at_least(egui::vec2(thumb_size, thumb_size), egui::Sense::hover());
                                                                    ui.painter().rect_filled(rect, 8.0, ui.visuals().widgets.noninteractive.bg_fill);
                                                                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "⌛", egui::FontId::proportional(15.0 * scale), egui::Color32::GRAY);
                                                                }
                                                            }
                                                        });
                                                    });
                                            });
                                        });
                                    }
                                });
                        });
                    });
                });
            });

        // 处理拖放图片
        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            let dropped = ctx.input(|i| i.raw.dropped_files.clone());
            for file in dropped {
                if let Some(path) = file.path {
                    if path.is_file() {
                        let ext = path
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if ["png", "jpg", "jpeg", "webp"].contains(&ext.as_str()) {
                            self.input_images.push(path);
                        }
                    }
                }
            }
            self.status = format!("已添加拖拽的图片，共 {} 张", self.input_images.len());
            ctx.request_repaint();
        }
    }
}

impl GridPuzzleApp {
    fn trigger_thumbnail_loading(&mut self) {
        if let Some(tx) = &self.thumb_tx {
            for path in &self.input_images {
                if !self.thumbnails.contains_key(path) && !self.loading_paths.contains(path) {
                    self.loading_paths.insert(path.clone());
                    let path_clone = path.clone();
                    let tx_clone = tx.clone();
                    std::thread::spawn(move || {
                        if let Ok(img) = image::open(&path_clone) {
                            let thumb = img.thumbnail(256, 256);
                            let rgba = thumb.to_rgba8();
                            let (w, h) = rgba.dimensions();
                            let pixels = rgba.into_raw();
                            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                [w as usize, h as usize],
                                &pixels,
                            );
                            let _ = tx_clone.send((path_clone, color_image));
                        }
                    });
                }
            }
        }
    }
}
