fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        // 使用 embed-resource 处理 Windows 资源
        // 它会自动寻找 rc.exe 或 llvm-rc
        embed_resource::compile("resources.rc", embed_resource::NONE);
    }
}
