//! 模型数据加载模块
//! 负责从文件系统加载3D模型数据

use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::constants::Vec2;
use crate::constants::Vec3;
use crate::types::{AppData, ModelVertex};

/// 模型加载配置
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// 模型文件路径
    pub path: String,
    /// 是否三角化网格
    pub triangulate: bool,
    /// 是否去除重复顶点
    pub deduplicate: bool,
    /// 默认顶点颜色
    pub default_color: Vec3,
}

impl ModelConfig {
    /// 创建默认模型配置
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            triangulate: true,
            deduplicate: true,
            default_color: Vec3::new(1.0, 1.0, 1.0), // 白色
        }
    }

    /// 设置是否三角化
    pub fn with_triangulate(mut self, triangulate: bool) -> Self {
        self.triangulate = triangulate;
        self
    }

    /// 设置是否去重
    pub fn with_deduplicate(mut self, deduplicate: bool) -> Self {
        self.deduplicate = deduplicate;
        self
    }

    /// 设置默认颜色
    pub fn with_default_color(mut self, color: Vec3) -> Self {
        self.default_color = color;
        self
    }
}

/// 加载模型数据
/// 从OBJ文件读取顶点和索引数据
pub fn load_model_data(data: &mut AppData, config: ModelConfig) -> Result<()> {
    let mut reader = BufReader::new(File::open(&config.path)?);
    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            triangulate: config.triangulate,
            ..Default::default()
        },
        |_| Ok(Default::default()),
    )?;

    // 清空现有数据
    data.vertices.clear();
    data.indices.clear();

    if config.deduplicate {
        load_with_deduplication(data, &models, &config)?;
    } else {
        load_without_deduplication(data, &models, &config)?;
    }

    log::info!(
        "模型加载完成 ({}): {} 顶点, {} 索引 (去重: {})",
        config.path,
        data.vertices.len(),
        data.indices.len(),
        config.deduplicate
    );

    Ok(())
}

/// 加载模型数据并去除重复顶点
fn load_with_deduplication(
    data: &mut AppData,
    models: &[tobj::Model],
    config: &ModelConfig,
) -> Result<()> {
    let mut unique_vertices = HashMap::new();

    for model in models {
        for &index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            // 确保索引在范围内
            if pos_offset + 2 >= model.mesh.positions.len() {
                continue;
            }
            if tex_coord_offset + 1 >= model.mesh.texcoords.len() {
                continue;
            }

            let vertex = ModelVertex::new(
                Vec3::new(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                config.default_color,
                Vec2::new(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1], // 翻转Y坐标
                ),
            );

            if let Some(&existing_index) = unique_vertices.get(&vertex) {
                data.indices.push(existing_index as u32);
            } else {
                let new_index = data.vertices.len();
                unique_vertices.insert(vertex, new_index);
                data.vertices.push(vertex);
                data.indices.push(new_index as u32);
            }
        }
    }

    Ok(())
}

/// 加载模型数据但不去重
fn load_without_deduplication(
    data: &mut AppData,
    models: &[tobj::Model],
    config: &ModelConfig,
) -> Result<()> {
    for model in models {
        for &index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            // 确保索引在范围内
            if pos_offset + 2 >= model.mesh.positions.len() {
                continue;
            }
            if tex_coord_offset + 1 >= model.mesh.texcoords.len() {
                continue;
            }

            let vertex = ModelVertex::new(
                Vec3::new(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                config.default_color,
                Vec2::new(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1], // 翻转Y坐标
                ),
            );

            data.vertices.push(vertex);
            data.indices.push(data.vertices.len() as u32 - 1);
        }
    }

    Ok(())
}

/// 获取模型统计信息
pub fn get_model_stats(data: &AppData) -> ModelStats {
    ModelStats {
        vertex_count: data.vertices.len(),
        index_count: data.indices.len(),
        triangle_count: data.indices.len() / 3,
    }
}

/// 模型统计信息
#[derive(Debug, Clone)]
pub struct ModelStats {
    pub vertex_count: usize,
    pub index_count: usize,
    pub triangle_count: usize,
}

impl std::fmt::Display for ModelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "顶点: {}, 索引: {}, 三角形: {}",
            self.vertex_count, self.index_count, self.triangle_count
        )
    }
}
