//! Code shared by the wgpu and WebGL backends.

mod filters;
mod image_cache;
mod options;
mod scene;

pub(crate) use filters::convert_filter;
#[cfg(feature = "ahcut")]
pub use filters::{CheckedFilterLimits, convert_filter_checked, convert_filter_checked_with};
pub use image_cache::ImageCache;
pub use options::VelloGpuRendererOptions;
pub(crate) use scene::{DEFAULT_TOLERANCE, LayerKind, supports_glyph_caching};
