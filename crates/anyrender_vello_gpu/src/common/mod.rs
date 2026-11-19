//! Code shared by the wgpu and WebGL backends.

mod filters;
mod image_cache;
mod options;
mod scene;

#[cfg(feature = "ahcut")]
pub use anyrender_vello_filters::{
    CheckedFilterLimits, convert_filter_checked, convert_filter_checked_with,
};
pub(crate) use filters::convert_filter;
pub use image_cache::ImageCache;
pub use options::VelloGpuRendererOptions;
pub(crate) use scene::{DEFAULT_TOLERANCE, LayerKind, supports_glyph_caching};
