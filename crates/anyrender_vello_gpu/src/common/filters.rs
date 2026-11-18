use anyrender::filters::{EdgeMode, Filter, FilterEffect};
#[cfg(feature = "ahcut")]
use anyrender::filters::{FilterError, GaussianBlurXY};
use vello_common::filter_effects::FilterPrimitive;

/// Limits of the checked filter path. Exceeding one is [`FilterError::Budget`], never a clamped or unfiltered
/// result.
#[cfg(feature = "ahcut")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckedFilterLimits {
    /// The largest device-space standard deviation of either axis.
    pub max_device_sigma: f32,
    /// The largest sum, over a chain's stages, of the RGBA16 scratch an intermediate layer may need.
    pub max_scratch_bytes: f64,
    /// The smallest padding, per axis, assumed around each stage's region when estimating scratch.
    pub min_padding: f64,
}

#[cfg(feature = "ahcut")]
impl Default for CheckedFilterLimits {
    /// Conservative bounds for interactive use: 64 device pixels of sigma, 64 MiB of scratch, 384 pixels of padding.
    fn default() -> Self {
        Self {
            max_device_sigma: 64.0,
            max_scratch_bytes: 64.0 * 1024.0 * 1024.0,
            min_padding: 384.0,
        }
    }
}

/// [`convert_filter_checked_with`] under [`CheckedFilterLimits::default`].
#[cfg(feature = "ahcut")]
pub fn convert_filter_checked(
    filter: &Filter,
    transform: kurbo::Affine,
) -> Result<vello_common::filter_effects::Filter, FilterError> {
    convert_filter_checked_with(filter, transform, &CheckedFilterLimits::default())
}

/// Translates a whole Gaussian chain, or refuses it: every edge, primitive and transform is validated, and the
/// scratch the serial stages keep at once is bounded, before anything reaches the GPU.
#[cfg(feature = "ahcut")]
pub fn convert_filter_checked_with(
    filter: &Filter,
    transform: kurbo::Affine,
    limits: &CheckedFilterLimits,
) -> Result<vello_common::filter_effects::Filter, FilterError> {
    use vello_common::filter::gaussian_xy::{GaussianXYError, transform_sigma};
    let stages = filter.checked_gaussian_stages()?;
    let final_region = filter
        .output_region()
        .ok_or(FilterError::Invalid("missing final crop"))?;
    let mut converted = Vec::with_capacity(stages.len());
    let mut scratch_bytes = 0.0;
    let mut dependency_x = 0.0;
    let mut dependency_y = 0.0;
    for blur in &stages {
        GaussianBlurXY::new(blur.std_deviation, blur.region, blur.linear_rgb)?;
        let sigma =
            transform_sigma(blur.std_deviation, &transform).map_err(|error| match error {
                GaussianXYError::InvalidSigma => FilterError::Invalid("Gaussian sigma"),
                GaussianXYError::UnsupportedTransform => {
                    FilterError::Unsupported("Gaussian covariance transform")
                }
            })?;
        if sigma.iter().any(|v| *v > limits.max_device_sigma) {
            return Err(FilterError::Budget("device-space sigma"));
        }
        dependency_x += f64::from(sigma[0]) * 3.0;
        dependency_y += f64::from(sigma[1]) * 3.0;
        let region = transform.transform_rect_bbox(blur.region.union(final_region));
        let width = region.width().ceil();
        let height = region.height().ceil();
        // Each stage is charged its padded region, including the serial dependency halo; the chain keeps all of
        // its stages at once, so the charges add up.
        let padding_x = limits.min_padding.max(2.0 * dependency_x);
        let padding_y = limits.min_padding.max(2.0 * dependency_y);
        if ![width, height].iter().all(|v| v.is_finite() && *v >= 0.0)
            || width + padding_x > f64::from(u16::MAX)
            || height + padding_y > f64::from(u16::MAX)
        {
            return Err(FilterError::Budget("filter scratch region"));
        }
        scratch_bytes += (width + padding_x) * (height + padding_y) * 8.0;
        if !scratch_bytes.is_finite() || scratch_bytes > limits.max_scratch_bytes {
            return Err(FilterError::Budget("filter scratch region"));
        }
        converted.push((
            FilterPrimitive::GaussianBlurXY {
                std_deviation: blur.std_deviation,
                linear_rgb: blur.linear_rgb,
            },
            blur.region,
        ));
    }
    if stages.len() == 1 && stages[0].region == final_region {
        // A single stage whose crop is the final crop needs one output mask, not two.
        return Ok(vello_common::filter_effects::Filter::from_primitive(
            converted.remove(0).0,
        ));
    }
    vello_common::filter_effects::Filter::gaussian_chain(&converted, final_region)
        .map_err(|_| FilterError::Invalid("Gaussian chain translation"))
}

pub(crate) fn convert_filter(filter: &Filter) -> Option<vello_common::filter_effects::Filter> {
    let nodes = filter.nodes();
    if nodes.is_empty() {
        return None;
    }

    // Vello GPU only supports single-node filters at the moment
    let node = &filter.nodes()[0];
    let primitive = convert_filter_effect(&node.effect)?;
    Some(vello_common::filter_effects::Filter::from_primitive(
        primitive,
    ))
}

pub(crate) fn convert_filter_effect(effect: &FilterEffect) -> Option<FilterPrimitive> {
    Some(match effect {
        #[cfg(feature = "ahcut")]
        FilterEffect::GaussianBlurXY(blur) => FilterPrimitive::GaussianBlurXY {
            std_deviation: blur.std_deviation,
            linear_rgb: blur.linear_rgb,
        },
        FilterEffect::Flood(color) => FilterPrimitive::Flood { color: *color },
        FilterEffect::GaussianBlur(blur) => FilterPrimitive::GaussianBlur {
            std_deviation_x: blur.std_deviation_x,
            std_deviation_y: blur.std_deviation_y,
            edge_mode: convert_edge_mode(blur.edge_mode),
        },
        // Vello has one standard deviation for both axes.
        FilterEffect::DropShadow(shadow) if shadow.std_deviation_x != shadow.std_deviation_y => {
            return None;
        }
        FilterEffect::DropShadow(shadow) => FilterPrimitive::DropShadow {
            dx: shadow.dx,
            dy: shadow.dy,
            std_deviation: shadow.std_deviation_x,
            color: shadow.color,
            edge_mode: convert_edge_mode(shadow.edge_mode),
        },
        FilterEffect::Offset(offset) => FilterPrimitive::Offset {
            dx: offset.x as f32,
            dy: offset.y as f32,
        },
        FilterEffect::ColorMatrix(_matrix) => return None, //FilterPrimitive::ColorMatrix { matrix: matrix.0 },
        FilterEffect::Blend(_mode) => return None,         //FilterPrimitive::Blend { mode: *mode },
        FilterEffect::ComponentTransfer(_component_transfer_filter) => return None,
        FilterEffect::Composite(_composite_operator) => return None,
        FilterEffect::Morphology(_morphology_filter) => return None,
        FilterEffect::ConvolveMatrix(_convolution_kernel) => return None,
        FilterEffect::Turbulence(_turbulence_filter) => return None,
        FilterEffect::DisplacementMap(_displacement_map_filter) => return None,
        FilterEffect::Image(_external_image_source) => return None,
        FilterEffect::Tile => return None,
        FilterEffect::DiffuseLighting(_diffuse_lighting_filter) => return None,
        FilterEffect::SpecularLighting(_specular_lighting_filter) => return None,
        // Kinds behind optional anyrender features that this crate was built without.
        #[allow(
            unreachable_patterns,
            reason = "anyrender's optional features add kinds"
        )]
        _ => return None,
    })
}

fn convert_edge_mode(edge_mode: EdgeMode) -> vello_common::filter_effects::EdgeMode {
    match edge_mode {
        EdgeMode::Duplicate => vello_common::filter_effects::EdgeMode::Duplicate,
        EdgeMode::Wrap => vello_common::filter_effects::EdgeMode::Wrap,
        EdgeMode::Mirror => vello_common::filter_effects::EdgeMode::Mirror,
        EdgeMode::None => vello_common::filter_effects::EdgeMode::None,
    }
}

#[cfg(all(test, feature = "ahcut"))]
mod checked_tests {
    use super::*;
    use anyrender::filters::{FilterInputs, GaussianBlurXY};
    use kurbo::{Affine, Rect};
    #[test]
    fn refuses_truncation() {
        let mut graph = Filter::single(FilterEffect::blur(1.0), Rect::ZERO);
        graph.add(FilterEffect::blur(2.0), FilterInputs::NONE);
        assert!(convert_filter_checked(&graph, Affine::IDENTITY).is_err());
    }
    #[test]
    fn admits_xy_and_rejects_general_covariance() {
        let region = Rect::new(-40.0, -40.0, 48.0, 48.0);
        let f = Filter::single(
            FilterEffect::GaussianBlurXY(GaussianBlurXY::new([12.0, 0.0], region, true).unwrap()),
            region,
        );
        assert!(convert_filter_checked(&f, Affine::IDENTITY).is_ok());
        assert!(convert_filter_checked(&f, Affine::new([1.0, 0.5, 0.0, 1.0, 0.0, 0.0])).is_err());
    }
    #[test]
    fn chain_preserves_each_crop_links_and_transform_refusals() {
        let final_crop = Rect::new(-40.0, -36.0, 48.0, 44.0);
        let narrow = Rect::new(0.0, 0.0, 4.0, 8.0);
        let graph = Filter::gaussian_chain(
            &[
                GaussianBlurXY::new([1.0, 0.0], final_crop, false).unwrap(),
                GaussianBlurXY::new([0.0, 0.0], narrow, false).unwrap(),
            ],
            final_crop,
        )
        .unwrap();
        let converted = convert_filter_checked(&graph, Affine::translate((120.0, 44.0))).unwrap();
        let stages = converted.graph.checked_gaussian_chain().unwrap().unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].1, final_crop);
        assert_eq!(stages[1].1, narrow);
        assert_eq!(converted.graph.gaussian_final_region(), Some(final_crop));
        assert!(
            convert_filter_checked(&graph, Affine::new([1.0, 0.5, 0.0, 1.0, 0.0, 0.0])).is_err()
        );
        assert!(convert_filter_checked(&graph, Affine::scale(100.0)).is_err());
    }
    #[test]
    fn aggregate_scratch_cap_is_not_reset_for_the_second_stage() {
        let crop = Rect::new(0.0, 0.0, 2000.0, 2000.0);
        let stages = vec![GaussianBlurXY::new([1.0, 0.0], crop, false).unwrap(); 2];
        let graph = Filter::gaussian_chain(&stages, crop).unwrap();
        assert_eq!(
            convert_filter_checked(&graph, Affine::IDENTITY),
            Err(FilterError::Budget("filter scratch region"))
        );
    }
}
