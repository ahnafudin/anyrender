//! The checked conversion of an [`anyrender`] Gaussian filter chain into a [`vello_common`] filter.
//!
//! It depends on no GPU crate, so a process that only records paint commands can apply the admission of the
//! `anyrender_vello_gpu` painter that draws them.
//!
//! All of it is under the `ahcut` feature, which is off by default: the filter it converts exists in
//! `anyrender` only with that feature.
#![cfg(feature = "ahcut")]

use anyrender::filters::{Filter, FilterError, GaussianBlurXY};
use vello_common::filter_effects::FilterPrimitive;

/// Limits of the checked filter path. Exceeding one is [`FilterError::Budget`], never a clamped or unfiltered
/// result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckedFilterLimits {
    /// The largest device-space standard deviation of either axis.
    pub max_device_sigma: f32,
    /// The largest sum, over a chain's stages, of the RGBA16 scratch an intermediate layer may need.
    pub max_scratch_bytes: f64,
    /// The smallest padding, per axis, assumed around each stage's region when estimating scratch.
    pub min_padding: f64,
}

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
pub fn convert_filter_checked(
    filter: &Filter,
    transform: kurbo::Affine,
) -> Result<vello_common::filter_effects::Filter, FilterError> {
    convert_filter_checked_with(filter, transform, &CheckedFilterLimits::default())
}

/// Translates a whole Gaussian chain, or refuses it: every edge, primitive and transform is validated, and the
/// scratch the serial stages keep at once is bounded, before anything reaches the GPU.
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyrender::filters::{FilterEffect, FilterInputs};
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
