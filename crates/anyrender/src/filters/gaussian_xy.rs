//! Independent-axis Gaussian blur parameters, and the error of the checked filter path.

use kurbo::Rect;

/// Why a renderer's checked filter path refused a filter; it never draws the content unfiltered instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    /// The renderer does not draw this kind of filter.
    Unsupported(&'static str),
    /// Malformed or non-finite input.
    Invalid(&'static str),
    /// A resource limit the caller set would be exceeded.
    Budget(&'static str),
}
impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FilterError {}

/// A Gaussian blur of the source graphic with independent axes, a local output crop and a colour space.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianBlurXY {
    /// Standard deviation in local X and Y units.
    pub std_deviation: [f32; 2],
    /// Hard local output crop, independent from the input halo.
    pub region: Rect,
    /// True for linearRGB; false for explicit sRGB working values.
    pub linear_rgb: bool,
}
impl GaussianBlurXY {
    /// Validates finite non-negative standard deviations and a finite region. Size limits are the renderer's.
    pub fn new(
        std_deviation: [f32; 2],
        region: Rect,
        linear_rgb: bool,
    ) -> Result<Self, FilterError> {
        if std_deviation.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(FilterError::Invalid("stdDeviation"));
        }

        if ![region.x0, region.y0, region.x1, region.y1]
            .iter()
            .all(|v| v.is_finite())
            || region.width() < 0.0
            || region.height() < 0.0
        {
            return Err(FilterError::Invalid("filter region"));
        }
        Ok(Self {
            std_deviation,
            region,
            linear_rgb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_independent_and_zero_axes() {
        let region = kurbo::Rect::new(-40.0, -40.0, 48.0, 48.0);
        for sigma in [[12.0, 0.0], [0.0, 12.0], [0.0, 0.0]] {
            let blur = GaussianBlurXY::new(sigma, region, true).unwrap();
            assert_eq!(blur.std_deviation, sigma);
        }
    }
    #[test]
    fn rejects_invalid_and_over_budget_parameters() {
        let region = kurbo::Rect::new(0.0, 0.0, 8.0, 8.0);
        for sigma in [
            [f32::NAN, 0.0],
            [-1.0, 1.0],
            [65.0, 0.0],
            [0.0, f32::INFINITY],
        ] {
            assert!(GaussianBlurXY::new(sigma, region, true).is_err());
        }
        assert!(
            GaussianBlurXY::new([1.0, 1.0], kurbo::Rect::new(f64::NAN, 0.0, 1.0, 1.0), true)
                .is_err()
        );
    }
}
