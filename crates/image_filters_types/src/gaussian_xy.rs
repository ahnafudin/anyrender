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
    /// A measured device-space cost exceeds the renderer's limit for this field.
    BudgetExceeded(FilterBudgetExceeded),
}
impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Self::BudgetExceeded(budget) = self {
            let (field, unit) = match budget.field {
                FilterBudgetField::DeviceSigmaX => ("device sigma X", "px"),
                FilterBudgetField::DeviceSigmaY => ("device sigma Y", "px"),
                FilterBudgetField::HaloX => ("one-sided halo X", "px"),
                FilterBudgetField::HaloY => ("one-sided halo Y", "px"),
                FilterBudgetField::PaddedWidth => ("padded width", "px"),
                FilterBudgetField::PaddedHeight => ("padded height", "px"),
                FilterBudgetField::ScratchBytes => ("aggregate scratch", "bytes"),
            };
            return write!(
                f,
                "{field}: {} {unit} exceeds {} {unit}",
                budget.supplied, budget.limit
            );
        }
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FilterError {}

/// The independently checked quantity in a Gaussian chain's resource budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterBudgetField {
    /// Horizontal standard deviation after applying the drawing transform, in pixels.
    DeviceSigmaX,
    /// Vertical standard deviation after applying the drawing transform, in pixels.
    DeviceSigmaY,
    /// Accumulated dependency reach to either side along X, in device pixels.
    HaloX,
    /// Accumulated dependency reach to either side along Y, in device pixels.
    HaloY,
    /// Device width of a stage including both sides' scratch padding, in pixels.
    PaddedWidth,
    /// Device height of a stage including both sides' scratch padding, in pixels.
    PaddedHeight,
    /// Total estimated scratch over the stages considered so far, in bytes.
    ScratchBytes,
}

/// Values from the comparison that refused a filter. Pixel costs may be fractional.
#[derive(Debug, Clone, Copy)]
pub struct FilterBudgetExceeded {
    /// Which resource exceeded its limit.
    pub field: FilterBudgetField,
    /// Measured cost in the field's units.
    pub supplied: f64,
    /// Largest admitted value in the same units.
    pub limit: f64,
}

// FilterError has total equality. Compare the exact reported values without rounding them.
impl PartialEq for FilterBudgetExceeded {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.supplied.to_bits() == other.supplied.to_bits()
            && self.limit.to_bits() == other.limit.to_bits()
    }
}
impl Eq for FilterBudgetExceeded {}

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
    // A large but finite deviation is valid here: the size limits are the renderer's.
    #[test]
    fn rejects_invalid_parameters() {
        let region = kurbo::Rect::new(0.0, 0.0, 8.0, 8.0);
        for sigma in [[f32::NAN, 0.0], [-1.0, 1.0], [0.0, f32::INFINITY]] {
            assert!(GaussianBlurXY::new(sigma, region, true).is_err());
        }
        assert!(
            GaussianBlurXY::new([1.0, 1.0], kurbo::Rect::new(f64::NAN, 0.0, 1.0, 1.0), true)
                .is_err()
        );
    }
}
