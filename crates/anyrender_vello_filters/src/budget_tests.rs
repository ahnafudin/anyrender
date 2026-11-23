//! Boundary tests for the values returned by the checked Gaussian converter.

use super::*;
use kurbo::{Affine, Rect};

fn graph(sigma: [f32; 2], width: f64, height: f64, count: usize) -> Filter {
    let crop = Rect::new(0.0, 0.0, width, height);
    let stages = vec![GaussianBlurXY::new(sigma, crop, false).unwrap(); count];
    Filter::gaussian_chain(&stages, crop).unwrap()
}

fn exceeded(field: FilterBudgetField, supplied: f64, limit: f64) -> FilterError {
    FilterError::BudgetExceeded(FilterBudgetExceeded {
        field,
        supplied,
        limit,
    })
}

#[test]
fn each_device_sigma_uses_the_transformed_value() {
    for (sigma, field) in [
        ([32.5, 0.0], FilterBudgetField::DeviceSigmaX),
        ([0.0, 32.5], FilterBudgetField::DeviceSigmaY),
    ] {
        let exact = sigma.map(|v: f32| v.min(32.0));
        assert!(convert_filter_checked(&graph(exact, 8.0, 8.0, 1), Affine::scale(2.0)).is_ok());
        assert_eq!(
            convert_filter_checked(&graph(sigma, 8.0, 8.0, 1), Affine::scale(2.0)).unwrap_err(),
            exceeded(field, 65.0, 64.0),
        );
    }
}

#[test]
fn each_one_sided_halo_adds_across_stages_independently_of_scratch_padding() {
    let limits = CheckedFilterLimits::default();
    for (sigma, field) in [
        ([64.0, 0.0], FilterBudgetField::HaloX),
        ([0.0, 64.0], FilterBudgetField::HaloY),
    ] {
        let chain = graph(sigma, 8.0, 8.0, 2);
        assert!(convert_filter_checked(&chain, Affine::IDENTITY).is_ok());
        assert_eq!(
            convert_filter_checked_with_halo_limit(&chain, Affine::IDENTITY, &limits, 383.0)
                .unwrap_err(),
            exceeded(field, 384.0, 383.0),
        );
        // A 385 px halo cannot be reached with two stages of at most 64 px sigma each.
        assert_eq!(check_limit(field, 384.0, DEFAULT_MAX_HALO), Ok(()));
        assert_eq!(
            check_limit(field, 385.0, DEFAULT_MAX_HALO),
            Err(exceeded(field, 385.0, 384.0))
        );
    }
}

#[test]
fn padded_extent_reports_the_dimension_before_charging_scratch() {
    let limits = CheckedFilterLimits {
        max_scratch_bytes: f64::INFINITY,
        ..Default::default()
    };
    for (width, height, field) in [
        (65_152.0, 1.0, FilterBudgetField::PaddedWidth),
        (1.0, 65_152.0, FilterBudgetField::PaddedHeight),
    ] {
        let convert = |w, h| {
            convert_filter_checked_with(&graph([1.0, 0.0], w, h, 1), Affine::IDENTITY, &limits)
        };
        assert!(convert(f64::min(width, 65_151.0), f64::min(height, 65_151.0)).is_ok());
        assert_eq!(
            convert(width, height).unwrap_err(),
            exceeded(field, 65_536.0, 65_535.0)
        );
    }
}

#[test]
fn aggregate_scratch_accepts_the_exact_budget_and_reports_the_full_cost() {
    let exact = graph([1.0, 0.0], 1_664.0, 3_712.0, 1);
    let wider = graph([1.0, 0.0], 1_665.0, 3_712.0, 1);
    assert!(convert_filter_checked(&exact, Affine::IDENTITY).is_ok());
    assert_eq!(
        convert_filter_checked(&wider, Affine::IDENTITY).unwrap_err(),
        exceeded(FilterBudgetField::ScratchBytes, 67_141_632.0, 67_108_864.0),
    );
    let limits = CheckedFilterLimits {
        max_scratch_bytes: 67_108_863.0,
        ..Default::default()
    };
    assert_eq!(
        convert_filter_checked_with(&exact, Affine::IDENTITY, &limits).unwrap_err(),
        exceeded(FilterBudgetField::ScratchBytes, 67_108_864.0, 67_108_863.0),
    );
}

#[test]
fn diagnostic_preserves_fractional_measurements() {
    let chain = graph([0.0, 64.25], 1.0, 1.0, 1);
    let limits = CheckedFilterLimits {
        max_device_sigma: 64.125,
        ..Default::default()
    };
    assert_eq!(
        convert_filter_checked_with(&chain, Affine::IDENTITY, &limits).unwrap_err(),
        exceeded(FilterBudgetField::DeviceSigmaY, 64.25, 64.125),
    );
}

#[test]
fn invalid_bounds_and_unsupported_transforms_do_not_report_a_measured_budget() {
    let huge = graph([0.0, 0.0], f64::MAX, 8.0, 1);
    assert!(matches!(
        convert_filter_checked(&huge, Affine::scale(2.0)),
        Err(FilterError::Invalid(_))
    ));
    let ordinary = graph([2.0, 1.0], 8.0, 8.0, 1);
    assert!(matches!(
        convert_filter_checked(&ordinary, Affine::rotate(0.3)),
        Err(FilterError::Unsupported(_))
    ));
}
