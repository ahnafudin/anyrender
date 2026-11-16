//! Tests of the serial Gaussian chain: links, crops and refusals.
use super::*;
use kurbo::{Affine, Rect};

fn blur(sigma: [f32; 2], region: Rect) -> GaussianBlurXY {
    GaussianBlurXY::new(sigma, region, false).unwrap()
}
#[test]
fn two_explicit_links_preserve_stages_and_accumulate_dependency_halo() {
    let crop = Rect::new(80.0, 8.0, 168.0, 88.0);
    let stages = [blur([12.0, 0.0], crop), blur([0.0, 12.0], crop)];
    let f = Filter::gaussian_chain(&stages, crop).unwrap();
    assert_eq!(f.output(), FilterId(1));
    assert_eq!(
        f.nodes()[0].inputs.primary,
        Some(FilterInput::Source(FilterSource::SourceGraphic))
    );
    assert_eq!(
        f.nodes()[1].inputs.primary,
        Some(FilterInput::Result(FilterId(0)))
    );
    assert_eq!(
        f.checked_gaussian_stages().unwrap(),
        stages.iter().collect::<Vec<_>>()
    );
    assert_eq!(f.output_region(), Some(crop));
    let repeated =
        Filter::gaussian_chain(&[blur([12.0, 0.0], crop), blur([12.0, 0.0], crop)], crop).unwrap();
    assert_eq!(repeated.bounds_expansion(&Affine::IDENTITY).x1, 72.0);
}
#[test]
fn stage_crop_is_not_final_crop_and_single_zero_remains_zero() {
    let final_crop = Rect::new(80.0, 8.0, 168.0, 88.0);
    let stage_crop = Rect::new(120.0, 44.0, 124.0, 52.0);
    let f = Filter::gaussian_chain(&[blur([0.0, 0.0], stage_crop)], final_crop).unwrap();
    assert_eq!(f.checked_gaussian_stages().unwrap()[0].region, stage_crop);
    assert_eq!(f.output_region(), Some(final_crop));
    assert_eq!(f.expansion_rect(), Rect::ZERO);
}
#[test]
fn mutated_links_output_count_and_general_effects_are_refused() {
    let crop = Rect::new(0.0, 0.0, 40.0, 40.0);
    let valid =
        Filter::gaussian_chain(&[blur([1.0, 0.0], crop), blur([0.0, 1.0], crop)], crop).unwrap();
    for primary in [
        None,
        Some(FilterInput::Source(FilterSource::SourceGraphic)),
        Some(FilterInput::Result(FilterId(1))),
    ] {
        let mut f = valid.clone();
        f.primitives[1].inputs.primary = primary;
        assert!(f.checked_gaussian_stages().is_err());
    }
    let mut f = valid.clone();
    f.output = FilterId(0);
    assert!(f.checked_gaussian_stages().is_err());
    let mut f = valid.clone();
    f.primitives[0].inputs.primary = None;
    assert!(f.checked_gaussian_stages().is_err());
    let mut f = valid.clone();
    f.primitives[1].inputs.secondary = Some(FilterInput::Result(FilterId(0)));
    assert!(f.checked_gaussian_stages().is_err());
    assert!(Filter::gaussian_chain(&vec![blur([0.0, 0.0], crop); 3], crop).is_err());
    assert!(
        Filter::single(FilterEffect::blur(1.0))
            .checked_gaussian_stages()
            .is_err()
    );
}
