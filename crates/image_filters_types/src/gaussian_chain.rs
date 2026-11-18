//! Serial chains of one or two independent-axis Gaussian stages. Input edges and each output crop are retained.
use super::*;

impl<I> Filter<I> {
    /// Build one or two ordered stages; the final filter crop is separate from primitive crops.
    pub fn gaussian_chain(
        stages: &[GaussianBlurXY],
        final_region: Rect,
    ) -> Result<Self, FilterError> {
        if !(1..=2).contains(&stages.len()) {
            return Err(FilterError::Unsupported(
                "Gaussian chain longer than two stages",
            ));
        }
        GaussianBlurXY::new([0.0, 0.0], final_region, false)?;
        let mut graph = Self::new(final_region);
        let mut expansion = Rect::ZERO;
        for (index, blur) in stages.iter().enumerate() {
            GaussianBlurXY::new(blur.std_deviation, blur.region, blur.linear_rgb)?;
            let effect = FilterEffect::GaussianBlurXY(blur.clone());
            let halo = effect.expansion_rect();
            // A downstream stage depends on the upstream halo too (Minkowski sum).
            expansion = Rect::new(
                expansion.x0 + halo.x0,
                expansion.y0 + halo.y0,
                expansion.x1 + halo.x1,
                expansion.y1 + halo.y1,
            );
            let input = if index == 0 {
                FilterInput::Source(FilterSource::SourceGraphic)
            } else {
                FilterInput::Result(FilterId((index - 1) as u16))
            };
            let color_space = if blur.linear_rgb {
                FilterColorSpace::LinearRgb
            } else {
                FilterColorSpace::Srgb
            };
            graph.add_with_subregion(
                effect,
                FilterInputs::single(input),
                color_space,
                Some(blur.region),
            );
        }
        graph.expansion_rect = expansion;
        graph.final_region = Some(final_region);
        Ok(graph)
    }

    /// Revalidate every edge and stage, including descriptors mutated or deserialized by callers.
    pub fn checked_gaussian_stages(&self) -> Result<Vec<&GaussianBlurXY>, FilterError> {
        let count = self.nodes().len();
        if !(1..=2).contains(&count) || self.output() != FilterId((count - 1) as u16) {
            return Err(FilterError::Unsupported("filter graph/output"));
        }
        if let Some(region) = self.final_region {
            GaussianBlurXY::new([0.0, 0.0], region, false)?;
        }
        let mut stages = Vec::with_capacity(count);
        for (index, node) in self.nodes().iter().enumerate() {
            let valid_input = match (index, &node.inputs.primary) {
                (0, None) => count == 1, // Historical supported one-node shorthand.
                (0, Some(FilterInput::Source(FilterSource::SourceGraphic))) => true,
                (1, Some(FilterInput::Result(FilterId(0)))) => true,
                _ => false,
            };
            if !valid_input || node.inputs.secondary.is_some() {
                return Err(FilterError::Unsupported("filter input/result reference"));
            }
            let FilterEffect::GaussianBlurXY(blur) = &node.effect else {
                return Err(FilterError::Unsupported(
                    "Gaussian chain stage is not an independent-axis Gaussian",
                ));
            };
            GaussianBlurXY::new(blur.std_deviation, blur.region, blur.linear_rgb)?;
            stages.push(blur);
        }
        Ok(stages)
    }
}

#[cfg(test)]
mod tests;
