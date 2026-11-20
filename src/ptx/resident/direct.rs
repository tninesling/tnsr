//! Recycle retired mixed-executor master leases only after refresh commits.
use super::*;

impl ResidentAdam {
    pub(super) fn reclaim_mixed_masters(
        &mut self,
        graph: &TensorGraph<f32, WithGrad>,
        retired: HashMap<crate::graph::NodeIndex, Arc<CudaSlice<f32>>>,
    ) -> Result<()> {
        let mut candidates = Vec::with_capacity(retired.len());
        let mut seen = std::collections::HashSet::new();
        // Validate the complete map before changing any staging owner. A stale
        // executor/reader may still hold a lease; try_unwrap below then fails
        // safely, retaining the ordinary fresh-staging allocation fallback.
        for (node, source) in retired {
            let Some(TensorGraphNode::Parameter { id, data, .. }) = graph.graph.node_weight(node)
            else {
                anyhow::bail!("retired mixed source is not a parameter");
            };
            let index = self
                .parameters
                .iter()
                .position(|p| p.id == *id)
                .context("retired mixed master is unregistered")?;
            let parameter = &self.parameters[index];
            ensure!(
                Arc::ptr_eq(data, &parameter.storage)
                    && Arc::ptr_eq(source.context(), &self.device)
                    && source.len() == parameter.range.len(),
                "retired mixed master context/extent/storage mismatch"
            );
            ensure!(
                seen.insert(index),
                "retired mixed master has duplicate parameter ownership"
            );
            ensure!(
                !Arc::ptr_eq(&source, &parameter.weight),
                "current mixed master cannot become staging"
            );
            // Both allocations are alive here, so a repeated physical address
            // would indicate an invalid alias rather than allocator reuse.
            use cudarc::driver::DevicePtr;
            ensure!(
                source.device_ptr(&self.stream).0 != parameter.weight.device_ptr(&self.stream).0,
                "retired mixed master aliases the current allocation"
            );
            candidates.push((index, source));
        }
        for (index, source) in candidates {
            if self.parameters[index].staged_weight.is_none() {
                self.parameters[index].staged_weight = Arc::try_unwrap(source).ok();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "direct_tests.rs"]
mod tests;
