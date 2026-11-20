//! Test-only snapshots of actual operation operands before prefix pruning.
use super::*;
use std::{cell::RefCell, collections::HashSet};

struct State {
    targets: HashSet<NodeIndex>,
    snapshots: HashMap<NodeIndex, Vec<Value>>,
}
thread_local! {
    static CURRENT: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Metadata only: Value clones contain raw bindings/maps, never buffer owners.
/// This does not extend any GPU allocation lifetime or change reuse decisions.
pub(in crate::ptx::mixed) fn record(node: NodeIndex, operands: &[Value]) {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if let Some(state) = current.as_mut()
            && state.targets.contains(&node)
        {
            state.snapshots.insert(node, operands.to_vec());
        }
    });
}

pub(in crate::ptx::mixed) struct Guard {
    active: bool,
}
impl Guard {
    pub(in crate::ptx::mixed) fn finish(mut self) -> HashMap<NodeIndex, Vec<Value>> {
        self.active = false;
        CURRENT.with(|current| {
            current
                .borrow_mut()
                .take()
                .expect("observer guard is active")
                .snapshots
        })
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        if self.active {
            CURRENT.with(|current| {
                current.borrow_mut().take();
            });
        }
    }
}

pub(in crate::ptx::mixed) fn install(nodes: impl IntoIterator<Item = NodeIndex>) -> Result<Guard> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        ensure!(current.is_none(), "binding observer cannot be nested");
        *current = Some(State {
            targets: nodes.into_iter().collect(),
            snapshots: HashMap::new(),
        });
        Ok(Guard { active: true })
    })
}
