//! Shared parameter storage with lazy synchronization from a device owner.
use anyhow::{Context, Result, ensure};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

type Reader<D> = Arc<dyn Fn() -> Result<Vec<D>> + Send + Sync>;

struct State<D> {
    values: Vec<D>,
    revision: u64,
    reader: Option<Reader<D>>,
}

/// Shared, versioned parameter data. Host reads synchronize pending device
/// writes; host mutations invalidate cached device versions when the guard drops.
pub struct ParameterStorage<D> {
    state: Mutex<State<D>>,
}

impl<D> fmt::Debug for ParameterStorage<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParameterStorage").finish_non_exhaustive()
    }
}

impl<D> ParameterStorage<D> {
    pub fn new(values: Vec<D>) -> Self {
        Self {
            state: Mutex::new(State {
                values,
                revision: 0,
                reader: None,
            }),
        }
    }

    /// Lock current host values, synchronizing a pending device write once.
    /// A failed read leaves the device reader installed so callers can retry.
    pub fn lock(&self) -> Result<ParameterGuard<'_, D>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("parameter storage lock is poisoned"))?;
        if let Some(reader) = &state.reader {
            let values = reader().context("synchronize device parameter")?;
            ensure!(
                values.len() == state.values.len(),
                "device parameter length changed"
            );
            state.values = values;
            state.reader = None;
        }
        Ok(ParameterGuard {
            state,
            modified: false,
        })
    }

    /// Inspect the current host/device revision without downloading device data.
    pub fn revision(&self) -> Result<u64> {
        Ok(self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("parameter storage lock is poisoned"))?
            .revision)
    }

    /// Publish a completed device update without reading it back. The reader owns
    /// the device allocation, so the latest values survive the session's lifetime.
    #[cfg(any(feature = "cuda", test))]
    pub fn publish_device(&self, expected: u64, reader: Reader<D>) -> Result<u64> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("parameter storage lock is poisoned"))?;
        ensure!(
            state.revision == expected,
            "parameter changed during device update"
        );
        state.revision = state.revision.wrapping_add(1);
        state.reader = Some(reader);
        Ok(state.revision)
    }
}

/// Guard exposing current host values while tracking mutable access.
pub struct ParameterGuard<'a, D> {
    state: MutexGuard<'a, State<D>>,
    modified: bool,
}

impl<D> ParameterGuard<'_, D> {
    pub fn revision(&self) -> u64 {
        self.state.revision
    }
}

impl<D> Deref for ParameterGuard<'_, D> {
    type Target = Vec<D>;
    fn deref(&self) -> &Self::Target {
        &self.state.values
    }
}
impl<D> DerefMut for ParameterGuard<'_, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.modified = true;
        &mut self.state.values
    }
}
impl<D> Drop for ParameterGuard<'_, D> {
    fn drop(&mut self) {
        if self.modified {
            self.state.revision = self.state.revision.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn lazy_reads_share_one_download_and_host_writes_change_revision() {
        let storage = Arc::new(ParameterStorage::new(vec![1.0f32, 2.0]));
        let reads = Arc::new(AtomicUsize::new(0));
        let counter = reads.clone();
        storage
            .publish_device(
                0,
                Arc::new(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok(vec![3.0, 4.0])
                }),
            )
            .unwrap();
        assert_eq!(storage.revision().unwrap(), 1);
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert_eq!(&**storage.clone().lock().unwrap(), &[3.0, 4.0]);
        assert_eq!(&**storage.lock().unwrap(), &[3.0, 4.0]);
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert_eq!(storage.revision().unwrap(), 1);
        storage.lock().unwrap()[0] = 5.0;
        assert_eq!(storage.revision().unwrap(), 2);
        assert!(
            storage
                .publish_device(1, Arc::new(|| Ok(vec![0.0; 2])))
                .is_err()
        );
        assert_eq!(storage.lock().unwrap()[0], 5.0);
    }

    #[test]
    fn failed_download_remains_retryable() {
        let storage = ParameterStorage::new(vec![0]);
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        storage
            .publish_device(
                0,
                Arc::new(move || {
                    ensure!(
                        counter.fetch_add(1, Ordering::Relaxed) > 0,
                        "transient failure"
                    );
                    Ok(vec![7])
                }),
            )
            .unwrap();
        assert!(storage.lock().is_err());
        assert_eq!(storage.lock().unwrap()[0], 7);
        assert_eq!(storage.revision().unwrap(), 1);
    }

    #[test]
    fn latest_device_write_replaces_reader_and_rejects_bad_lengths() {
        let storage = ParameterStorage::new(vec![0]);
        storage
            .publish_device(0, Arc::new(|| anyhow::bail!("obsolete reader")))
            .unwrap();
        storage.publish_device(1, Arc::new(|| Ok(vec![9]))).unwrap();
        assert_eq!(storage.lock().unwrap()[0], 9);
        storage.publish_device(2, Arc::new(|| Ok(vec![]))).unwrap();
        assert!(storage.lock().is_err());
    }
}
