//! Sparse owner versions; cached functions contain no parameter/gradient bindings.
use super::super::*;
use super::{AdamKind, initialize_ir, paired_ir, payload_budget};
use crate::ptx::mixed::{MixedOptions, RetiredShadowBindings, ShadowBinding, shadow_declarations};
use cudarc::driver::{CudaFunction, DevicePtr, DeviceRepr};
use half::bf16;
use std::sync::Weak;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Bf16MasterShadowStatus {
    pub enabled: bool,
    pub declared_casts: usize,
    pub bound_cast_aliases: usize,
    pub bound_cast_alias_bytes: usize,
    pub removed_cast_kernels: usize,
    pub last_parameter_copies: usize,
    pub selected_parameters: usize,
    pub selected_elements: usize,
    pub rejected_declarations: Vec<String>,
    pub paired_f32_parameters: usize,
    pub paired_bf16_parameters: usize,
    pub paired_vector_parameters: usize,
    pub paired_vector_fallback_parameters: usize,
    pub paired_vector_fallback_reasons: BTreeMap<String, usize>,
    pub optional_vector_rejections: Vec<String>,
    pub logical_two_version_shadow_bytes: usize,
    pub logical_replaced_cast_version_bytes: usize,
    pub logical_net_additional_shadow_bytes: usize,
    pub submitted_parameters: usize,
    pub submitted_elements: usize,
    pub committed_parameters: usize,
    pub committed_elements: usize,
    pub initialization_launches: usize,
    pub host_mutation_initialization_launches: usize,
    pub current_shadow_bytes: usize,
    pub staged_shadow_bytes: usize,
    pub retained_shadow_bytes: usize,
    pub retained_f32_master_bytes: usize,
    pub current_shadow_versions: usize,
    pub staged_shadow_versions: usize,
    pub retained_shadow_versions: usize,
    pub maximum_versions_per_parameter: usize,
    pub resources: Vec<serde_json::Value>,
}
struct Code {
    initialize: CudaFunction,
    f32_adam: CudaFunction,
    bf16_adam: CudaFunction,
    vector_adam: Option<CudaFunction>,
    vector_rejection: Option<String>,
    resources: Vec<serde_json::Value>,
}
pub(in crate::ptx::resident) struct Shadow {
    pub current: Arc<CudaSlice<bf16>>,
    pub staged: Option<CudaSlice<bf16>>,
    pub master: Weak<CudaSlice<f32>>,
    pub revision: u64,
}
struct RetiredPair {
    parameter_id: usize,
    master: Arc<CudaSlice<f32>>,
    shadow: Arc<CudaSlice<bf16>>,
}
#[derive(Default)]
pub(in crate::ptx::resident) struct State {
    pub enabled: bool,
    pub options: MixedOptions,
    pub parameters: HashMap<usize, Shadow>,
    declarations: Vec<crate::ptx::mixed::ShadowDeclaration>,
    code: HashMap<usize, std::result::Result<Code, String>>,
    retired: Vec<RetiredPair>,
    pub status: Bf16MasterShadowStatus,
}
fn function_resources(
    function: &CudaFunction,
    name: &str,
    block: u32,
) -> Result<serde_json::Value> {
    let local = function.local_size_bytes()?;
    let shared = function.shared_size_bytes()?;
    let registers = function.num_regs()?;
    let max_threads = function.max_threads_per_block()?;
    let capacity = function.occupancy_max_active_blocks_per_multiprocessor(block, 0, None)?;
    ensure!(
        local == 0
            && registers > 0
            && registers <= 128
            && max_threads >= i32::try_from(block)?
            && capacity > 0,
        "paired shadow function resources unsupported: {name}"
    );
    if name == "initialize" {
        ensure!(
            shared == 0,
            "shadow initialization unexpectedly uses shared memory"
        );
    } else if block == 256 {
        ensure!(
            shared == 32 && capacity >= 2,
            "paired vector resource bound failed"
        );
    } else {
        ensure!(shared == 128, "paired scalar health shared layout changed");
    }
    Ok(
        serde_json::json!({"kind":name,"registers":registers,"local_bytes":local,
        "shared_bytes":shared,"max_threads":max_threads,"capacity_at_block":capacity,"block":block}),
    )
}
fn compile(device: &Arc<CudaContext>, len: usize) -> Result<Code> {
    let kernels = kernels::Kernels::compile(
        device,
        vec![
            initialize_ir(len)?,
            paired_ir(len, AdamKind::F32)?,
            paired_ir(len, AdamKind::Bf16First)?,
        ],
    )?;
    let functions = kernels
        .names
        .iter()
        .map(|n| kernels.module.load_function(n).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        functions.len() == 3,
        "paired shadow mandatory function count changed"
    );
    let mut resources = vec![
        function_resources(&functions[0], "initialize", 1024)?,
        function_resources(&functions[1], "F32", 1024)?,
        function_resources(&functions[2], "BF16", 1024)?,
    ];
    // Optional vector compilation/resource failure cannot invalidate the two
    // mandatory scalar paths needed after removal of an ordinary cast.
    let vector_result = (|| -> Result<CudaFunction> {
        let kernels = kernels::Kernels::compile(device, vec![paired_ir(len, AdamKind::Vector4)?])?;
        let function = kernels.module.load_function(&kernels.names[0])?;
        resources.push(function_resources(&function, "Vector4", 256)?);
        Ok(function)
    })();
    let (vector_adam, vector_rejection) = match vector_result {
        Ok(function) => (Some(function), None),
        Err(error) => {
            let reason = format!("{error:#}");
            resources
                .push(serde_json::json!({"kind":"Vector4","available":false,"rejection":reason}));
            (None, Some(reason))
        }
    };
    Ok(Code {
        initialize: functions[0].clone(),
        f32_adam: functions[1].clone(),
        bf16_adam: functions[2].clone(),
        vector_adam,
        vector_rejection,
        resources,
    })
}
pub(in crate::ptx::resident) fn operand_range<T: DeviceRepr, S: DevicePtr<T>>(
    operand: &S,
    stream: &Arc<CudaStream>,
) -> Result<(u64, u64)> {
    ensure!(
        Arc::ptr_eq(operand.stream().context(), stream.context()),
        "shadow operand context mismatch"
    );
    let start = operand.device_ptr(stream).0;
    let bytes = operand
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .context("shadow operand size overflow")?;
    let end = start
        .checked_add(u64::try_from(bytes)?)
        .context("shadow operand address overflow")?;
    ensure!(start > 0, "shadow operand has null address");
    Ok((start, end))
}
impl State {
    fn initialize(
        &self,
        stream: &Arc<CudaStream>,
        master: &Arc<CudaSlice<f32>>,
    ) -> Result<CudaSlice<bf16>> {
        let len = master.len();
        let code = self
            .code
            .get(&len)
            .and_then(|c| c.as_ref().ok())
            .context("mandatory shadow initializer missing")?;
        let mut shadow = stream.alloc_zeros::<bf16>(len)?;
        let mut launch = stream.launch_builder(&code.initialize);
        launch.arg(master.as_ref()).arg(&mut shadow);
        // SAFETY: full typed input/output allocations cover len; the canonical
        // initializer predicates every scalar load/store. Both use this stream.
        unsafe { launch.launch(launch::standalone_adam(len, false)?) }?;
        Ok(shadow)
    }
    #[cfg(test)]
    pub fn force_optional_vector_unavailable_for_test(&mut self) {
        for code in self.code.values_mut().filter_map(|code| code.as_mut().ok()) {
            code.vector_adam = None;
            code.vector_rejection = Some("test-forced optional paired vector unavailable".into());
        }
    }
    pub fn reset_step_status(&mut self) {
        self.status.paired_f32_parameters = 0;
        self.status.paired_bf16_parameters = 0;
        self.status.paired_vector_parameters = 0;
        self.status.paired_vector_fallback_parameters = 0;
        self.status.paired_vector_fallback_reasons.clear();
        self.status.submitted_parameters = 0;
        self.status.submitted_elements = 0;
        self.status.committed_parameters = 0;
        self.status.committed_elements = 0;
    }
    pub fn refresh_accounting(&mut self) {
        self.status.current_shadow_bytes =
            self.parameters.values().map(|p| p.current.len() * 2).sum();
        self.status.staged_shadow_bytes = self
            .parameters
            .values()
            .filter_map(|p| p.staged.as_ref())
            .map(|s| s.len() * 2)
            .sum();
        self.status.retained_shadow_bytes = self.retired.iter().map(|p| p.shadow.len() * 2).sum();
        self.status.retained_f32_master_bytes =
            self.retired.iter().map(|p| p.master.len() * 4).sum();
        self.status.current_shadow_versions = self.parameters.len();
        self.status.staged_shadow_versions = self
            .parameters
            .values()
            .filter(|p| p.staged.is_some())
            .count();
        self.status.retained_shadow_versions = self.retired.len();
        let maximum = self
            .parameters
            .iter()
            .map(|(id, p)| {
                1 + usize::from(p.staged.is_some())
                    + self
                        .retired
                        .iter()
                        .filter(|r| r.parameter_id == *id)
                        .count()
            })
            .max()
            .unwrap_or(0);
        self.status.maximum_versions_per_parameter =
            self.status.maximum_versions_per_parameter.max(maximum);
    }
    pub fn prepare_staging(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let result = (|| -> Result<()> {
            for shadow in self.parameters.values_mut() {
                if shadow.staged.is_none() {
                    shadow.staged = Some(stream.alloc_zeros::<bf16>(shadow.current.len())?);
                }
            }
            Ok(())
        })();
        self.refresh_accounting();
        result
    }
    pub fn paired_function(
        &self,
        len: usize,
        requested: AdamKind,
    ) -> Result<(CudaFunction, AdamKind)> {
        let code = self
            .code
            .get(&len)
            .and_then(|c| c.as_ref().ok())
            .context("mandatory paired shadow code missing")?;
        Ok(match requested {
            AdamKind::F32 => (code.f32_adam.clone(), AdamKind::F32),
            AdamKind::Bf16First => (code.bf16_adam.clone(), AdamKind::Bf16First),
            AdamKind::Vector4 => match &code.vector_adam {
                Some(function) => (function.clone(), AdamKind::Vector4),
                None => (code.bf16_adam.clone(), AdamKind::Bf16First),
            },
        })
    }
    pub fn validate_staged(
        &self,
        id: usize,
        stream: &Arc<CudaStream>,
        len: usize,
        operands: &[(u64, u64)],
    ) -> Result<()> {
        let shadow = self
            .parameters
            .get(&id)
            .context("selected shadow missing")?;
        let staged = shadow
            .staged
            .as_ref()
            .context("paired shadow staging missing")?;
        ensure!(
            staged.len() == len && shadow.current.len() == len,
            "paired shadow extent mismatch"
        );
        let output = operand_range(staged, stream)?;
        let current = operand_range(shadow.current.as_ref(), stream)?;
        ensure!(
            output.0.is_multiple_of(2)
                && (output.1 <= current.0 || current.1 <= output.0)
                && operands
                    .iter()
                    .all(|&(start, end)| output.1 <= start || end <= output.0),
            "staged shadow aliases a current/read/write operand"
        );
        Ok(())
    }
    pub fn record_submission(&mut self, kind: AdamKind, len: usize) {
        self.status.submitted_parameters += 1;
        self.status.submitted_elements += len;
        match kind {
            AdamKind::F32 => self.status.paired_f32_parameters += 1,
            AdamKind::Bf16First => self.status.paired_bf16_parameters += 1,
            AdamKind::Vector4 => self.status.paired_vector_parameters += 1,
        }
    }
    pub fn record_vector_fallback(&mut self, len: usize) {
        self.status.paired_vector_fallback_parameters += 1;
        let reason = self
            .code
            .get(&len)
            .and_then(|c| c.as_ref().ok())
            .and_then(|c| c.vector_rejection.as_deref())
            .unwrap_or("paired vector unavailable")
            .to_owned();
        *self
            .status
            .paired_vector_fallback_reasons
            .entry(reason)
            .or_default() += 1;
    }
    pub fn validate_publication(&self, parameters: &[ResidentParameter]) -> Result<()> {
        for parameter in parameters {
            let next = parameter
                .staged_weight
                .as_ref()
                .context("resident staged master absent before publication")?;
            ensure!(
                next.len() == parameter.range.len(),
                "resident staged master extent changed"
            );
            if let Some(shadow) = self.parameters.get(&parameter.id) {
                ensure!(
                    shadow.revision == parameter.revision
                        && shadow
                            .master
                            .upgrade()
                            .is_some_and(|m| Arc::ptr_eq(&m, &parameter.weight)),
                    "paired publication current version mismatch"
                );
                let next_shadow = shadow
                    .staged
                    .as_ref()
                    .context("staged shadow absent before publication")?;
                ensure!(
                    next_shadow.len() == parameter.range.len()
                        && Arc::ptr_eq(next_shadow.context(), parameter.weight.context()),
                    "staged shadow publication extent/context changed"
                );
            }
        }
        Ok(())
    }
    pub fn publish_parameter(
        &mut self,
        parameter: &mut ResidentParameter,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        let mut shadow_record = self.parameters.get_mut(&parameter.id);
        if let Some(shadow) = shadow_record.as_deref_mut() {
            ensure!(
                shadow.revision == parameter.revision
                    && shadow
                        .master
                        .upgrade()
                        .is_some_and(|m| Arc::ptr_eq(&m, &parameter.weight)),
                "paired publication current version mismatch"
            );
        }
        let next_shadow = if let Some(shadow) = shadow_record.as_deref_mut() {
            Some(Arc::new(
                shadow
                    .staged
                    .take()
                    .context("staged shadow absent before publication")?,
            ))
        } else {
            None
        };
        let weight = Arc::new(
            parameter
                .staged_weight
                .take()
                .context("resident weight staging absent after update")?,
        );
        let latest = weight.clone();
        let shadow_lease = next_shadow.clone();
        let readback = stream.clone();
        let published = parameter.storage.publish_device(
            parameter.revision,
            Arc::new(move || {
                // The lazy F32 reader retains the immutable matching BF16 version.
                let _paired_version = &shadow_lease;
                let mut host = vec![0.; latest.len()];
                readback
                    .memcpy_dtoh(latest.as_ref(), &mut host)
                    .context("read resident parameter")?;
                Ok(host)
            }),
        );
        let revision = match published {
            Ok(revision) => revision,
            Err(error) => {
                parameter.staged_weight = Arc::try_unwrap(weight).ok();
                if let (Some(shadow), Some(next)) = (shadow_record, next_shadow) {
                    shadow.staged = Arc::try_unwrap(next).ok();
                }
                return Err(error);
            }
        };
        // All fallible work precedes these ownership changes. Master/shadow and
        // revision become one current owner version after the existing health gate.
        parameter.revision = revision;
        let old_master = std::mem::replace(&mut parameter.weight, weight);
        if let (Some(shadow), Some(next)) = (shadow_record, next_shadow) {
            let old_shadow = std::mem::replace(&mut shadow.current, next);
            shadow.master = Arc::downgrade(&parameter.weight);
            shadow.revision = revision;
            self.retired.push(RetiredPair {
                parameter_id: parameter.id,
                master: old_master,
                shadow: old_shadow,
            });
        } else {
            parameter.staged_weight = Arc::try_unwrap(old_master).ok();
        }
        Ok(())
    }
}
impl ResidentAdam {
    /// Declare graph-proved sparse matrix shadows before the first begin_step.
    /// Unsupported scalar code/resource declarations keep ordinary casts.
    pub fn declare_experimental_bf16_master_shadows(
        &mut self,
        graph: &TensorGraph<f32, WithGrad>,
        options: MixedOptions,
    ) -> Result<()> {
        ensure!(
            !self.active && self.timestep == 0 && !self.bf16_master_shadows.enabled,
            "declare BF16 master shadows once before the first begin_step"
        );
        ensure!(
            options.physical_order_casts && options.direct_master_bindings,
            "BF16 master shadows require physical_order_casts and direct_master_bindings"
        );
        let declarations = shadow_declarations(graph)?;
        let mut state = State {
            enabled: true,
            options,
            ..State::default()
        };
        state.status.enabled = true;
        state.status.declared_casts = declarations.len();
        for declaration in declarations {
            let parameter = self
                .parameters
                .iter()
                .find(|p| p.id == declaration.parameter_id)
                .context("shadow declaration refers to an unregistered parameter")?;
            ensure!(
                Arc::ptr_eq(&declaration.storage, &parameter.storage)
                    && declaration.elements == parameter.range.len()
                    && declaration.elements > 4096
                    && declaration
                        .source_shape
                        .iter()
                        .try_fold(1usize, |n, &d| n.checked_mul(d))
                        == Some(declaration.elements)
                    && Arc::ptr_eq(parameter.weight.context(), &self.device)
                    && parameter.storage.revision()? == parameter.revision,
                "shadow declaration storage/extent/context/revision mismatch"
            );
            let len = declaration.elements;
            state
                .code
                .entry(len)
                .or_insert_with(|| compile(&self.device, len).map_err(|e| format!("{e:#}")));
            if let Some(Err(reason)) = state.code.get(&len) {
                state.status.rejected_declarations.push(format!(
                    "parameter {} cast {}: {reason}",
                    parameter.id,
                    declaration.cast_node.index()
                ));
                continue;
            }
            if !state.parameters.contains_key(&parameter.id) {
                let current = Arc::new(state.initialize(&self.stream, &parameter.weight)?);
                let staged = Some(self.stream.alloc_zeros::<bf16>(len)?);
                state.parameters.insert(
                    parameter.id,
                    Shadow {
                        current,
                        staged,
                        master: Arc::downgrade(&parameter.weight),
                        revision: parameter.revision,
                    },
                );
                state.status.initialization_launches += 1;
                state.status.selected_parameters += 1;
                state.status.selected_elements += len;
            }
            state.declarations.push(declaration);
        }
        // No shadow binding can escape before initialization has completed.
        self.stream.synchronize()?;
        state.status.resources = state
            .code
            .iter()
            .filter_map(|(&len, code)| {
                code.as_ref()
                    .ok()
                    .map(|c| serde_json::json!({"elements":len,"functions":c.resources}))
            })
            .collect();
        state.status.optional_vector_rejections = state
            .code
            .iter()
            .filter_map(|(&len, c)| {
                c.as_ref()
                    .ok()
                    .and_then(|c| c.vector_rejection.as_ref())
                    .map(|reason| format!("len{len}: {reason}"))
            })
            .collect();
        let budget = payload_budget(
            state.status.selected_elements,
            state.status.selected_elements,
        )?;
        state.status.logical_two_version_shadow_bytes = budget.two_bf16_versions_bytes;
        state.status.logical_replaced_cast_version_bytes = budget.replaced_cast_payload_bytes;
        state.status.logical_net_additional_shadow_bytes = budget.net_additional_payload_bytes;
        state.refresh_accounting();
        self.bf16_master_shadows = state;
        Ok(())
    }
    pub fn bf16_master_shadow_status(&self) -> &Bf16MasterShadowStatus {
        &self.bf16_master_shadows.status
    }
    pub(in crate::ptx::resident) fn mixed_shadow_bindings(
        &self,
        graph: &TensorGraph<f32, WithGrad>,
    ) -> Result<HashMap<crate::graph::NodeIndex, ShadowBinding>> {
        let mut result = HashMap::new();
        for declaration in &self.bf16_master_shadows.declarations {
            let parameter = self
                .parameters
                .iter()
                .find(|p| p.id == declaration.parameter_id)
                .context("declared shadow parameter absent")?;
            let Some(TensorGraphNode::Parameter { id, data, shape }) =
                graph.graph.node_weight(declaration.parameter_node)
            else {
                anyhow::bail!("declared shadow graph parameter changed");
            };
            ensure!(
                *id == declaration.parameter_id
                    && Arc::ptr_eq(data, &declaration.storage)
                    && Arc::ptr_eq(data, &parameter.storage)
                    && shape == &declaration.source_shape,
                "declared shadow graph storage/shape identity changed"
            );
            let shadow = self
                .bf16_master_shadows
                .parameters
                .get(id)
                .context("declared shadow current version absent")?;
            ensure!(
                shadow.revision == parameter.revision
                    && data.revision()? == parameter.revision
                    && shadow.current.len() == declaration.elements
                    && shadow
                        .master
                        .upgrade()
                        .is_some_and(|m| Arc::ptr_eq(&m, &parameter.weight))
                    && Arc::ptr_eq(shadow.current.context(), &self.device),
                "declared shadow/master current version mismatch"
            );
            result.insert(
                declaration.cast_node,
                ShadowBinding {
                    parameter_node: declaration.parameter_node,
                    parameter_id: *id,
                    storage: data.clone(),
                    revision: parameter.revision,
                    master: Arc::downgrade(&parameter.weight),
                    shadow: shadow.current.clone(),
                },
            );
        }
        Ok(result)
    }
    pub(in crate::ptx::resident) fn reclaim_shadow_pairs(&mut self) -> Result<()> {
        let mut retained = Vec::new();
        for pair in std::mem::take(&mut self.bf16_master_shadows.retired) {
            let parameter = self
                .parameters
                .iter_mut()
                .find(|p| p.id == pair.parameter_id)
                .context("retired paired parameter absent")?;
            let shadow = self
                .bf16_master_shadows
                .parameters
                .get_mut(&pair.parameter_id)
                .context("retired shadow selection absent")?;
            ensure!(
                !Arc::ptr_eq(&pair.master, &parameter.weight)
                    && !Arc::ptr_eq(&pair.shadow, &shadow.current),
                "current pair cannot become staging"
            );
            if Arc::strong_count(&pair.master) != 1 || Arc::strong_count(&pair.shadow) != 1 {
                retained.push(pair);
                continue;
            }
            if parameter.staged_weight.is_none() && shadow.staged.is_none() {
                // Both exclusive allocations may be recycled together. If a
                // racing weak upgrade defeats either unwrap, rewrap/retain both.
                match (Arc::try_unwrap(pair.master), Arc::try_unwrap(pair.shadow)) {
                    (Ok(master), Ok(bf16)) => {
                        parameter.staged_weight = Some(master);
                        shadow.staged = Some(bf16);
                    }
                    (master, bf16) => retained.push(RetiredPair {
                        parameter_id: pair.parameter_id,
                        master: master.map(Arc::new).unwrap_or_else(|a| a),
                        shadow: bf16.map(Arc::new).unwrap_or_else(|a| a),
                    }),
                }
            }
            // Fully exclusive retired pairs not needed for staging are freed.
        }
        self.bf16_master_shadows.retired = retained;
        self.bf16_master_shadows.refresh_accounting();
        Ok(())
    }
    pub(in crate::ptx::resident) fn reclaim_mixed_shadow_bindings(
        &mut self,
        graph: &TensorGraph<f32, WithGrad>,
        retired: RetiredShadowBindings,
    ) -> Result<()> {
        let mut ordinary = HashMap::new();
        let mut paired_masters = Vec::new();
        for (node, master) in retired.masters {
            let Some(TensorGraphNode::Parameter { id, data, .. }) = graph.graph.node_weight(node)
            else {
                anyhow::bail!("retired paired master is not a parameter");
            };
            if self.bf16_master_shadows.parameters.contains_key(id) {
                ensure!(
                    self.parameters
                        .iter()
                        .any(|p| p.id == *id && Arc::ptr_eq(data, &p.storage)),
                    "retired paired master storage mismatch"
                );
                ensure!(
                    self.bf16_master_shadows
                        .retired
                        .iter()
                        .any(|p| p.parameter_id == *id && Arc::ptr_eq(&p.master, &master)),
                    "retired mixed master does not match an owner-retired pair"
                );
                paired_masters.push(master);
            } else {
                ordinary.insert(node, master);
            }
        }
        for (id, shadow) in &retired.shadows {
            ensure!(
                self.bf16_master_shadows
                    .retired
                    .iter()
                    .any(|p| p.parameter_id == *id && Arc::ptr_eq(&p.shadow, shadow)),
                "retired mixed shadow does not match an owner-retired pair"
            );
        }
        // Returned active leases duplicate the owner's retired pair. Drop every
        // duplicate before attempting the joint exclusive-ownership proof.
        drop(paired_masters);
        drop(retired.shadows);
        self.reclaim_mixed_masters(graph, ordinary)?;
        self.reclaim_shadow_pairs()
    }
    pub(in crate::ptx::resident) fn import_host_shadow_pair(
        &mut self,
        index: usize,
        master: Arc<CudaSlice<f32>>,
        revision: u64,
    ) -> Result<()> {
        let id = self.parameters[index].id;
        let initialized = Arc::new(self.bf16_master_shadows.initialize(&self.stream, &master)?);
        self.stream.synchronize()?;
        let parameter = &mut self.parameters[index];
        let shadow = self
            .bf16_master_shadows
            .parameters
            .get_mut(&id)
            .context("host-import shadow absent")?;
        let old_master = std::mem::replace(&mut parameter.weight, master);
        let old_shadow = std::mem::replace(&mut shadow.current, initialized);
        parameter.revision = revision;
        shadow.revision = revision;
        shadow.master = Arc::downgrade(&parameter.weight);
        self.bf16_master_shadows.retired.push(RetiredPair {
            parameter_id: id,
            master: old_master,
            shadow: old_shadow,
        });
        self.bf16_master_shadows
            .status
            .host_mutation_initialization_launches += 1;
        self.bf16_master_shadows.refresh_accounting();
        Ok(())
    }
}
