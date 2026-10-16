//! Preallocated matmul plans shared by precision experiments.
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    },
    nvrtc::Ptx,
};
use num_traits::{ToPrimitive, cast};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tnsr::{
    graph::{NodeIndex, TensorGraph, TensorGraphNode},
    ptx::{PtxExecutor, PtxPlanAction, types::CudaDType},
    tensor::{Parameter, TensorExpr},
    tile::MatMulPrecision,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub label: String,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
    pub shared_rhs: bool,
    pub bias: bool,
}
impl Shape {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [self.batch, self.m, self.k, self.n].iter().all(|&x| x > 0),
            "zero GEMM extent"
        );
        for dims in [
            [self.batch, self.m, self.k],
            [self.rhs_batch(), self.k, self.n],
            [self.batch, self.m, self.n],
        ] {
            let elements = dims
                .into_iter()
                .try_fold(1usize, usize::checked_mul)
                .context("GEMM size overflow")?;
            ensure!(
                u32::try_from(elements).is_ok(),
                "probe conversion extent exceeds u32"
            );
        }
        Ok(())
    }
    pub fn rhs_batch(&self) -> usize {
        if self.shared_rhs { 1 } else { self.batch }
    }
    pub fn master_values(&self) -> Vec<Vec<f32>> {
        let values = |count, period| {
            (0..count)
                .map(|i| (i % period) as f32 / period as f32 - 0.5)
                .collect()
        };
        let mut result = vec![
            values(self.batch * self.m * self.k, 29),
            values(self.rhs_batch() * self.k * self.n, 31),
        ];
        if self.bias {
            result.push(vec![0.1; self.n]);
        }
        result
    }
}
#[derive(Clone)]
pub struct Launch {
    pub function: CudaFunction,
    pub config: LaunchConfig,
    pub pointers: Vec<u64>,
}
pub fn submit(stream: &Arc<CudaStream>, launches: &[Launch]) -> Result<()> {
    for launch in launches {
        let mut builder = stream.launch_builder(&launch.function);
        for pointer in &launch.pointers {
            builder.arg(pointer);
        }
        // SAFETY: pointers refer to retained, correctly typed device allocations;
        // the compiled ABI and launch configuration come from this exact plan.
        // Inputs, outputs and all graph captures use the same serialized stream.
        unsafe { builder.launch(launch.config) }.context("probe kernel launch")?;
    }
    Ok(())
}
pub struct ResidentGemm<D: CudaDType> {
    pub launches: Vec<Launch>,
    pub ports: Vec<(u64, usize)>,
    pub output_pointer: u64,
    pub ptx: String,
    pub plan: String,
    pub schedules: Vec<String>,
    storage: HashMap<NodeIndex, CudaSlice<D::HostType>>,
    output: NodeIndex,
}
impl<D: CudaDType> ResidentGemm<D> {
    pub fn compile(
        device: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        shape: &Shape,
        precision: MatMulPrecision,
    ) -> Result<Self> {
        shape.validate()?;
        let cast_values = |values: &[f32]| {
            values
                .iter()
                .map(|&value| cast(value).context("probe value not representable"))
                .collect::<Result<Vec<D::HostType>>>()
        };
        let values = shape.master_values();
        let lhs_shape = if shape.transpose_lhs {
            vec![shape.batch, shape.k, shape.m]
        } else {
            vec![shape.batch, shape.m, shape.k]
        };
        let rhs_shape = if shape.transpose_rhs {
            vec![shape.rhs_batch(), shape.n, shape.k]
        } else {
            vec![shape.rhs_batch(), shape.k, shape.n]
        };
        let lhs = Parameter::new(cast_values(&values[0])?, lhs_shape);
        let rhs = Parameter::new(cast_values(&values[1])?, rhs_shape);
        let mut parameter_ids = vec![lhs.id(), rhs.id()];
        let lhs = if shape.transpose_lhs {
            TensorExpr::from(lhs).transpose()
        } else {
            TensorExpr::from(lhs)
        };
        let rhs = if shape.transpose_rhs {
            TensorExpr::from(rhs).transpose()
        } else {
            TensorExpr::from(rhs)
        };
        let mut expression = lhs.matmul(rhs);
        if shape.bias {
            let bias = Parameter::new(cast_values(&values[2])?, vec![shape.n]);
            parameter_ids.push(bias.id());
            expression = expression + TensorExpr::from(bias);
        }
        let graph: TensorGraph<D::HostType> = expression.into();
        let mut executor = PtxExecutor::<D>::try_new_for_dtype()?;
        executor.set_matmul_precision(precision);
        executor.compile_owned(graph.clone())?;
        let description = executor.describe_plan(&graph)?;
        let source = executor.module_source().context("missing probe PTX")?;
        let plan = executor.execution_plan().context("missing probe plan")?;
        let module = device.load_module(Ptx::from_src(&source))?;
        let mut storage = HashMap::new();
        let mut aliases = HashMap::new();
        let names: Vec<_> = description
            .lines()
            .skip(1)
            .map(|line| {
                line.split("action=kernel ")
                    .nth(1)
                    .and_then(|name| name.split_whitespace().next())
            })
            .collect();
        ensure!(
            names.len() == plan.steps().len(),
            "probe plan length mismatch"
        );
        let mut launches = Vec::new();
        for (step, name) in plan.steps().iter().zip(names) {
            if step.action == PtxPlanAction::Upload {
                let data = match &graph[step.node] {
                    TensorGraphNode::Parameter { data, .. } => data
                        .lock()
                        .map_err(|_| anyhow::anyhow!("probe parameter poisoned"))?
                        .clone(),
                    TensorGraphNode::Constant { data, .. } => data.as_ref().clone(),
                    _ => anyhow::bail!("unexpected probe upload"),
                };
                storage.insert(step.node, stream.memcpy_stod(&data)?);
                continue;
            }
            if step.action == PtxPlanAction::VirtualView {
                ensure!(
                    step.virtual_outputs.len() == 1,
                    "unexpected multiple probe views"
                );
                aliases.insert(step.node, step.virtual_outputs[0].source);
                continue;
            }
            let name = name.context("unsupported physical action in GEMM probe")?;
            for (&output, output_shape) in step.outputs.iter().zip(&step.output_shapes) {
                storage.insert(
                    output,
                    stream.alloc_zeros::<D::HostType>(output_shape.iter().product())?,
                );
            }
            let pointer = |node: &NodeIndex| -> Result<u64> {
                let node = aliases.get(node).unwrap_or(node);
                Ok(storage
                    .get(node)
                    .context("probe buffer unavailable")?
                    .device_ptr(stream)
                    .0)
            };
            let inputs = step
                .inputs
                .iter()
                .map(pointer)
                .collect::<Result<Vec<_>>>()?;
            let outputs = step
                .outputs
                .iter()
                .map(pointer)
                .collect::<Result<Vec<_>>>()?;
            let function = module.load_function(name)?;
            if step.action == PtxPlanAction::Kernel
                && matches!(graph[step.node], TensorGraphNode::MatMul { .. })
            {
                let schedule = plan
                    .matmul_schedules()
                    .get(&step.node)
                    .context("missing standalone GEMM schedule")?;
                let width = std::mem::size_of::<D::HostType>();
                for batch in 0..shape.batch {
                    launches.push(Launch {
                        function: function.clone(),
                        config: LaunchConfig {
                            grid_dim: (
                                u32::try_from(shape.n.div_ceil(schedule.block_tile.n))?,
                                u32::try_from(shape.m.div_ceil(schedule.block_tile.m))?,
                                1,
                            ),
                            block_dim: schedule.block_threads,
                            shared_mem_bytes: 0,
                        },
                        pointers: vec![
                            inputs[0] + u64::try_from(batch * shape.m * shape.k * width)?,
                            inputs[1]
                                + u64::try_from(if shape.shared_rhs {
                                    0
                                } else {
                                    batch * shape.k * shape.n * width
                                })?,
                            outputs[0] + u64::try_from(batch * shape.m * shape.n * width)?,
                        ],
                    });
                }
            } else {
                let config = if let PtxPlanAction::MatMulRegion(id) = step.action {
                    let schedule = plan.matmul_regions()[id].schedule;
                    LaunchConfig {
                        grid_dim: (
                            u32::try_from(shape.n.div_ceil(schedule.block_tile.n))?,
                            u32::try_from(shape.m.div_ceil(schedule.block_tile.m))?,
                            u32::try_from(shape.batch)?,
                        ),
                        block_dim: schedule.block_threads,
                        shared_mem_bytes: 0,
                    }
                } else {
                    LaunchConfig::for_num_elems(u32::try_from(
                        step.shape.iter().product::<usize>(),
                    )?)
                };
                launches.push(Launch {
                    function,
                    config,
                    pointers: inputs.into_iter().chain(outputs).collect(),
                });
            }
        }
        let ports=parameter_ids.iter().map(|id| {
            let node=graph.graph.node_indices().find(|&node|matches!(&graph[node],TensorGraphNode::Parameter{id:other,..} if other==id)).context("missing probe parameter")?;
            let buffer=storage.get(&node).context("missing probe parameter storage")?;
            Ok((buffer.device_ptr(stream).0,buffer.len()))
        }).collect::<Result<Vec<_>>>()?;
        let output = plan.graph_output().context("missing probe output")?;
        let output_pointer = storage
            .get(&output)
            .context("missing probe output buffer")?
            .device_ptr(stream)
            .0;
        let schedules = plan
            .matmul_regions()
            .iter()
            .map(|region| format!("{:?}", region.schedule))
            .chain(
                plan.matmul_schedules()
                    .values()
                    .map(|schedule| format!("{schedule:?}")),
            )
            .collect();
        Ok(Self {
            launches,
            ports,
            output_pointer,
            ptx: source,
            plan: description,
            schedules,
            storage,
            output,
        })
    }
    pub fn replicate(&self, stream: &Arc<CudaStream>) -> Result<Self> {
        let mut storage = HashMap::new();
        let mut ranges = Vec::new();
        let width = std::mem::size_of::<D::HostType>();
        for (&node, buffer) in &self.storage {
            let clone = stream.clone_dtod(buffer)?;
            ranges.push((
                buffer.device_ptr(stream).0,
                u64::try_from(buffer.len() * width)?,
                clone.device_ptr(stream).0,
            ));
            storage.insert(node, clone);
        }
        let remap = |pointer: u64| -> Result<u64> {
            let &(base, _, destination) = ranges
                .iter()
                .find(|&&(base, bytes, _)| pointer >= base && pointer - base < bytes)
                .context("replicated GEMM pointer is not in an owned allocation")?;
            Ok(destination + (pointer - base))
        };
        let launches = self
            .launches
            .iter()
            .map(|launch| {
                Ok(Launch {
                    function: launch.function.clone(),
                    config: launch.config,
                    pointers: launch
                        .pointers
                        .iter()
                        .map(|&pointer| remap(pointer))
                        .collect::<Result<_>>()?,
                })
            })
            .collect::<Result<_>>()?;
        let ports = self
            .ports
            .iter()
            .map(|&(pointer, extent)| Ok((remap(pointer)?, extent)))
            .collect::<Result<_>>()?;
        Ok(Self {
            launches,
            ports,
            output_pointer: remap(self.output_pointer)?,
            ptx: self.ptx.clone(),
            plan: self.plan.clone(),
            schedules: self.schedules.clone(),
            storage,
            output: self.output,
        })
    }
    pub fn allocated_bytes(&self) -> usize {
        self.storage
            .values()
            .map(|buffer| buffer.len() * std::mem::size_of::<D::HostType>())
            .sum()
    }
    pub fn validate_inputs(&self, stream: &Arc<CudaStream>, shape: &Shape) -> Result<()> {
        for (&(pointer, _), expected) in self.ports.iter().zip(shape.master_values()) {
            let buffer = self
                .storage
                .values()
                .find(|buffer| buffer.device_ptr(stream).0 == pointer)
                .context("missing input port")?;
            let actual = stream.memcpy_dtov(buffer)?;
            for (actual, value) in actual.iter().zip(expected) {
                let expected: D::HostType =
                    cast(value).context("invalid input rounding reference")?;
                ensure!(
                    *actual == expected,
                    "device input conversion differs from host dtype rounding"
                );
            }
        }
        Ok(())
    }
    pub fn output(&self, stream: &Arc<CudaStream>) -> Result<Vec<f32>> {
        stream
            .memcpy_dtov(
                self.storage
                    .get(&self.output)
                    .context("missing probe output")?,
            )?
            .into_iter()
            .map(|value| value.to_f32().context("invalid probe output"))
            .collect()
    }
}
