# Named parameter archives

`parameters::NamedParameters` stores f32 parameters under stable names and group
labels. Repeated `get` calls share parameter storage; `Parameter::set_values`
updates graphs already built from those parameters without changing identity.

An archive directory contains `manifest.json` and raw little-endian f32 data.
The manifest has version `1`, dtype `f32-le`, a single-component `data_file`
filename, application-owned `config`, and an ordered `tensors` array. Each tensor
has `name`, `shape`, `offset` (in f32 elements), `count`, and `group`. Tensor ranges
must be contiguous and cover the data file exactly. Names are unique, shape/count
arithmetic is checked, and parameter values must be finite.

Loading validates the archive structure. Applications call `get(name, shape)` and
`validate_names` for their complete model schema. Group labels are owned by the
application (for example, policy/value optimizer groups). Checkpoints contain
parameter values and model metadata; they do not implicitly restore optimizer
state or random-number-generator state.

Saving writes temporary data and manifest files before installing each file.
Use a fresh directory for a new checkpoint generation: the two file renames are
not a transaction across an existing archive.
