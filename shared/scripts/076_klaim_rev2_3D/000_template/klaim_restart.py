"""Checkpoint / restart after a SLURM timeout for run_simulation.py (KLAIM 2025 rev. 2, 075 2D and 076 3D).

Identical file in 075_klaim_rev2_2D/000_template and 076_klaim_rev2_3D/000_template.

Checkpoint file  <script>_checkpoint.h5  (one file, written by rank 0, atomically: tmp file + os.replace)
    attrs["meta"]               JSON: t, dt, trestart, cumulative quantities, state dict, table sizes, ... (see run_simulation.py)
    p1/<name>/sub<i>/original_index  (N,)   int64   mesh-file index of the node of each owned dof
                                                     (dolfinx geometry.input_global_indices)
    p1/<name>/sub<i>/x               (N, 3) float64 node coordinates
    p1/<name>/sub<i>/values          (N, bs)        values of sub-space i of a P1 (mixed) function
    cells/original_index        (M,)   int64   cell index in the mesh file (dolfinx topology.original_cell_index)
    cells/midpoint              (M, 3) float64 cell midpoints (= coordinate of the single quadrature point, deg_quad = 1)
    quad/<name>                 (M,)           values of a quadrature function (one point per cell), rows = cells/
A copy of the meta data is written to <script>_checkpoint.json for reading by humans/scripts.

Assignment old -> new is independent of the partition (number of processes, PT-SCOTCH numbering; see
Meshing/pygalmesh/data/scripts/014-Yield-Surface-From-leS/RESTART_NACH_TIMEOUT.md): the rows are keyed by the node / cell
index of the *mesh file*, which dolfinx keeps for every process count; the stored coordinates are compared with the local
ones as a check (and are the fallback key if the indices do not match, e.g. a regenerated identical mesh).

Legacy restart (runs started before the checkpoint code existed): restore_from_field_output() reads the last complete
time step of run_simulation.xdmf/.h5 (u, s, alpha, eps_p written every postprocessing_interval steps) and matches nodes and
cells by their coordinates, as yield_restart.py in 014/017 does. Cumulative scalars then come from the graph table
(4 significant digits) - see RESTART_NACH_TIMEOUT.md.
"""

import glob
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
from mpi4py import MPI

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # many ranks read the same file (Lustre/GPFS)

FORMAT_VERSION = 1


class RestartError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RestartError("checkpointing needs h5py in the container (apptainer exec <sif> python3 -c 'import h5py')") from exc
    return h5py


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def _gather_rows(comm, arr):
    """Concatenate the rows of arr from all ranks on rank 0 (None on the other ranks)."""
    parts = comm.gather(np.ascontiguousarray(arr), root=0)
    if comm.Get_rank() != 0:
        return None
    return np.concatenate(parts, axis=0)


def _all_ok(comm, ok_local, msg_local=""):
    """Collective decision; returns (ok, first error message of any rank)."""
    msgs = comm.allgather("" if ok_local else (msg_local or "unknown error"))
    bad = [m for m in msgs if m]
    return (len(bad) == 0), (bad[0] if bad else "")


def _match(old_keys, new_keys):
    """Row index in old_keys for every row of new_keys (-1 = not found / ambiguous). Keys: int64 (n,) or (n, k)."""
    old_keys = np.ascontiguousarray(old_keys, dtype=np.int64)
    new_keys = np.ascontiguousarray(new_keys, dtype=np.int64)
    if old_keys.ndim == 1:
        old_keys = old_keys[:, None]
        new_keys = new_keys[:, None]
    n_old = old_keys.shape[0]
    if new_keys.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if n_old == 0:
        return np.full(new_keys.shape[0], -1, dtype=np.int64)
    _, inv = np.unique(np.concatenate([old_keys, new_keys], axis=0), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    n_ids = int(inv.max()) + 1
    old_of_id = np.full(n_ids, -1, dtype=np.int64)
    old_ids = inv[:n_old]
    old_of_id[old_ids] = np.arange(n_old, dtype=np.int64)
    dup = np.bincount(old_ids, minlength=n_ids) > 1
    res = old_of_id[inv[n_old:]]
    res[dup[inv[n_old:]]] = -1
    return res


def _coord_keys(xyz, quantum=None, shift=0.0):
    xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64)[:, :3]) + 0.0   # -0.0 -> 0.0
    if quantum is None:
        return xyz.view(np.int64)
    return np.round(xyz / float(quantum) + float(shift)).astype(np.int64)


def _match_coords(old_xyz, new_xyz, quantum=1.0e-9):
    """Coordinate matching: exact float64 bits, fallback quantized on two shifted grids (as in yield_restart.py)."""
    idx = _match(_coord_keys(old_xyz), _coord_keys(new_xyz))
    for shift in (0.0, 0.5):
        miss = np.nonzero(idx < 0)[0]
        if miss.size == 0:
            break
        idx[miss] = _match(_coord_keys(old_xyz, quantum, shift), _coord_keys(new_xyz[miss], quantum, shift))
    return idx


def _as_2d(dofmap_list, ncells):
    arr = np.asarray(getattr(dofmap_list, "array", dofmap_list))
    if arr.ndim == 1:
        arr = arr.reshape(ncells, -1)
    return arr.astype(np.int64, copy=False)


def truncate_file(path, size):
    """Cut a text file back to `size` bytes (the size it had when the checkpoint was written)."""
    if size is None or not os.path.isfile(path):
        return False
    if os.path.getsize(path) < size:
        raise RestartError(f"{os.path.basename(path)} is shorter ({os.path.getsize(path)} B) than at the checkpoint ({size} B)")
    with open(path, "r+b") as f:
        f.truncate(size)
    return True


def file_size(path):
    return os.path.getsize(path) if os.path.isfile(path) else None


def count_data_lines(path):
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def next_generation_path(path_xdmf):
    """run_simulation.xdmf -> run_simulation_r<N>.xdmf with the next free N."""
    base = path_xdmf[:-5]
    used = [0]
    for p in glob.glob(base + "_r*.xdmf"):
        m = re.search(r"_r(\d+)\.xdmf$", p)
        if m:
            used.append(int(m.group(1)))
    n = max(used) + 1
    return n, f"{base}_r{n}.xdmf"


# ----------------------------------------------------------------------------------------------------------------------
# mesh-level index structures
# ----------------------------------------------------------------------------------------------------------------------
class _MeshIndex:
    def __init__(self, domain):
        self.domain = domain
        tdim = domain.topology.dim
        cmap = domain.topology.index_map(tdim)
        self.ncells_own = cmap.size_local
        self.ncells_all = cmap.size_local + cmap.num_ghosts
        gimap = domain.geometry.index_map()
        self.nnodes_own = gimap.size_local
        self.nnodes_all = gimap.size_local + gimap.num_ghosts
        self.x = np.ascontiguousarray(np.asarray(domain.geometry.x)[: self.nnodes_all, :3], dtype=np.float64)
        self.node_orig = np.asarray(domain.geometry.input_global_indices, dtype=np.int64)[: self.nnodes_all]
        self.cell_orig = np.asarray(domain.topology.original_cell_index, dtype=np.int64)[: self.ncells_all]
        self.gdm = _as_2d(domain.geometry.dofmap, self.ncells_all)
        self.midpoints = np.ascontiguousarray(self.x[self.gdm].mean(axis=1))   # (ncells_all, 3)
        # vertex-coordinate key of each cell (sorted vertices): partition-independent, used for the legacy XDMF restart
        v = self.x[self.gdm]                                                     # (nc, nv, 3)
        order = np.lexsort((v[:, :, 2], v[:, :, 1], v[:, :, 0]), axis=1) if v.size else np.zeros(v.shape[:2], dtype=int)
        self.cell_vertex_xyz = np.take_along_axis(v, order[:, :, None], axis=1)


class _P1Sub:
    """One (collapsed) P1 sub-space of a function space: node index of every (blocked) dof and the parent dof map."""

    def __init__(self, V, sub_index, mi):
        self.sub_index = sub_index
        mesh_index = mi
        if V.num_sub_spaces > 0 and sub_index is not None:
            Vi, parent = V.sub(sub_index).collapse()
            parent = np.asarray(parent, dtype=np.int64)
        else:
            Vi = V
            parent = None
        self.bs = Vi.dofmap.index_map_bs
        imap = Vi.dofmap.index_map
        self.ndofs_own = imap.size_local
        self.ndofs_all = imap.size_local + imap.num_ghosts
        c2d = _as_2d(Vi.dofmap.list, mesh_index.ncells_all)
        if c2d.shape[1] != mesh_index.gdm.shape[1]:
            raise RestartError("checkpointing supports P1 spaces with dofs at the mesh vertices only")
        node_of_dof = np.full(self.ndofs_all, -1, dtype=np.int64)
        node_of_dof[c2d.reshape(-1)] = mesh_index.gdm.reshape(-1)
        if np.any(node_of_dof < 0):
            raise RestartError("dof without node in a P1 space")
        self.node_of_dof = node_of_dof
        if parent is None:
            parent = np.arange(self.ndofs_all * self.bs, dtype=np.int64)
        self.parent = parent.reshape(self.ndofs_all, self.bs)  # parent[d, c] = index in f.x.array


# ----------------------------------------------------------------------------------------------------------------------
# checkpoint manager
# ----------------------------------------------------------------------------------------------------------------------
class Checkpointer:
    """Periodic checkpoints (every `minutes` of wall time) and restore.

    register_p1(name, function)          P1 or mixed-P1 function (each sub-space is stored separately)
    register_quadrature(name, function)  quadrature function with one point per cell (deg_quad = 1)
    """

    def __init__(self, comm, domain, path_h5, minutes):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.domain = domain
        self.path_h5 = path_h5
        self.path_json = path_h5[:-3] + ".json"
        self.seconds = float(minutes) * 60.0
        self.last_write = time.time()
        self.mi = _MeshIndex(domain)
        self.p1 = []    # (name, function, [_P1Sub])
        self.quad = []  # (name, function, qdof_of_cell)
        self._sub_cache = {}
        self.n_written = 0

    # -- registration --------------------------------------------------------------------------------------------------
    def _subs_for(self, V):
        key = id(V)
        if key not in self._sub_cache:
            if V.num_sub_spaces > 0:
                self._sub_cache[key] = [_P1Sub(V, i, self.mi) for i in range(V.num_sub_spaces)]
            else:
                self._sub_cache[key] = [_P1Sub(V, None, self.mi)]
        return self._sub_cache[key]

    def register_p1(self, name, func):
        self.p1.append((name, func, self._subs_for(func.function_space)))

    def register_quadrature(self, name, func):
        space = func.function_space
        q = _as_2d(space.dofmap.list, self.mi.ncells_all)
        if q.shape[1] != 1 or space.dofmap.index_map_bs != 1:
            raise RestartError(f"{name}: expected one scalar quadrature point per cell (deg_quad = 1)")
        self.quad.append((name, func, q[:, 0]))

    # -- writing -------------------------------------------------------------------------------------------------------
    def due(self):
        """Collective: True if the checkpoint interval has elapsed (decided on rank 0)."""
        if self.seconds <= 0.0:
            return False
        flag = (time.time() - self.last_write) >= self.seconds if self.rank == 0 else None
        return bool(self.comm.bcast(flag, root=0))

    def write(self, meta):
        """Collective. Writes all registered fields + meta atomically."""
        t0 = time.time()
        mi = self.mi
        # P1 values: per function and sub-space, the owned dofs with the mesh-file index and coordinates of their node
        p1_data = {}
        for name, func, subs in self.p1:
            arr = func.x.array
            for sub in subs:
                own = np.arange(sub.ndofs_own)
                nodes = sub.node_of_dof[own]
                key = f"{name}/sub{sub.sub_index if sub.sub_index is not None else 0}"
                p1_data[key] = (_gather_rows(self.comm, mi.node_orig[nodes]),
                                _gather_rows(self.comm, mi.x[nodes]),
                                _gather_rows(self.comm, arr[sub.parent[own]]))
        own_cells = np.arange(mi.ncells_own)
        cell_orig = _gather_rows(self.comm, mi.cell_orig[own_cells])
        cell_mid = _gather_rows(self.comm, mi.midpoints[own_cells])
        quad_data = {name: _gather_rows(self.comm, func.x.array[q[own_cells]]) for name, func, q in self.quad}

        err = ""
        if self.rank == 0:
            try:
                h5py = _h5py()
                meta = dict(meta)
                meta["format"] = FORMAT_VERSION
                meta["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                meta["n_p1_dofs"] = {k: int(v[0].shape[0]) for k, v in p1_data.items()}
                meta["n_cells"] = int(cell_orig.shape[0])
                meta["nprocs"] = int(self.comm.Get_size())
                tmp = self.path_h5 + ".tmp"
                with h5py.File(tmp, "w") as h5:
                    for key, (orig, xyz, vals) in p1_data.items():
                        h5.create_dataset(f"p1/{key}/original_index", data=orig)
                        h5.create_dataset(f"p1/{key}/x", data=xyz)
                        h5.create_dataset(f"p1/{key}/values", data=vals)
                    h5.create_dataset("cells/original_index", data=cell_orig)
                    h5.create_dataset("cells/midpoint", data=cell_mid)
                    for key, data in quad_data.items():
                        h5.create_dataset("quad/" + key, data=data)
                    h5.attrs["meta"] = json.dumps(meta, default=_json_default)
                    h5.flush()
                os.replace(tmp, self.path_h5)
                tmpj = self.path_json + ".tmp"
                with open(tmpj, "w") as f:
                    json.dump(meta, f, indent=1, default=_json_default)
                os.replace(tmpj, self.path_json)
            except Exception as exc:  # a failed checkpoint must not kill the run
                err = str(exc)
        err = self.comm.bcast(err, root=0)
        self.last_write = time.time()
        if self.rank == 0:
            if err:
                print(f"WARNING: checkpoint failed ({err})", flush=True)
            else:
                self.n_written += 1
                print(f"[CHECKPOINT] t={meta['t']:.10e} dt={meta['dt']:.3e} step={meta.get('step')} written in "
                      f"{time.time() - t0:.1f} s -> {os.path.basename(self.path_h5)}", flush=True)
        return not err

    # -- reading -------------------------------------------------------------------------------------------------------
    def read_meta(self):
        """Collective. Meta dict of a valid checkpoint or None."""
        meta = None
        if self.rank == 0 and os.path.isfile(self.path_h5):
            try:
                h5py = _h5py()
                with h5py.File(self.path_h5, "r") as h5:
                    meta = json.loads(h5.attrs["meta"])
                if meta.get("format") != FORMAT_VERSION:
                    print(f"[RESTART] {self.path_h5}: unknown format {meta.get('format')}, ignored", flush=True)
                    meta = None
            except Exception as exc:
                print(f"[RESTART] {self.path_h5} not readable ({exc}), ignored", flush=True)
                meta = None
        return self.comm.bcast(meta, root=0)

    def restore(self):
        """Collective. Loads all registered fields from the checkpoint. Raises RestartError on mismatch."""
        mi = self.mi
        ok, msg = True, ""
        try:
            h5py = _h5py()
            with h5py.File(self.path_h5, "r") as h5:
                old_cell_orig = h5["cells/original_index"][...]
                old_mid = h5["cells/midpoint"][...]
                cell_row = _match(old_cell_orig, mi.cell_orig)
                # check with the coordinates; if the mesh-file indices do not fit, fall back to coordinate matching
                if np.any(cell_row < 0) or not np.allclose(old_mid[np.maximum(cell_row, 0)], mi.midpoints, rtol=0.0, atol=1e-10):
                    cell_row = _match_coords(old_mid, mi.midpoints)
                if np.any(cell_row < 0):
                    raise RestartError(f"{int(np.sum(cell_row < 0))} cells of the current mesh are not in the checkpoint "
                                       "(different mesh?)")
                for name, func, subs in self.p1:
                    for sub in subs:
                        key = f"p1/{name}/sub{sub.sub_index if sub.sub_index is not None else 0}"
                        old_orig = h5[key + "/original_index"][...]
                        old_x = h5[key + "/x"][...]
                        nodes = sub.node_of_dof
                        row = _match(old_orig, mi.node_orig[nodes])
                        if np.any(row < 0) or not np.allclose(old_x[np.maximum(row, 0)], mi.x[nodes], rtol=0.0, atol=1e-10):
                            row = _match_coords(old_x, mi.x[nodes])
                        if np.any(row < 0):
                            raise RestartError(f"{key}: {int(np.sum(row < 0))} dofs of the current mesh are not in the "
                                               "checkpoint (different mesh?)")
                        vals = h5[key + "/values"][...]
                        func.x.array[sub.parent.reshape(-1)] = vals[row].reshape(-1)
                for name, func, q in self.quad:
                    vals = h5["quad/" + name][...]
                    func.x.array[q] = vals[cell_row]
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        ok, msg = _all_ok(self.comm, ok, msg)
        if not ok:
            raise RestartError(msg)
        for _, func, _ in self.p1:
            func.x.scatter_forward()
        for _, func, _ in self.quad:
            func.x.scatter_forward()


# ----------------------------------------------------------------------------------------------------------------------
# legacy restart from the periodic field output (run_simulation.xdmf / .h5)
# ----------------------------------------------------------------------------------------------------------------------
def _strip_ns(tag):
    return tag.split("}", 1)[-1]


def parse_xdmf(xdmf_path):
    """{'functions': {name: [(t, h5file, dset), ...]}, 'geometry': (h5file, dset), 'topology': (h5file, dset)}"""
    grid_dir = os.path.dirname(os.path.abspath(xdmf_path))

    def target(text):
        text = (text or "").strip()
        if ":" not in text:
            return None
        f, d = text.split(":", 1)
        return (f if os.path.isabs(f) else os.path.join(grid_dir, f)), d

    res = {"functions": {}, "geometry": None, "topology": None}

    def walk(elem):
        for child in elem:
            if _strip_ns(child.tag) != "Grid":
                walk(child)
                continue
            tval, attrs = None, []
            for sub in child:
                st = _strip_ns(sub.tag)
                if st == "Time":
                    try:
                        tval = float(sub.get("Value"))
                    except (TypeError, ValueError):
                        tval = None
                elif st in ("Geometry", "Topology") and res[st.lower()] is None:
                    for di in sub:
                        if _strip_ns(di.tag) == "DataItem":
                            res[st.lower()] = target(di.text)
                elif st == "Attribute":
                    for di in sub:
                        if _strip_ns(di.tag) == "DataItem":
                            tg = target(di.text)
                            if tg:
                                attrs.append((sub.get("Name"), sub.get("Center"), tg))
            if tval is not None:
                for name, center, tg in attrs:
                    res["functions"].setdefault(name, []).append((tval, center, tg[0], tg[1]))
            walk(child)

    walk(ET.parse(xdmf_path).getroot())
    return res


def restore_from_field_output(comm, domain, xdmf_path, w, wm1, alpha_n, ep_funcs, ep_kind, t_max=None):
    """Legacy restore of w = wm1 (u, s), alpha_n and the plastic strain from the last complete time step of the field output.

    ep_funcs: 2D [e11, e22, e12, e33] with ep_kind="2D" (eps_p written as 2x2, e33 = -(e11 + e22));
              3D [e11, e22, e33, e12, e13, e23] with ep_kind="3D" (eps_p written as 3x3).
              None for elastic runs (only alpha = 0 and w are restored then).
    Returns the time value of the restored output step, or None if nothing usable exists.
    Raises RestartError if the output exists but does not match the mesh.
    """
    rank = comm.Get_rank()
    info = None
    if rank == 0 and os.path.isfile(xdmf_path):
        try:
            info = parse_xdmf(xdmf_path)
        except Exception as exc:  # truncated XML after the kill
            print(f"[RESTART] {os.path.basename(xdmf_path)} not readable ({exc})", flush=True)
    info = comm.bcast(info, root=0)
    if not info or info["geometry"] is None or info["topology"] is None:
        return None
    funcs = info["functions"]
    needed = ["u", "s"] + (["eps_p", "alpha"] if ep_funcs is not None else [])
    if any(n not in funcs for n in needed):
        return None

    def entry(name, t):
        for tt, center, f, d in funcs[name]:
            if abs(tt - t) <= 1e-12 * max(1.0, abs(t)):
                return center, f, d
        return None

    times = sorted({tt for tt, _, _, _ in funcs["u"]}, reverse=True)
    times = [t for t in times if all(entry(n, t) for n in needed) and (t_max is None or t <= t_max * (1 + 1e-12))]
    if not times:
        return None

    mi = _MeshIndex(domain)
    h5py = _h5py()
    ok, msg = True, ""
    node_row = cell_row = None
    try:
        gfile, gdset = info["geometry"]
        tfile, tdset = info["topology"]
        with h5py.File(gfile, "r") as h5:
            geo = np.asarray(h5[gdset][...], dtype=np.float64)
        if geo.shape[1] < 3:
            geo = np.hstack([geo, np.zeros((geo.shape[0], 3 - geo.shape[1]))])
        with h5py.File(tfile, "r") as h5:
            topo = np.asarray(h5[tdset][...], dtype=np.int64)
        node_row = _match_coords(geo, mi.x)
        if np.any(node_row < 0):
            raise RestartError(f"{int(np.sum(node_row < 0))} local nodes not in {os.path.basename(xdmf_path)} (different mesh?)")
        cell_row = _match(np.sort(topo, axis=1), np.sort(node_row[mi.gdm], axis=1))
        if np.any(cell_row < 0):
            raise RestartError(f"{int(np.sum(cell_row < 0))} local cells not in {os.path.basename(xdmf_path)} (different mesh?)")
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {exc}"
    ok, msg = _all_ok(comm, ok, msg)
    if not ok:
        raise RestartError(msg)

    V = w.function_space
    subs = [_P1Sub(V, i, mi) for i in range(V.num_sub_spaces)]
    qdof = None
    if ep_funcs is not None:
        qdof = _as_2d(alpha_n.function_space.dofmap.list, mi.ncells_all)[:, 0]

    for t in times:
        ok, msg = True, ""
        try:
            _, uf, ud = entry("u", t)
            _, sf, sd = entry("s", t)
            with h5py.File(uf, "r") as h5:
                u_rows = np.asarray(h5[ud][...], dtype=np.float64)
            with h5py.File(sf, "r") as h5:
                s_rows = np.asarray(h5[sd][...], dtype=np.float64).reshape(-1)
            bs_u = subs[0].bs
            new = w.x.array.copy()
            new[subs[0].parent.reshape(-1)] = u_rows[node_row[subs[0].node_of_dof], :bs_u].reshape(-1)
            new[subs[1].parent.reshape(-1)] = s_rows[node_row[subs[1].node_of_dof]]
            if ep_funcs is not None:
                _, ef, ed = entry("eps_p", t)
                _, af, ad = entry("alpha", t)
                with h5py.File(ef, "r") as h5:
                    ep = np.asarray(h5[ed][...], dtype=np.float64)[cell_row]
                with h5py.File(af, "r") as h5:
                    al = np.asarray(h5[ad][...], dtype=np.float64).reshape(-1)[cell_row]
            if not np.all(np.isfinite(new)):
                raise RestartError("non-finite values")
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        ok, msg = _all_ok(comm, ok, msg)
        if not ok:
            if rank == 0:
                print(f"[RESTART] field output at t={t:g} not usable ({msg}), trying an older one", flush=True)
            continue
        w.x.array[:] = new
        wm1.x.array[:] = new
        if ep_funcs is not None:
            alpha_n.x.array[qdof] = al
            if ep_kind == "2D":   # eps_p written as 2x2 DG0 tensor (row-major: 11 12 21 22)
                e11, e12, e22 = ep[:, 0], ep[:, 1], ep[:, 3]
                vals = [e11, e22, e12, -(e11 + e22)]
            else:                 # 3x3: 11 12 13 / 21 22 23 / 31 32 33
                vals = [ep[:, 0], ep[:, 4], ep[:, 8], ep[:, 1], ep[:, 2], ep[:, 5]]
            for f, v in zip(ep_funcs, vals):
                f.x.array[qdof] = v
            alpha_n.x.scatter_forward()
            for f in ep_funcs:
                f.x.scatter_forward()
        w.x.scatter_forward()
        wm1.x.scatter_forward()
        return float(t)
    return None


def read_newton_log(path):
    """[(t, dt, iters)] of the accepted steps (full precision, written by sol.write_to_newton_logfile)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            try:
                rows.append((float(p[0]), float(p[1]), int(float(p[2]))))
            except (ValueError, IndexError):
                pass
    return rows


def _data_line_offsets(path):
    """Byte offset after each data line (non-comment) of a text file, and the offset after the leading comment lines."""
    offsets, header_end, pos, seen_data = [], 0, 0, False
    with open(path, "rb") as f:
        for line in f:
            pos += len(line)
            if line.startswith(b"#") or not line.strip():
                if not seen_data:
                    header_end = pos
                continue
            seen_data = True
            offsets.append(pos)
    return offsets, header_end


def legacy_meta(t_x, graph_path, columns_path, newton_log_path, events_path, min_iters, dt_max, scale=1.0):
    """Scalar restart state for a field-output restart at time t_x (rank 0 only).

    The time-stepping state comes from the Newton log (full precision), the cumulative quantities W, D_visc, D_pl_deg,
    D_pl_undeg and the jump-detection state from the graph table (4 significant digits; `scale` = thickness for 3D, where
    the table stores values per unit thickness). Returns a dict in the format of the checkpoint meta data.
    """
    log = read_newton_log(newton_log_path)
    if not log:
        raise RestartError("no Newton log")
    k = int(np.argmin([abs(t - t_x) for t, _, _ in log]))
    t_k, dt_k, it_k = log[k]
    if abs(t_k - t_x) > 1e-12 * max(abs(t_x), 1.0):   # XDMF time values and the Newton log have full precision
        raise RestartError(f"field output time {t_x} not found in the Newton log (closest {t_k})")
    with open(columns_path) as f:
        cols = {name: int(i) for i, name in (tok.split(":", 1) for tok in f.read().split())}
    table = np.atleast_2d(np.loadtxt(graph_path, comments="#"))
    if table.shape[0] <= k:
        raise RestartError(f"graph table has {table.shape[0]} rows, need {k + 1}")
    row = table[k]
    if abs(row[cols["t"]] - t_k) > 1e-3 * max(abs(t_k), 1e-12):
        raise RestartError(f"graph row {k} has t = {row[cols['t']]}, Newton log t = {t_k}")
    # next time step exactly as alex/solution.py would have chosen it after step k
    dt_next = dt_k
    if it_k < min_iters and t_k > np.finfo(float).eps and it_k > 0:
        dt_next = dt_max if 2.0 * dt_k > dt_max else 2.0 * dt_k
    # jump-detection state and event snapshots up to step k
    n_ev, last_ev_t = 0, None
    if os.path.isfile(events_path):
        with open(events_path) as f:
            for line in f:
                p = line.split()
                if line.startswith("#") or len(p) < 3:
                    continue
                if p[2] == "post_jump" and float(p[1]) <= t_k * (1 + 1e-12):
                    n_ev = int(p[0]) // 2 + 1
                    last_ev_t = float(p[1])
    last_event_step = -10 ** 9
    if last_ev_t is not None:
        last_event_step = int(np.argmin([abs(t - last_ev_t) for t, _, _ in log[: k + 1]]))
    g_off, _ = _data_line_offsets(graph_path)
    n_off, _ = _data_line_offsets(newton_log_path)
    e_size = None
    if os.path.isfile(events_path):
        e_off, e_head = _data_line_offsets(events_path)
        e_size = e_off[2 * n_ev - 1] if n_ev > 0 else e_head
    jx = table[: k + 1, cols["Jx"]]
    return {
        "source": "field_output",
        "t": t_k + dt_next, "dt": dt_next, "trestart": t_k, "step": k + 1,
        "Work": float(row[cols["W"]]) * scale, "D_visc": float(row[cols["D_visc"]]) * scale,
        "D_pl_deg": float(row[cols["D_pl_deg"]]) * scale, "D_pl_undeg": float(row[cols["D_pl_undeg"]]) * scale,
        "success_timestep_counter": float(k + 1),
        "state": {"x_ct_prev": float(row[cols["x_ct"]]), "t_prev": t_k, "Jx_prev": float(row[cols["Jx"]]),
                  "Jx_running_max": float(np.max(jx)), "n_event_snapshots": n_ev, "last_event_step": last_event_step,
                  "has_prev": True},
        "graph_rows": k + 1,
        "sizes": {"graphs": g_off[k], "newton_log": n_off[k], "events": e_size},
    }


def parameter_mismatches(parameters, expected, rtol=1e-9):
    """Legacy restart: compare the case parameters with the values the first run appended to parameters.txt.
    expected: {key in parameters.txt: current value}; keys missing in parameters.txt are not checked."""
    bad = []
    for key, val in expected.items():
        if key not in parameters:
            continue
        old = parameters[key]
        try:
            same = abs(float(old) - float(val)) <= rtol * max(abs(float(val)), 1e-300)
        except (TypeError, ValueError):
            same = str(old).strip() == str(val).strip()
        if not same:
            bad.append(f"{key}: parameters.txt {old} vs now {val}")
    return bad
