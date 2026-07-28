"""
Script to convert Nyx-style raw binary initial conditions into MP-Gadget bigfile
initial conditions.

The Nyx IC file is assumed to be a flat, header-less stream of particle records,
each record being 7 double precision numbers written back-to-back, exactly as
produced by the C++ pattern:

    dm_ics_file.write((char*)&part_x,  sizeof(double)); //position
    dm_ics_file.write((char*)&part_y,  sizeof(double));
    dm_ics_file.write((char*)&part_z,  sizeof(double));
    dm_ics_file.write((char*)&part_mass_unit, sizeof(double)); //mass
    dm_ics_file.write((char*)&part_vx, sizeof(double)); //velocity
    dm_ics_file.write((char*)&part_vy, sizeof(double));
    dm_ics_file.write((char*)&part_vz, sizeof(double));

So each record is 56 little-endian bytes: (x, y, z, mass, vx, vy, vz). Dark matter (MP-Gadget
particle type 1) is given with --input and gas (particle type 0) with --gas-input;
at least one of the two is required and both may be given together. Gas is assumed
to be written with the same 7-double record layout as the dark matter. Multiple
input files per type (e.g. one per MPI rank that wrote the ICs) may be given; they
are concatenated in sorted order. IDs are assigned sequentially and are unique
across both particle types.

Only Position, Velocity, ID and Mass are written: for initial conditions MP-Gadget
reads just these blocks (see petaio.cpp) and sets the gas internal energy from the
InitTemp parameter at run time, so no InternalEnergy/Density/SmoothingLength block
is needed here.

UNITS: The raw Nyx numbers are assumed to be in Nyx's native units, which carry no
little-h: comoving Mpc for position, Msun for mass, and proper (physical peculiar)
km/s for velocity. MP-Gadget's internal code units, written into the output header,
do carry little-h (kpc/h, 1e10 Msun/h, km/s), so the defaults convert between them
using the supplied --hubble:

    position: Mpc     -> kpc/h        (x 1000 * h)
    mass:     Msun    -> 1e10 Msun/h  (x h * 1e-10)
    velocity: km/s    -> km/s         (x 1)

Override --pos-unit, --vel-unit or --mass-unit if your raw numbers use a different
convention. The box size is given in the SAME raw length units as the positions
(Mpc), so it is scaled by --pos-unit too and stays consistent with the particles.

The MP-Gadget velocity convention (UsePeculiarVelocity=1) stores the physical
peculiar velocity v = a dx/dt in km/s, which is the Nyx proper km/s, so no
conversion is needed by default.

Cosmology (Omega0, HubbleParam, ...) is not present in the raw Nyx file and must be
supplied on the command line so it can be written into the bigfile header.
"""

import argparse
import os
import os.path
import numpy as np
import bigfile

# One Nyx particle record: position (3), mass (1), velocity (3), little-endian float64.
NYX_DTYPE = np.dtype([("pos", ("<f8", 3)),
                      ("mass", "<f8"),
                      ("vel", ("<f8", 3))])
RECORD_BYTES = NYX_DTYPE.itemsize


def count_particles(infiles):
    """Total number of particles across all input files, checking record alignment."""
    npart = 0
    for f in infiles:
        sz = os.path.getsize(f)
        if sz % RECORD_BYTES != 0:
            raise IOError("File %s size %d is not a multiple of the %d byte record size"
                          % (f, sz, RECORD_BYTES))
        npart += sz // RECORD_BYTES
    return npart


def compute_nfiles(npart):
    """Work out how many files to split the bigfile blocks into.
       We want less than 2^31 bytes per data array file, and a power of two."""
    nfiles = 1
    # Largest per-particle data array: a 3-vector in double precision (Position).
    maxarray = npart * 3 * 8
    while maxarray // nfiles >= 2**31:
        nfiles *= 2
    return nfiles


def write_header(bf, args, counts):
    """Write the bigfile Header block from the command line cosmology and units.
       counts maps particle type -> number of particles of that type."""
    bf.create("Header")
    battr = bf["Header"].attrs

    totnumpart = np.zeros(6, dtype=np.uint64)
    for ptype, n in counts.items():
        totnumpart[ptype] = n
    battr["TotNumPart"] = totnumpart
    battr["TotNumPartInit"] = totnumpart

    time = args.time
    battr["Time"] = np.float64(time)
    battr["TimeIC"] = np.float64(time)
    battr["Redshift"] = np.float64(1.0 / time - 1.0)
    battr["BoxSize"] = np.float64(args.box * args.pos_unit)

    # Per-particle mass is stored in the 1/Mass block, so leave the MassTable at zero.
    # (MP-Gadget requires the Mass block whenever MassTable[type] <= 0.)
    battr["MassTable"] = np.zeros(6, dtype=np.float64)

    battr["Omega0"] = np.float64(args.omega0)
    battr["OmegaBaryon"] = np.float64(args.omegab)
    battr["OmegaLambda"] = np.float64(args.omegal)
    battr["HubbleParam"] = np.float64(args.hubble)

    # Velocities are stored as physical peculiar velocity v = a dx/dt (km/s).
    battr["UsePeculiarVelocity"] = np.int32(1)

    # Unit system recorded in the snapshot: traditional MP-Gadget defaults.
    battr["UnitLength_in_cm"] = np.float64(args.unit_length_cm)
    battr["UnitMass_in_g"] = np.float64(args.unit_mass_g)
    battr["UnitVelocity_in_cm_per_s"] = np.float64(args.unit_velocity_cms)


def stream_type(bf, ptype, infiles, args, idstart, nfiles):
    """Create the bigfile blocks for one particle type and stream the raw data in.
       IDs are assigned sequentially starting from idstart; returns the next free ID."""
    npart = count_particles(infiles)
    grp = str(ptype)
    bf.create(grp)
    bf.create(grp + "/Position", dtype=("f8", 3), size=npart, Nfile=nfiles)
    bf.create(grp + "/Velocity", dtype=("f4", 3), size=npart, Nfile=nfiles)
    bf.create(grp + "/Mass", dtype=("f4", 1), size=npart, Nfile=nfiles)
    bf.create(grp + "/ID", dtype=("u8", 1), size=npart, Nfile=nfiles)

    offset = 0
    for f in infiles:
        with open(f, "rb") as fh:
            while True:
                chunk = np.fromfile(fh, dtype=NYX_DTYPE, count=args.chunk)
                n = chunk.shape[0]
                if n == 0:
                    break
                # Positions stay double precision, scaled to the output length unit.
                bf[grp + "/Position"].write(offset, chunk["pos"] * args.pos_unit)
                # Velocity and mass are stored single precision, as MP-Gadget expects.
                vel = (chunk["vel"] * args.vel_unit).astype(np.float32)
                bf[grp + "/Velocity"].write(offset, vel)
                mass = (chunk["mass"] * args.mass_unit).astype(np.float32)
                bf[grp + "/Mass"].write(offset, mass)
                ids = np.arange(offset, offset + n, dtype=np.uint64) + np.uint64(idstart)
                bf[grp + "/ID"].write(offset, ids)
                offset += n
        print("Copied %s (type %d, %d particles so far)" % (f, ptype, offset))

    if offset != npart:
        raise IOError("Read %d particles but expected %d for type %d" % (offset, npart, ptype))
    return idstart + npart


# Names for the particle types we can write, for informative messages.
PTYPE_NAMES = {0: "gas", 1: "dark matter"}


def convert(args):
    """Stream the raw Nyx particle data into MP-Gadget bigfile blocks."""
    # Gather the (particle type, input files) to write. Gas is type 0, DM is type 1.
    types = []
    if args.gas_input:
        types.append((0, sorted(args.gas_input)))
    if args.input:
        types.append((1, sorted(args.input)))
    if len(types) == 0:
        raise IOError("No input files given: supply --input (dark matter) and/or --gas-input (gas).")

    counts = {ptype: count_particles(fl) for ptype, fl in types}
    nfiles = compute_nfiles(max(counts.values()))
    summary = ", ".join("%d %s" % (counts[pt], PTYPE_NAMES[pt]) for pt, _ in types)
    print("Converting %s into %s (Nfile=%d)" % (summary, args.output, nfiles))

    if os.path.exists(args.output):
        raise IOError("Refusing to overwrite existing output %s" % args.output)

    bf = bigfile.BigFile(args.output, create=True)
    write_header(bf, args, counts)

    idstart = args.firstid
    for ptype, infiles in types:
        idstart = stream_type(bf, ptype, infiles, args, idstart, nfiles)

    print("Done. Wrote %s to %s" % (summary, args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=str, nargs='+', default=None,
                        help='Dark matter (type 1) Nyx raw binary IC file(s). Multiple files are concatenated in sorted order.')
    parser.add_argument('--gas-input', type=str, nargs='+', default=None,
                        help='Gas (type 0) Nyx raw binary IC file(s), same record layout as the dark matter.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output MP-Gadget bigfile directory to create.')
    parser.add_argument('--box', type=float, required=True,
                        help='Box size, in the SAME raw length units as the input positions (scaled by --pos-unit).')

    # Cosmology (not present in the raw file).
    parser.add_argument('--omega0', type=float, required=True, help='Omega_matter (total matter density).')
    parser.add_argument('--hubble', type=float, required=True, help='Dimensionless Hubble parameter h.')
    parser.add_argument('--omegab', type=float, default=0.0, help='Omega_baryon (0 for DM-only ICs).')
    parser.add_argument('--omegal', type=float, default=None, help='Omega_Lambda (defaults to 1 - Omega0).')

    # Time of the ICs: give exactly one of redshift or scale factor.
    tgroup = parser.add_mutually_exclusive_group(required=True)
    tgroup.add_argument('--redshift', type=float, help='Redshift of the ICs.')
    tgroup.add_argument('--time', dest='scalefac', type=float, help='Scale factor a of the ICs.')

    # Unit conversion factors from the raw Nyx numbers to MP-Gadget code units.
    # The defaults assume Nyx native units (no little-h) and are derived from --hubble below.
    parser.add_argument('--pos-unit', type=float, default=None,
                        help='Multiply raw positions (and --box) by this to get output length units (default 1000*h: Mpc -> kpc/h).')
    parser.add_argument('--vel-unit', type=float, default=None,
                        help='Multiply raw velocities by this to get physical peculiar velocity in km/s (default 1: proper km/s).')
    parser.add_argument('--mass-unit', type=float, default=None,
                        help='Multiply raw masses by this to get output mass units (default h*1e-10: Msun -> 1e10 Msun/h).')

    # Output unit system written into the header (defaults to traditional MP-Gadget units).
    parser.add_argument('--unit-length-cm', type=float, default=3.085678e21, help='UnitLength_in_cm (default 1 kpc/h).')
    parser.add_argument('--unit-mass-g', type=float, default=1.989e43, help='UnitMass_in_g (default 1e10 Msun/h).')
    parser.add_argument('--unit-velocity-cms', type=float, default=1e5, help='UnitVelocity_in_cm_per_s (default 1 km/s).')

    parser.add_argument('--firstid', type=int, default=1, help='ID of the first particle (IDs are assigned sequentially).')
    parser.add_argument('--chunk', type=int, default=1 << 22, help='Number of particles to read/write per chunk.')

    args = parser.parse_args()

    # Resolve the time of the ICs into a scale factor.
    if args.redshift is not None:
        args.time = 1.0 / (1.0 + args.redshift)
    else:
        args.time = args.scalefac
    # Default Omega_Lambda to a flat universe.
    if args.omegal is None:
        args.omegal = 1.0 - args.omega0

    # Default unit factors convert Nyx native units (no little-h) into MP-Gadget code units.
    if args.pos_unit is None:
        args.pos_unit = 1000.0 * args.hubble   # Mpc -> kpc/h
    if args.vel_unit is None:
        args.vel_unit = 1.0                     # proper km/s -> km/s
    if args.mass_unit is None:
        args.mass_unit = args.hubble * 1e-10    # Msun -> 1e10 Msun/h

    convert(args)
