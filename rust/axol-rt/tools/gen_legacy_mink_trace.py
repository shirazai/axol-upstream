"""Regenerate the hardware-free XR-1 tracker fixture using the frozen Rust source.

Run from any directory. Requires the provenance commit in this checkout and rustc;
it compiles only filter.rs into a temporary executable, never a CAN service.
"""

import subprocess
import tempfile
from pathlib import Path

REVISION = "b32002c0507db5ab03a421c9fb1f2ebf4b7fd49b"
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "rust/axol-rt/tests/data/legacy_mink_tracking.csv"
DRIVER = r"""
mod filter;
use filter::{BandPass, LpDiff, Trapezoid};

fn main() {
    let mut tracker = Trapezoid::new(3.0 * std::f64::consts::PI, 10.5 * std::f64::consts::PI);
    tracker.seed(0.0);
    let mut vel = LpDiff::new(20.0);
    let mut acc = LpDiff::new(20.0);
    let mut fast = LpDiff::new(80.0);
    let mut bp = BandPass::new();
    let mut target = 0.0;
    println!("# tick,dt,target,adopt,overrun,position,velocity,acceleration,fast_velocity,friction,inertia,damping");
    for tick in 0..176 {
        // 30 Hz policy, 240 Hz rest trajectory, then 30 Hz again. Two missing
        // policy updates and two core overruns exercise both changed paths.
        let adopt = ((32..64).contains(&tick) || tick % 8 == 0) && tick != 96 && tick != 104;
        if adopt { target = 0.2 * (tick as f64 * 0.07).sin(); }
        let overrun = tick == 112 || tick == 136;
        let dt = match tick { 0 => 0.0, 112 => 0.025, 136 => 0.015, _ => 1.0 / 240.0 };
        // Exact b320 serve.rs tracked command branch: latest literal target,
        // measured dt, uninterrupted low-pass derivatives, timing-gated BP.
        let (position, _, _) = tracker.update(target, dt);
        let velocity = vel.update(position, dt);
        let acceleration = acc.update(velocity, dt);
        let fast_velocity = fast.update(position, dt);
        let friction = filter::friction(velocity, 0.6, 250.0, 0.15, 0.02);
        let inertia = 0.08 * acceleration;
        let damping = if overrun {
            bp.reset();
            0.0
        } else {
            0.7 * bp.update(fast_velocity - 0.02 * (tick as f64 * 0.03).sin(), 20.0, 0.8, dt)
        };
        println!("{tick},{dt:.17e},{target:.17e},{},{},{position:.17e},{velocity:.17e},{acceleration:.17e},{fast_velocity:.17e},{friction:.17e},{inertia:.17e},{damping:.17e}", u8::from(adopt), u8::from(overrun));
    }
}
"""


def main() -> None:
    source = subprocess.check_output(
        ["git", "show", f"{REVISION}:rust/axol-rt/src/filter.rs"], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(prefix="axol-legacy-filter-") as directory:
        work = Path(directory)
        (work / "filter.rs").write_bytes(source)
        (work / "main.rs").write_text(DRIVER)
        executable = work / "trace"
        subprocess.run(
            [
                "rustc",
                "--edition=2021",
                "-Awarnings",
                str(work / "main.rs"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        trace = subprocess.check_output([str(executable)])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(
        f"# Source: Axol {REVISION}; regenerate: tools/gen_legacy_mink_trace.py\n".encode()
        + trace
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
