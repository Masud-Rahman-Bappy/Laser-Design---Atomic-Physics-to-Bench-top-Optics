"""Project: compact diode-pumped Nd:GdVO4 laser design.

=============================================================================
1. SYSTEM AND DESIGN VARIABLES
=============================================================================

The optical train is

  diode --L1-- thin lens --L2-- curved/dichroic mirror M1 --d1--
  gain crystal (length Lg) --d2-- frequency doubler --d3-- output mirror M2.

The principal longitudinal constraint is

  L1 + L2 + d1 + Lg + d2 + d3 <= Lmax = 10 cm.

The project chooses a symmetric emission cavity.  Its minimum waist is located
at the frequency doubler, and the effective optical distances from that waist
to both mirrors are chosen equal:

  x = d3,
  y = d1 + Lg/n + d2,
  x = y,

where the crystal contributes Lg/n rather than Lg to Gaussian diffraction.
The two mirror radii then become equal, r1=r2.

=============================================================================
2. GAIN-CRYSTAL ABSORPTION
=============================================================================

For an unsaturated absorber,

  absorbed fraction = 1-exp(-sigma_a * DeltaN * Lg).

The minimum length for a target fraction p is

  Lg_min = -ln(1-p)/(sigma_a*DeltaN).

The report obtains about 1.4 mm at zero emission and deliberately selects
Lg=2.7 mm to retain high absorption after gain saturation begins.  This code
reports both the unsaturated estimate and the selected design value.

=============================================================================
3. RESONATOR STABILITY AND GAUSSIAN WAIST
=============================================================================

For cavity effective length d=x+y and mirror radii r1,r2,

  g1 = 1-d/r1,     g2 = 1-d/r2,
  0 <= g1*g2 <= 1.

The project derives, for the symmetric design, a cavity-waist/mirror-radius
family that also obeys the aperture condition.  A Gaussian intensity is

  I(rho)=I0 exp[-2 rho^2/w(z)^2].

Demanding I(rho_edge)/I0 <= exp(-4) gives w(z)^2 <= rho_edge^2/2.  The code
uses the project formulas to find the largest permitted emission waist and the
corresponding equal mirror radius as d3 is varied.

=============================================================================
4. ABCD PROPAGATION OF THE PUMP BEAM
=============================================================================

The complex beam parameter is

  1/q = 1/R - i*lambda/(pi*w^2).

For an ABCD system,

  q_out = (A*q_in+B)/(C*q_in+D).

The matrices used here are

  free space:       [[1,L],[0,1]],
  thin lens:        [[1,0],[-1/f,1]],
  curved reflection:[[1,0],[+2/r1,1]],

where the last sign follows the propagation convention used in the project.
The pump starts at a waist w0=467 um, travels to the lens, then M1, reflects,
and reaches the middle of the gain crystal.  For each d1, the code searches L1
with f=L1 so that the pump waist best matches the emission-mode radius inside
the crystal.  It also records 1/R at the crystal; an ideal focus has 1/R=0.

=============================================================================
5. THREE-LEVEL STEADY-STATE APPROXIMATION
=============================================================================

Nd:GdVO4 is physically a four-level system, but the fast upper relaxation lets
the report treat it effectively as three levels.  With no emission field, the
pump rate is

  Wp = sigma_a*Ip/(h*nu_p),

and the approximate unsaturated inversion is

  DeltaN0 = Nd*Wp/(Wp+1/tau21).

The saturation intensity is

  Is = h*nu_em/(sigma_s*tau21),

and the dimensionless single-pass gain parameter is

  gamma = DeltaN0*sigma_s*Lg.

=============================================================================
6. NONLINEAR OUTPUT COUPLING AND R2 OPTIMIZATION
=============================================================================

The frequency doubler introduces an intensity-dependent loss.  The project
models the effective infrared reflectivity as

  R2_eff(i) = R2*[1-tanh(sqrt(i))^2] = R2*sech(sqrt(i))^2,

where i=I/Is is normalized intensity.  Writing the project equations in this
normalized form avoids mixing W/m^2 with GW/cm^2.  The forward/backward fixed
point equations are solved numerically for every trial R2, and

  i_out = (1-R2)*(i_plus+i_minus)

is maximized.  With the default project parameters the optimum remains near
R2=0.54, consistent with the report.

IMPORTANT PROJECT CONSISTENCY NOTE
----------------------------------
The narrative mentions 1 kW in one location, while the supplied MATLAB code
uses P_pump=1 W and later states Pin=0.95 W.  This Python code defaults to 1 W
to reproduce the actual calculation.  Change --pump-power 1000 to investigate
the 1 kW case.  The very small power/efficiency estimate in the report should
not be treated as a validated experimental prediction; it follows a simplified
rate/area model.  Geometry, normalized output-coupling optimization, and SI
population calculations are kept separate in the exported table.

=============================================================================
7. OUTPUTS
=============================================================================

  design_summary.csv
  optimized_design_curve.csv
  Figure_output_coupler_optimization.png
  Figure_cavity_geometry.png
  Figure_pump_matching_surfaces.png
  Figure_optimized_lengths.png
  Figure_beam_waist_comparison.png
  Figure_final_laser_setup.png
  Table_design_summary.png

"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class LaserParameters:
    """SI-valued material, optical, and packaging parameters."""

    sigma_abs: float = 5.0e-23       # m^2 (5e-19 cm^2)
    sigma_em: float = 7.7e-23        # m^2 (7.7e-19 cm^2)
    doping: float = 1.35e26          # m^-3 (1.35e20 cm^-3)
    lambda_em: float = 1063.8e-9     # m
    lambda_pump: float = 808.5e-9    # m
    lambda_green: float = 531.9e-9   # m
    tau21: float = 100.0e-6          # s
    pump_transmission: float = 0.95
    pump_power: float = 1.0           # W; matches supplied code, not the 1 kW sentence
    diode_waist: float = 467.0e-6     # m
    gain_length: float = 2.7e-3       # m, selected design length
    gain_index: float = 2.0
    target_absorption: float = 0.9999
    aperture_radius: float = 5.0e-3   # m (1 cm diameter)
    max_total_length: float = 0.10    # m
    h: float = 6.62607015e-34
    c: float = 299792458.0

    @property
    def pump_photon_energy(self) -> float:
        return self.h * self.c / self.lambda_pump

    @property
    def emission_photon_energy(self) -> float:
        return self.h * self.c / self.lambda_em

    @property
    def pump_intensity(self) -> float:
        # The report uses P/(pi*w^2), rather than the peak Gaussian 2P/(pi*w^2).
        return self.pump_transmission * self.pump_power / (np.pi * self.diode_waist**2)

    @property
    def pump_rate(self) -> float:
        return self.sigma_abs * self.pump_intensity / self.pump_photon_energy

    @property
    def unsaturated_inversion(self) -> float:
        Wp = self.pump_rate
        return self.doping * Wp / (Wp + 1.0 / self.tau21)

    @property
    def saturation_intensity(self) -> float:
        return self.emission_photon_energy / (self.sigma_em * self.tau21)

    @property
    def gain_parameter(self) -> float:
        return self.unsaturated_inversion * self.sigma_em * self.gain_length

    @property
    def absorption_length_unsaturated(self) -> float:
        # Small-signal estimate uses essentially all ions in the lower state.
        return -np.log(1.0 - self.target_absorption) / (self.sigma_abs * self.doping)


def free_space(length: float) -> np.ndarray:
    return np.array([[1.0, length], [0.0, 1.0]])


def thin_lens(focal_length: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-1.0 / focal_length, 1.0]])


def curved_reflection(radius: float) -> np.ndarray:
    """Project sign convention: M=[[1,0],[+2/r,1]]."""
    return np.array([[1.0, 0.0], [2.0 / radius, 1.0]])


def q_from_waist(waist: float, wavelength: float, inverse_radius: float = 0.0) -> complex:
    inverse_q = inverse_radius - 1j * wavelength / (np.pi * waist**2)
    return 1.0 / inverse_q


def propagate_q(q_in: complex, matrix: np.ndarray) -> complex:
    A, B, C, D = matrix.ravel()
    return (A * q_in + B) / (C * q_in + D)


def waist_and_inverse_radius(q: complex, wavelength: float) -> tuple[float, float]:
    inverse_q = 1.0 / q
    waist_sq = -wavelength / (np.pi * inverse_q.imag)
    return float(np.sqrt(max(waist_sq, 0.0))), float(inverse_q.real)


def gaussian_waist_at_z(w0: float, z: np.ndarray | float, wavelength: float) -> np.ndarray:
    z_rayleigh = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1.0 + (np.asarray(z) / z_rayleigh) ** 2)


def cavity_family(parameters: LaserParameters, samples: int = 500) -> dict[str, np.ndarray]:
    """Reproduce the project symmetric-cavity aperture-limited family."""
    maximum_x = 0.5 * (
        parameters.max_total_length
        - (1.0 - 1.0 / parameters.gain_index) * parameters.gain_length
    )
    x = np.linspace(1.0e-5, maximum_x, samples)
    kfactor = np.pi / parameters.lambda_em
    N = parameters.aperture_radius**2 * kfactor**2 / 2.0
    discriminant = np.maximum(N**2 - 4.0 * kfactor**2 * x**2, 0.0)
    waist_sq = (N + np.sqrt(discriminant)) / (2.0 * kfactor**2)
    F = (waist_sq / ((parameters.lambda_em / np.pi) * x)) ** 2
    radius = 2.0 * x * (F + 1.0) / (F - 1.0)
    g = 1.0 - 2.0 * x / radius
    stability = g**2
    return {"x": x, "waist": np.sqrt(waist_sq), "radius": radius, "stability": stability}


def choose_cavity(parameters: LaserParameters) -> dict[str, float]:
    """Follow the MATLAB selection: use the mean waist and infer d3/radius."""
    family = cavity_family(parameters)
    waist_sq_mean = float(np.mean(family["waist"] ** 2))
    d3 = np.sqrt(
        (np.pi / parameters.lambda_em) ** 2
        * (parameters.aperture_radius**2 / 2.0 - waist_sq_mean)
        * waist_sq_mean
    )
    radius = float(np.interp(d3, family["x"], family["radius"]))
    d1_plus_d2 = d3 - parameters.gain_length / parameters.gain_index
    L1_plus_L2 = parameters.max_total_length - (
        d1_plus_d2 + d3 + parameters.gain_length
    )
    return {
        "cavity_waist": np.sqrt(waist_sq_mean),
        "d3": d3,
        "mirror_radius": radius,
        "d1_plus_d2": d1_plus_d2,
        "L1_plus_L2": L1_plus_L2,
    }


def pump_at_locations(
    parameters: LaserParameters, L1: float, L2: float, d1: float, mirror_radius: float
) -> dict[str, float]:
    """Propagate pump q from diode to lens, M1, and crystal center."""
    q0 = q_from_waist(parameters.diode_waist, parameters.lambda_pump)

    M_lens_plane = free_space(L1)
    q_lens = propagate_q(q0, M_lens_plane)

    # f=L1 is the imaging choice made in the project.
    total_to_mirror = free_space(L2) @ thin_lens(L1) @ free_space(L1)
    q_mirror = propagate_q(q0, total_to_mirror)

    total_to_crystal = (
        free_space(d1 + 0.5 * parameters.gain_length)
        @ curved_reflection(mirror_radius)
        @ total_to_mirror
    )
    q_crystal = propagate_q(q0, total_to_crystal)

    w_lens, invR_lens = waist_and_inverse_radius(q_lens, parameters.lambda_pump)
    w_mirror, invR_mirror = waist_and_inverse_radius(q_mirror, parameters.lambda_pump)
    w_crystal, invR_crystal = waist_and_inverse_radius(q_crystal, parameters.lambda_pump)
    return {
        "w_lens": w_lens, "invR_lens": invR_lens,
        "w_mirror": w_mirror, "invR_mirror": invR_mirror,
        "w_crystal": w_crystal, "invR_crystal": invR_crystal,
    }


def emission_waist_in_crystal(
    parameters: LaserParameters, cavity_waist: float, d2: np.ndarray | float
) -> np.ndarray:
    distance = np.asarray(d2) + 0.5 * parameters.gain_length / parameters.gain_index
    return gaussian_waist_at_z(cavity_waist, distance, parameters.lambda_em)


def optimize_longitudinal_design(
    parameters: LaserParameters, cavity: dict[str, float],
    n_d1: int = 90, n_L1: int = 130
) -> dict[str, np.ndarray]:
    """Grid-search L1 for each d1 to match pump and emission waists at crystal."""
    dsum = cavity["d1_plus_d2"]
    Lsum = cavity["L1_plus_L2"]
    d1_values = np.linspace(1.0e-3, dsum - 1.0e-3, n_d1)
    L1_values = np.linspace(1.0e-3, Lsum - 1.0e-3, n_L1)
    mismatch = np.empty((n_L1, n_d1))
    inv_radius = np.empty_like(mismatch)
    pump_waist = np.empty_like(mismatch)
    emission_waist = np.empty_like(mismatch)

    for column, d1 in enumerate(d1_values):
        d2 = dsum - d1
        target = float(emission_waist_in_crystal(parameters, cavity["cavity_waist"], d2))
        for row, L1 in enumerate(L1_values):
            L2 = Lsum - L1
            result = pump_at_locations(parameters, L1, L2, d1, cavity["mirror_radius"])
            pump_waist[row, column] = result["w_crystal"]
            emission_waist[row, column] = target
            mismatch[row, column] = abs(result["w_crystal"] ** 2 - target**2)
            inv_radius[row, column] = result["invR_crystal"]

    # Refine every column continuously.  Using only the discrete surface-grid
    # minimum creates artificial staircase/sawtooth curves in L1 and waist.
    L1_best = np.empty_like(d1_values)
    pump_best = np.empty_like(d1_values)
    emission_best = np.empty_like(d1_values)
    invR_best = np.empty_like(d1_values)
    for column, d1 in enumerate(d1_values):
        d2 = dsum - d1
        target = float(emission_waist_in_crystal(parameters, cavity["cavity_waist"], d2))

        def objective(L1: float) -> float:
            result = pump_at_locations(
                parameters, L1, Lsum - L1, float(d1), cavity["mirror_radius"]
            )
            return abs(result["w_crystal"] ** 2 - target**2)

        refined = minimize_scalar(
            objective, bounds=(1.0e-3, Lsum - 1.0e-3), method="bounded",
            options={"xatol": 1.0e-10},
        )
        L1_best[column] = refined.x
        result = pump_at_locations(
            parameters, refined.x, Lsum - refined.x, float(d1), cavity["mirror_radius"]
        )
        pump_best[column] = result["w_crystal"]
        emission_best[column] = target
        invR_best[column] = result["invR_crystal"]
    design = {
        "d1_grid": d1_values,
        "L1_grid": L1_values,
        "mismatch_surface": mismatch,
        "invR_surface": inv_radius,
        "L1": L1_best,
        "L2": Lsum - L1_best,
        "d1": d1_values,
        "d2": dsum - d1_values,
        "pump_waist_crystal": pump_best,
        "emission_waist_crystal": emission_best,
        "invR_crystal": invR_best,
    }

    w_lens, w_mirror = [], []
    for L1, L2, d1 in zip(design["L1"], design["L2"], design["d1"]):
        result = pump_at_locations(parameters, L1, L2, d1, cavity["mirror_radius"])
        w_lens.append(result["w_lens"])
        w_mirror.append(result["w_mirror"])
    design["pump_waist_lens"] = np.asarray(w_lens)
    design["pump_waist_mirror"] = np.asarray(w_mirror)
    design["emission_waist_m1"] = gaussian_waist_at_z(
        cavity["cavity_waist"],
        -(design["d1"] + 0.5 * parameters.gain_length / parameters.gain_index + design["d2"]),
        parameters.lambda_em,
    )
    design["emission_waist_m2"] = gaussian_waist_at_z(
        cavity["cavity_waist"], cavity["d3"], parameters.lambda_em
    ) * np.ones_like(design["d1"])
    return design


def positive_fixed_point(function, maximum: float = 30.0) -> float:
    """Find the largest non-negative root of i-function(i)=0 by bracketing."""
    grid = np.linspace(0.0, maximum, 2500)
    residual = np.array([x - max(function(x), 0.0) for x in grid])
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], residual[:-1], residual[1:]):
        if fa == 0.0 and a > 0:
            roots.append(a)
        elif fa * fb < 0.0:
            roots.append(brentq(lambda x: x - max(function(x), 0.0), a, b))
    return max(roots) if roots else 0.0


def intracavity_normalized_intensities(R2: float, gamma: float) -> tuple[float, float]:
    """Solve the project's two nonlinear normalized-intensity equations."""
    def rhs(i: float, forward: bool) -> float:
        effective_R = R2 / np.cosh(np.sqrt(max(i, 0.0))) ** 2
        effective_R = np.clip(effective_R, 1.0e-14, 1.0 - 1.0e-14)
        root_R = np.sqrt(effective_R)
        numerator = gamma + 0.5 * np.log(effective_R)
        denominator = (1.0 - root_R) * (
            1.0 + root_R if forward else 1.0 + 1.0 / root_R
        )
        return numerator / denominator

    i_plus = positive_fixed_point(lambda i: rhs(i, True))
    i_minus = positive_fixed_point(lambda i: rhs(i, False))
    return i_plus, i_minus


def optimize_output_coupler(parameters: LaserParameters, samples: int = 180) -> dict[str, np.ndarray | float]:
    reflectivity = np.linspace(0.25, 0.98, samples)
    plus = np.empty_like(reflectivity)
    minus = np.empty_like(reflectivity)
    for index, R2 in enumerate(reflectivity):
        plus[index], minus[index] = intracavity_normalized_intensities(
            float(R2), parameters.gain_parameter
        )
    total_output = (1.0 - reflectivity) * (plus + minus)
    best = int(np.argmax(total_output))
    return {
        "R2": reflectivity, "i_plus_out": (1.0 - reflectivity) * plus,
        "i_minus_out": (1.0 - reflectivity) * minus,
        "i_total_out": total_output,
        "R2_opt": float(reflectivity[best]),
        "i_plus": float(plus[best]), "i_minus": float(minus[best]),
        "i_out_opt": float(total_output[best]),
    }


def approximate_populations(parameters: LaserParameters, normalized_intensity: float) -> tuple[float, float, float]:
    """Project's simplified N1~0 steady-state population estimate."""
    Wp = parameters.pump_rate
    stimulated_rate = normalized_intensity / parameters.tau21
    denominator = Wp + stimulated_rate + 1.0 / parameters.tau21
    N2 = parameters.doping * Wp / denominator
    N0 = parameters.doping - N2
    N1 = 0.0
    return N0, N1, N2


def representative_design(
    parameters: LaserParameters, cavity: dict[str, float], design: dict[str, np.ndarray],
    output_model: dict[str, np.ndarray | float]
) -> dict[str, float]:
    """Choose the scan point closest to the report's d1=1.47 cm example."""
    index = int(np.argmin(abs(design["d1"] - 0.0147)))
    i_cavity = float(output_model["i_plus"] + output_model["i_minus"])
    N0, N1, N2 = approximate_populations(parameters, i_cavity)
    values = {
        "L1_m": float(design["L1"][index]), "L2_m": float(design["L2"][index]),
        "d1_m": float(design["d1"][index]), "Lg_m": parameters.gain_length,
        "d2_m": float(design["d2"][index]), "d3_m": cavity["d3"],
        "total_length_m": float(design["L1"][index] + design["L2"][index]
                                + design["d1"][index] + parameters.gain_length
                                + design["d2"][index] + cavity["d3"]),
        "mirror_radius_m": cavity["mirror_radius"],
        "cavity_waist_m": cavity["cavity_waist"],
        "pump_waist_crystal_m": float(design["pump_waist_crystal"][index]),
        "emission_waist_crystal_m": float(design["emission_waist_crystal"][index]),
        "pump_inverse_radius_crystal_per_m": float(design["invR_crystal"][index]),
        "R2_opt": float(output_model["R2_opt"]),
        "normalized_output_intensity": float(output_model["i_out_opt"]),
        "pump_intensity_W_per_m2": parameters.pump_intensity,
        "saturation_intensity_W_per_m2": parameters.saturation_intensity,
        "gain_parameter": parameters.gain_parameter,
        "absorption_length_estimate_m": parameters.absorption_length_unsaturated,
        "N0_per_m3": N0, "N1_per_m3": N1, "N2_per_m3": N2,
    }
    return values


def export_results(output: Path, parameters: LaserParameters, cavity: dict[str, float],
                   design: dict[str, np.ndarray], summary: dict[str, float]) -> None:
    with (output / "design_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["quantity", "value_SI"])
        for key, value in summary.items(): writer.writerow([key, f"{value:.12g}"])

    with (output / "optimized_design_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["d1_m", "L1_m", "L2_m", "d2_m", "pump_waist_crystal_m",
                         "emission_waist_crystal_m", "inverse_radius_crystal_per_m"])
        for row in zip(design["d1"], design["L1"], design["L2"], design["d2"],
                       design["pump_waist_crystal"], design["emission_waist_crystal"],
                       design["invR_crystal"]):
            writer.writerow([f"{v:.12g}" for v in row])


def plot_final_setup(
    output: Path,
    parameters: LaserParameters,
    cavity: dict[str, float],
    summary: dict[str, float],
) -> None:
    """Draw a final, dimensioned setup tied directly to computed results.

    Horizontal locations use the representative optimized distances.  Beam
    envelopes are plotted on a common illustrative vertical scale so that both
    the 808.5-nm pump path and 1063.8-nm resonator mode remain readable; the
    annotated numerical waist values retain their physical units.
    """
    from matplotlib.patches import Ellipse, Polygon, Rectangle

    L1, L2 = summary["L1_m"], summary["L2_m"]
    d1, Lg = summary["d1_m"], summary["Lg_m"]
    d2, d3 = summary["d2_m"], summary["d3_m"]

    z_diode = 0.0
    z_lens = L1
    z_m1 = z_lens + L2
    z_crystal_start = z_m1 + d1
    z_crystal_end = z_crystal_start + Lg
    z_crystal_center = 0.5 * (z_crystal_start + z_crystal_end)
    z_doubler = z_crystal_end + d2
    z_m2 = z_doubler + d3

    pump = pump_at_locations(parameters, L1, L2, d1, summary["mirror_radius_m"])
    pump_z = np.array([z_diode, z_lens, z_m1, z_crystal_center])
    pump_w = np.array([
        parameters.diode_waist, pump["w_lens"], pump["w_mirror"], pump["w_crystal"]
    ])
    pump_display = 0.005 + 0.055 * pump_w / max(pump_w)

    cavity_z = np.linspace(z_m1, z_m2, 700)
    distance_from_waist = cavity_z - z_doubler
    cavity_w = gaussian_waist_at_z(
        summary["cavity_waist_m"], distance_from_waist, parameters.lambda_em
    )
    cavity_display = 0.012 + 0.042 * cavity_w / max(cavity_w)

    fig, ax = plt.subplots(figsize=(16, 7.8), constrained_layout=True)
    ax.axhline(0, color="#555555", lw=1.2)

    # Pump path (blue) and resonator mode (red); transparency preserves elements.
    pump_dense = np.linspace(z_diode, z_crystal_center, 500)
    pump_env = np.interp(pump_dense, pump_z, pump_display)
    ax.fill_between(pump_dense, -pump_env, pump_env, color="#3d8bd3", alpha=0.25)
    ax.plot(pump_dense, pump_env, color="#2a73b8", lw=1.6, label="808.5-nm pump envelope")
    ax.plot(pump_dense, -pump_env, color="#2a73b8", lw=1.6)
    ax.fill_between(cavity_z, -cavity_display, cavity_display, color="#d94b45", alpha=0.18)
    ax.plot(cavity_z, cavity_display, color="#b63834", lw=1.6,
            label="1063.8-nm cavity mode")
    ax.plot(cavity_z, -cavity_display, color="#b63834", lw=1.6)

    # Diode and collimating/focusing lens.
    ax.add_patch(Rectangle((z_diode - 0.0015, -0.025), 0.003, 0.05,
                           facecolor="#365f9d", edgecolor="black", zorder=5))
    ax.text(z_diode - 0.0015, 0.073, "Pump diode\n808.5 nm",
            ha="center", va="bottom", fontsize=9)
    ax.add_patch(Ellipse((z_lens, 0), 0.0024, 0.11,
                         facecolor="#a8def0", edgecolor="#176b87", lw=1.5, zorder=6))
    ax.text(z_lens + 0.004, 0.073, "Lens\n" + fr"$f=L_1={100*L1:.3f}$ cm",
            ha="left", va="bottom", fontsize=9)

    # Curved mirrors are drawn as narrow wedges to distinguish orientation.
    m1 = Polygon([[z_m1 - .0012, -.065], [z_m1 + .0012, -.055],
                  [z_m1 + .0012, .055], [z_m1 - .0012, .065]],
                 closed=True, facecolor="#c7ccd4", edgecolor="black", zorder=7)
    m2 = Polygon([[z_m2 + .0012, -.065], [z_m2 - .0012, -.055],
                  [z_m2 - .0012, .055], [z_m2 + .0012, .065]],
                 closed=True, facecolor="#c7ccd4", edgecolor="black", zorder=7)
    ax.add_patch(m1); ax.add_patch(m2)
    ax.text(z_m1, 0.073, "M1\ncurved/dichroic", ha="center", va="bottom", fontsize=9)
    ax.text(z_m2, 0.073, "M2 output\n" + fr"$R_2={summary['R2_opt']:.3f}$",
            ha="center", va="bottom", fontsize=9)

    # Gain medium and frequency doubler.
    ax.add_patch(Rectangle((z_crystal_start, -.043), Lg, .086,
                           facecolor="#78b96b", edgecolor="#245d21", lw=1.4, zorder=6))
    ax.text(z_crystal_center, 0.073, "Nd:GdVO$_4$\ngain crystal",
            ha="center", va="bottom", fontsize=9)
    doubler_width = 0.0035
    ax.add_patch(Rectangle((z_doubler - doubler_width/2, -.042), doubler_width, .084,
                           facecolor="#d99be5", edgecolor="#6d2878", lw=1.4, zorder=7))
    ax.text(z_doubler, 0.073, "Frequency doubler\nminimum cavity waist",
            ha="center", va="bottom", fontsize=9)

    # Green output arrow from M2.
    ax.annotate("531.9-nm output", xy=(z_m2 + .014, 0), xytext=(z_m2 + .003, 0),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#24a34a"),
                color="#147832", va="center", fontsize=10)

    # Dimension arrows use the actual computed segment endpoints.
    segments = [
        (z_diode, z_lens, r"$L_1$", L1),
        (z_lens, z_m1, r"$L_2$", L2),
        (z_m1, z_crystal_start, r"$d_1$", d1),
        (z_crystal_start, z_crystal_end, r"$L_g$", Lg),
        (z_crystal_end, z_doubler, r"$d_2$", d2),
        (z_doubler, z_m2, r"$d_3$", d3),
    ]
    alternating_y = [-0.095, -0.125]
    for index, (left, right, symbol, length) in enumerate(segments):
        y = alternating_y[index % 2]
        ax.annotate("", xy=(right, y), xytext=(left, y),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.1))
        ax.vlines([left, right], y, -0.068, color="#777777", lw=.7)
        ax.text((left + right)/2, y - .008, fr"{symbol}={100*length:.3f} cm",
                ha="center", va="top", fontsize=8.5)

    summary_text = (
        fr"Computed design: total length = {100*summary['total_length_m']:.3f} cm" "\n"
        fr"$r_1=r_2={100*summary['mirror_radius_m']:.3f}$ cm, "
        fr"$w_{{C0}}={1e3*summary['cavity_waist_m']:.3f}$ mm" "\n"
        fr"pump waist at crystal = {1e3*summary['pump_waist_crystal_m']:.3f} mm, "
        fr"$\gamma={summary['gain_parameter']:.3f}$"
    )
    ax.text(0.015, 0.965, summary_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round,pad=.45", facecolor="white", alpha=.9))

    ax.set_xlim(-.006, z_m2 + .018)
    ax.set_ylim(-.155, .135)
    ax.set_yticks([])
    ax.set_xlabel("Physical position along the computed 10-cm system (m)")
    ax.set_title("Final laser setup linked to the optimized code results", fontsize=16)
    ax.legend(loc="upper right", fontsize=9)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    fig.savefig(output / "Figure_final_laser_setup.png", dpi=250)


def plot_results(output: Path, parameters: LaserParameters, cavity: dict[str, float],
                 design: dict[str, np.ndarray], output_model: dict[str, np.ndarray | float],
                 summary: dict[str, float]) -> None:
    # Output coupler optimization
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.plot(output_model["R2"], output_model["i_total_out"], lw=2.5, label="total output")
    ax.plot(output_model["R2"], output_model["i_plus_out"], "--", lw=1.8, label="forward")
    ax.plot(output_model["R2"], output_model["i_minus_out"], ":", lw=2, label="backward")
    ax.axvline(output_model["R2_opt"], color="gray", ls="--",
               label=fr"optimum $R_2={output_model['R2_opt']:.3f}$")
    ax.set(xlabel=r"Infrared output-mirror reflectivity $R_2$",
           ylabel=r"Normalized output intensity $I_{out}/I_s$",
           title="Nonlinear output-coupler optimization")
    ax.grid(alpha=.23); ax.legend()
    fig.savefig(output / "Figure_output_coupler_optimization.png", dpi=240)

    # Cavity family
    family = cavity_family(parameters)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), constrained_layout=True)
    axes[0].plot(100 * family["x"], 100 * family["radius"], lw=2)
    axes[0].axvline(100 * cavity["d3"], color="gray", ls="--")
    axes[0].set(xlabel=r"$d_3$ (cm)", ylabel="Mirror radius (cm)",
                title="Equal mirror curvature satisfying the symmetric-cavity design")
    axes[1].plot(100 * family["x"], 1e3 * family["waist"], lw=2, color="#a53b32")
    axes[1].axvline(100 * cavity["d3"], color="gray", ls="--")
    axes[1].set(xlabel=r"$d_3$ (cm)", ylabel="Minimum waist (mm)",
                title="Aperture-limited cavity waist at the doubler")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for a in axes: a.grid(alpha=.23)
    fig.savefig(output / "Figure_cavity_geometry.png", dpi=240)

    # Matching surfaces
    D1, L1 = np.meshgrid(100 * design["d1_grid"], 100 * design["L1_grid"])
    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(D1, L1, 1e6 * design["mismatch_surface"], cmap="viridis", linewidth=0)
    ax1.set(xlabel="d1 (cm)", ylabel="L1 (cm)", zlabel=r"waist$^2$ mismatch (mm$^2$)",
            title="Pump/emission waist mismatch")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(D1, L1, design["invR_surface"], cmap="coolwarm", linewidth=0)
    ax2.set(xlabel="d1 (cm)", ylabel="L1 (cm)", zlabel=r"Pump $1/R$ (m$^{-1}$)",
            title="Pump wavefront curvature at crystal")
    fig.savefig(output / "Figure_pump_matching_surfaces.png", dpi=220)

    # Optimized lengths
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(100 * design["d1"], 100 * design["L1"], lw=2.3)
    axes[0].set(xlabel="d1 (cm)", ylabel="optimized L1 (cm)", title="Best L1 for each d1")
    total = (design["L1"] + design["L2"] + design["d1"] + design["d2"]
             + parameters.gain_length + cavity["d3"])
    for values, label in [(design["L1"], "L1"), (design["L2"], "L2"),
                          (design["d1"], "d1"), (design["d2"], "d2")]:
        axes[1].plot(100 * design["d1"], 100 * values, label=label)
    axes[1].plot(100 * design["d1"], 100 * total, "k--", label="total")
    axes[1].set(xlabel="d1 (cm)", ylabel="length (cm)", title="Longitudinal design family")
    axes[1].legend(ncol=2)
    for a in axes: a.grid(alpha=.23)
    fig.savefig(output / "Figure_optimized_lengths.png", dpi=240)

    # Waist comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    xcm = 100 * design["d1"]
    axes[0].plot(xcm, 1e3 * np.full_like(xcm, parameters.diode_waist), label="diode")
    axes[0].plot(xcm, 1e3 * design["pump_waist_lens"], "--", label="lens")
    axes[0].plot(xcm, 1e3 * design["pump_waist_mirror"], ":", label="M1")
    axes[1].plot(xcm, 1e3 * design["pump_waist_crystal"], label="pump")
    axes[1].plot(xcm, 1e3 * design["emission_waist_crystal"], "--", label="emission")
    axes[2].plot(xcm, 1e3 * design["emission_waist_m1"], label="M1")
    axes[2].plot(xcm, 1e3 * design["emission_waist_m2"], "--", label="M2")
    # The optimized curves agree at sub-nanometre scale.  Use a physically
    # meaningful millimetre-scale window instead of letting automatic limits
    # magnify harmless optimizer tolerance into apparent oscillations.
    match_center = 1e3 * cavity["cavity_waist"]
    axes[1].set_ylim(0.95 * match_center, 1.05 * match_center)
    axes[2].set_ylim(0.95 * match_center, 1.05 * match_center)
    titles = ["Pump beam along coupling optics", "Waist matching at gain crystal",
              "Equal emission size at mirrors"]
    for ax, title in zip(axes, titles):
        ax.set(xlabel="d1 (cm)", ylabel="beam radius (mm)", title=title)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=.23); ax.legend(fontsize=8)
    fig.savefig(output / "Figure_beam_waist_comparison.png", dpi=240)

    # Summary table
    rows = [
        ["L1", f"{100*summary['L1_m']:.5f} cm"], ["L2", f"{100*summary['L2_m']:.5f} cm"],
        ["d1", f"{100*summary['d1_m']:.5f} cm"], ["Lg", f"{100*summary['Lg_m']:.5f} cm"],
        ["d2", f"{100*summary['d2_m']:.5f} cm"], ["d3", f"{100*summary['d3_m']:.5f} cm"],
        ["total", f"{100*summary['total_length_m']:.5f} cm"],
        ["mirror radius", f"{100*summary['mirror_radius_m']:.5f} cm"],
        ["cavity waist", f"{1e3*summary['cavity_waist_m']:.5f} mm"],
        ["R2 optimum", f"{summary['R2_opt']:.6f}"],
        ["gain gamma", f"{summary['gain_parameter']:.6f}"],
        ["N0", f"{summary['N0_per_m3']:.5e} m^-3"],
        ["N2", f"{summary['N2_per_m3']:.5e} m^-3"],
    ]
    fig, ax = plt.subplots(figsize=(8.5, 7.4), constrained_layout=True); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Quantity", "Representative value"],
                     cellLoc="center", loc="center", colWidths=[.35, .45])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.45)
    ax.set_title("laser-design summary", fontsize=16, pad=12)
    fig.savefig(output / "Table_design_summary.png", dpi=230)


def numerical_checks(parameters: LaserParameters, cavity: dict[str, float]) -> None:
    for matrix in [free_space(.01), thin_lens(.02), curved_reflection(.05)]:
        assert np.isclose(np.linalg.det(matrix), 1.0)
    family = cavity_family(parameters)
    assert np.all((family["stability"] >= -1e-12) & (family["stability"] <= 1 + 1e-12))
    assert cavity["d3"] > 0 and cavity["mirror_radius"] > 0
    q = q_from_waist(parameters.diode_waist, parameters.lambda_pump)
    w, invR = waist_and_inverse_radius(q, parameters.lambda_pump)
    assert np.isclose(w, parameters.diode_waist) and abs(invR) < 1e-12


def run_design(parameters: LaserParameters, output: Path, n_d1: int = 90,
               n_L1: int = 130, show: bool = False) -> dict[str, float]:
    """Run the complete calculation; shared by the GUI and command line."""
    output.mkdir(parents=True, exist_ok=True)
    cavity = choose_cavity(parameters)
    numerical_checks(parameters, cavity)
    design = optimize_longitudinal_design(parameters, cavity, n_d1, n_L1)
    output_model = optimize_output_coupler(parameters)
    summary = representative_design(parameters, cavity, design, output_model)
    export_results(output, parameters, cavity, design, summary)
    plot_results(output, parameters, cavity, design, output_model, summary)
    plot_final_setup(output, parameters, cavity, summary)
    if show:
        plt.show()
    else:
        plt.close("all")
    return summary


def launch_gui() -> None:
    """Open a Tkinter interface for parameter editing, calculation, and previews."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    class LaserDesignGUI(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Nd:GdVO₄ Laser Design")
            self.geometry("1280x790")
            self.minsize(1050, 680)
            style = ttk.Style(self)
            if "clam" in style.theme_names():
                style.theme_use("clam")
            self.variables: dict[str, tk.StringVar] = {}
            self.preview_figure = None
            self.preview_canvas = None
            self.preview_toolbar = None
            self._build()

        def _entry(self, parent, row: int, label: str, key: str,
                   default: str, unit: str) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=default)
            self.variables[key] = variable
            ttk.Entry(parent, textvariable=variable, width=12).grid(
                row=row, column=1, sticky="ew", padx=(8, 5), pady=4)
            ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", pady=4)

        def _build(self) -> None:
            header = ttk.Frame(self, padding=(12, 9))
            header.pack(fill="x")
            ttk.Label(header, text="Nd:GdVO₄ Diode-Pumped Laser Design",
                      font=("TkDefaultFont", 16, "bold")).pack(side="left")
            ttk.Label(header, text="Gaussian cavity • ABCD pump matching • nonlinear R₂ optimization",
                      foreground="#4d5966").pack(side="left", padx=20)

            body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
            body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            left = ttk.Frame(body, padding=10); right = ttk.Frame(body)
            body.add(left, weight=0); body.add(right, weight=1)

            inputs = ttk.LabelFrame(left, text="Design inputs", padding=10)
            inputs.pack(fill="x")
            fields = [
                ("Pump power", "pump_power", "1.0", "W"),
                ("Pump transmission", "pump_transmission", "0.95", "0–1"),
                ("Diode waist", "diode_waist", "467", "µm"),
                ("Gain-crystal length", "gain_length", "2.7", "mm"),
                ("Gain refractive index", "gain_index", "2.0", ""),
                ("Aperture radius", "aperture_radius", "5.0", "mm"),
                ("Maximum system length", "max_length", "10.0", "cm"),
                ("Target absorption", "target_absorption", "99.99", "%"),
                ("d₁ grid samples", "n_d1", "90", ""),
                ("L₁ grid samples", "n_L1", "130", ""),
            ]
            for row, field in enumerate(fields): self._entry(inputs, row, *field)

            folder = ttk.LabelFrame(left, text="Output", padding=10)
            folder.pack(fill="x", pady=(10, 0))
            default_output = Path(__file__).resolve().parent / "results_laser_design"
            self.output_var = tk.StringVar(value=str(default_output))
            ttk.Entry(folder, textvariable=self.output_var, width=35).pack(fill="x")
            ttk.Button(folder, text="Choose folder...", command=self.choose_folder).pack(fill="x", pady=(6, 0))
            ttk.Button(left, text="RUN COMPLETE DESIGN", command=self.calculate).pack(fill="x", pady=(14, 5), ipady=7)
            ttk.Button(left, text="Reset defaults", command=self.reset).pack(fill="x")
            self.status = tk.StringVar(value="Ready — edit inputs and run the design.")
            ttk.Label(left, textvariable=self.status, wraplength=270,
                      foreground="#265f85").pack(fill="x", pady=(12, 0))

            tabs = ttk.Notebook(right); tabs.pack(fill="both", expand=True)
            summary_tab = ttk.Frame(tabs, padding=8)
            preview_tab = ttk.Frame(tabs, padding=8)
            theory_tab = ttk.Frame(tabs, padding=12)
            tabs.add(summary_tab, text="Numerical summary")
            tabs.add(preview_tab, text="Plot preview")
            tabs.add(theory_tab, text="Model guide")

            self.tree = ttk.Treeview(summary_tab, columns=("quantity", "value", "unit"), show="headings")
            for col, title, width in [("quantity", "Quantity", 310), ("value", "Value", 180), ("unit", "Unit", 120)]:
                self.tree.heading(col, text=title); self.tree.column(col, width=width, anchor="center")
            scroll = ttk.Scrollbar(summary_tab, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=scroll.set)
            self.tree.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")

            selector = ttk.Frame(preview_tab); selector.pack(fill="x", pady=(0, 6))
            ttk.Label(selector, text="Figure:").pack(side="left")
            self.figure_name = tk.StringVar(value="Figure_final_laser_setup.png")
            names = ["Figure_final_laser_setup.png", "Figure_output_coupler_optimization.png",
                     "Figure_cavity_geometry.png", "Figure_pump_matching_surfaces.png",
                     "Figure_optimized_lengths.png", "Figure_beam_waist_comparison.png",
                     "Table_design_summary.png"]
            combo = ttk.Combobox(selector, textvariable=self.figure_name, values=names,
                                 state="readonly", width=45)
            combo.pack(side="left", padx=8); combo.bind("<<ComboboxSelected>>", lambda _e: self.preview())
            ttk.Button(selector, text="Save preview as...", command=self.save_preview).pack(side="right")
            self.preview_host = ttk.Frame(preview_tab); self.preview_host.pack(fill="both", expand=True)

            guide = (
                "WORKFLOW\n\n1. Choose the pump, crystal, aperture, and package parameters.\n"
                "2. Run Complete Design. The program calculates the stable symmetric cavity, "
                "matches the 808.5-nm pump beam to the 1063.8-nm emission mode, and optimizes R₂.\n"
                "3. Inspect the numerical table and select any generated figure in Plot Preview.\n\n"
                "KEY EQUATIONS\n\n"
                "Absorption:  Lg,min = −ln(1−p)/(σa ΔN)\n"
                "Stability:  0 ≤ g₁g₂ ≤ 1, with gᵢ = 1−d/rᵢ\n"
                "ABCD propagation:  qout = (Aqin+B)/(Cqin+D)\n"
                "Pump rate:  Wp = σa Ip/(hνp)\n"
                "Nonlinear reflector:  R₂,eff = R₂ sech²(√i)\n\n"
                "Units shown in the input panel are converted internally to SI. All CSV output values use SI."
            )
            ttk.Label(theory_tab, text=guide, justify="left", wraplength=800,
                      font=("TkDefaultFont", 11)).pack(anchor="nw")

        def choose_folder(self) -> None:
            chosen = filedialog.askdirectory(initialdir=self.output_var.get())
            if chosen: self.output_var.set(chosen)

        def reset(self) -> None:
            defaults = {"pump_power":"1.0", "pump_transmission":"0.95", "diode_waist":"467",
                        "gain_length":"2.7", "gain_index":"2.0", "aperture_radius":"5.0",
                        "max_length":"10.0", "target_absorption":"99.99", "n_d1":"90", "n_L1":"130"}
            for key, value in defaults.items(): self.variables[key].set(value)
            self.status.set("Default project values restored.")

        def parameters(self) -> tuple[LaserParameters, int, int]:
            v = {key: float(var.get()) for key, var in self.variables.items() if key not in ("n_d1", "n_L1")}
            n_d1, n_L1 = int(self.variables["n_d1"].get()), int(self.variables["n_L1"].get())
            if v["pump_power"] <= 0 or v["diode_waist"] <= 0 or v["gain_length"] <= 0: raise ValueError("Power, waist, and gain length must be positive.")
            if not 0 < v["pump_transmission"] <= 1: raise ValueError("Pump transmission must be in (0, 1].")
            if not 0 < v["target_absorption"] < 100: raise ValueError("Target absorption must be between 0 and 100%.")
            if n_d1 < 10 or n_L1 < 10: raise ValueError("Each numerical grid must contain at least 10 samples.")
            p = LaserParameters(pump_power=v["pump_power"], pump_transmission=v["pump_transmission"],
                diode_waist=v["diode_waist"]*1e-6, gain_length=v["gain_length"]*1e-3,
                gain_index=v["gain_index"], aperture_radius=v["aperture_radius"]*1e-3,
                max_total_length=v["max_length"]*.01, target_absorption=v["target_absorption"]*.01)
            return p, n_d1, n_L1

        def calculate(self) -> None:
            try:
                parameters, n_d1, n_L1 = self.parameters()
                output = Path(self.output_var.get()).expanduser().resolve()
                self.status.set("Calculating cavity, pump matching, and output coupling…")
                self.update_idletasks()
                summary = run_design(parameters, output, n_d1, n_L1, show=False)
                self.show_summary(summary); self.preview()
                self.status.set(f"Complete. Results saved in {output}")
            except Exception as exc:
                self.status.set("Calculation stopped — check the highlighted values.")
                messagebox.showerror("Design calculation error", str(exc))

        def show_summary(self, s: dict[str, float]) -> None:
            for item in self.tree.get_children(): self.tree.delete(item)
            rows = [
                ("L₁", 100*s["L1_m"], "cm"), ("L₂", 100*s["L2_m"], "cm"),
                ("d₁", 100*s["d1_m"], "cm"), ("gain length Lg", 100*s["Lg_m"], "cm"),
                ("d₂", 100*s["d2_m"], "cm"), ("d₃", 100*s["d3_m"], "cm"),
                ("total physical length", 100*s["total_length_m"], "cm"),
                ("equal mirror radius", 100*s["mirror_radius_m"], "cm"),
                ("cavity waist", 1e3*s["cavity_waist_m"], "mm"),
                ("pump waist at crystal", 1e3*s["pump_waist_crystal_m"], "mm"),
                ("emission waist at crystal", 1e3*s["emission_waist_crystal_m"], "mm"),
                ("optimum R₂", s["R2_opt"], ""),
                ("normalized output intensity", s["normalized_output_intensity"], "I/Is"),
                ("gain parameter γ", s["gain_parameter"], ""),
                ("absorption-length estimate", 1e3*s["absorption_length_estimate_m"], "mm"),
                ("N₀", s["N0_per_m3"], "m⁻³"), ("N₂", s["N2_per_m3"], "m⁻³")]
            for name, value, unit in rows:
                formatted = f"{value:.6g}" if abs(value) < 1e5 else f"{value:.5e}"
                self.tree.insert("", "end", values=(name, formatted, unit))

        def preview(self) -> None:
            image_path = Path(self.output_var.get()) / self.figure_name.get()
            if not image_path.exists(): return
            if self.preview_canvas: self.preview_canvas.get_tk_widget().destroy()
            if self.preview_toolbar: self.preview_toolbar.destroy()
            image = plt.imread(image_path)
            self.preview_figure = plt.Figure(figsize=(9, 5.8), dpi=100, constrained_layout=True)
            ax = self.preview_figure.add_subplot(111); ax.imshow(image); ax.axis("off")
            self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=self.preview_host)
            self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.preview_toolbar = NavigationToolbar2Tk(self.preview_canvas, self.preview_host)
            self.preview_toolbar.update(); self.preview_canvas.draw_idle()

        def save_preview(self) -> None:
            if self.preview_figure is None: return
            path = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
            if path: self.preview_figure.savefig(path, dpi=300, bbox_inches="tight")

    LaserDesignGUI().mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pump-power", type=float, default=1.0, help="diode pump power in W")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--surface-d1", type=int, default=90)
    parser.add_argument("--surface-L1", type=int, default=130)
    parser.add_argument("--gui", action="store_true", help="open the interactive desktop GUI")
    default_output = Path(__file__).resolve().parent / "results_240A_laser_design"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    if args.gui:
        launch_gui(); return
    if args.pump_power <= 0: parser.error("--pump-power must be positive")

    parameters = LaserParameters(pump_power=args.pump_power)
    summary = run_design(parameters, args.output, args.surface_d1, args.surface_L1,
                         show=not args.no_show)

    print("All ABCD, stability, and beam-parameter checks passed.")
    print(f"Small-signal 99.99% absorption length: {1e3*parameters.absorption_length_unsaturated:.4f} mm")
    print(f"Selected gain length: {1e3*parameters.gain_length:.4f} mm")
    print(f"Normalized gain gamma: {parameters.gain_parameter:.6f}")
    print(f"Optimum R2: {summary['R2_opt']:.6f}")
    print("Representative design (cm):")
    for key in ["L1_m", "L2_m", "d1_m", "Lg_m", "d2_m", "d3_m", "total_length_m", "mirror_radius_m"]:
        print(f"  {key[:-2]:15s} = {100*summary[key]:.8f}")
    print(f"Results saved in: {args.output.resolve()}")


if __name__ == "__main__":
    # Double-clicking the file or running it without arguments opens the GUI.
    # Supplying command-line options preserves the original batch workflow.
    if len(sys.argv) == 1:
        launch_gui()
    else:
        main()
