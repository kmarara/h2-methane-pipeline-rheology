"""
Validation tests for core physics engine.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
from core import (
    peng_robinson_z, wilke_viscosity, blend_properties,
    darcy_weisbach_dp, weymouth_dp, equivalent_flow_rate,
    gas_velocity, embrittlement_tier, velocity_tier, pressure_drop_tier,
    CH4, H2, RiskTier
)


class TestPengRobinson:
    """Peng-Robinson EOS validation against known values."""

    def test_ch4_z_at_stp(self):
        """CH4 at STP should have Z ≈ 0.99 (near ideal)."""
        z = peng_robinson_z(273.15, 1.01325e5, CH4)
        assert 0.95 < z < 1.05

    def test_h2_z_at_stp(self):
        """H2 at STP should have Z ≈ 1.0 (ideal gas)."""
        z = peng_robinson_z(273.15, 1.01325e5, H2)
        assert 0.98 < z < 1.02

    def test_ch4_z_at_70bar(self):
        """CH4 at 70 bar, 283 K: Z ≈ 0.85-0.90 (GPSA Fig 23-14)."""
        z = peng_robinson_z(283.15, 70e5, CH4)
        assert 0.80 < z < 0.95

    def test_h2_z_at_70bar(self):
        """H2 at 70 bar, 283 K: Z ≈ 1.05-1.15 (above Boyle temp)."""
        z = peng_robinson_z(283.15, 70e5, H2)
        assert 1.0 < z < 1.2


class TestWilkeViscosity:
    """Wilke mixing rule validation."""

    def test_pure_ch4(self):
        """0% H2 should give pure CH4 viscosity."""
        mu = wilke_viscosity(0.0, 273.15)
        expected = 1.12e-5  # CH4 ref at 273.15 K
        assert abs(mu - expected) / expected < 0.05

    def test_pure_h2(self):
        """100% H2 should give pure H2 viscosity."""
        mu = wilke_viscosity(1.0, 273.15)
        expected = 8.92e-6  # H2 ref at 273.15 K
        assert abs(mu - expected) / expected < 0.05

    def test_wilke_behavior(self):
        """
        Wilke mixing for H2/CH4 shows non-monotonic behavior due to large MW disparity.
        Mixture viscosity peaks around 30-40% H2 then decreases.
        This is physically correct - validated against literature.
        """
        mu_0 = wilke_viscosity(0.0, 283.15)      # Pure CH4
        mu_20 = wilke_viscosity(0.2, 283.15)     # 20% H2
        mu_40 = wilke_viscosity(0.4, 283.15)     # 40% H2 (peak)
        mu_100 = wilke_viscosity(1.0, 283.15)    # Pure H2
        
        # CH4: ~11.6 μPa·s, H2: ~9.1 μPa·s at 10°C
        # Mixture peaks ~11.72 μPa·s at ~40% H2
        assert mu_20 > mu_0   # Rising
        assert mu_40 > mu_20  # Peak
        assert mu_100 < mu_0  # Pure H2 lower than CH4


class TestBlendProperties:
    """Blend property calculations."""

    def test_molecular_weight_linear(self):
        """M_blend should be linear in mole fraction."""
        for h2_frac in [0.0, 0.2, 0.5, 1.0]:
            props = blend_properties(h2_frac, 283.15, 70e5)
            expected = h2_frac * H2.molecular_weight + (1-h2_frac) * CH4.molecular_weight
            assert abs(props["M_blend"] - expected) < 0.01

    def test_density_decreases_with_h2(self):
        """Density should decrease as H2 fraction increases (at same P, T)."""
        rho_0 = blend_properties(0.0, 283.15, 70e5)["rho_blend"]
        rho_100 = blend_properties(1.0, 283.15, 70e5)["rho_blend"]
        assert rho_0 > rho_100

    def test_heating_value_mass_increases(self):
        """Mass HV should increase with H2 (141.8 vs 55.5 MJ/kg)."""
        hv_0 = blend_properties(0.0, 283.15, 70e5)["hv_mass_blend"]
        hv_100 = blend_properties(1.0, 283.15, 70e5)["hv_mass_blend"]
        assert hv_100 > hv_0
        assert abs(hv_100 - 141.8) < 1.0
        assert abs(hv_0 - 55.5) < 1.0

    def test_heating_value_vol_decreases(self):
        """Volumetric HV at STP should decrease with H2 (12.7 vs 35.8 MJ/m³)."""
        hv_0 = blend_properties(0.0, 283.15, 70e5)["hv_vol_blend"]
        hv_100 = blend_properties(1.0, 283.15, 70e5)["hv_vol_blend"]
        assert hv_0 > hv_100
        assert abs(hv_0 - 35.8) < 1.0
        assert abs(hv_100 - 12.7) < 1.0


class TestHydraulics:
    """Pressure drop and flow calculations."""

    def test_darcy_weisbach_positive(self):
        """ΔP should be positive for forward flow."""
        dp = darcy_weisbach_dp(1.0, 100, 500, 50, 1.2e-5)
        assert dp > 0

    def test_weymouth_positive(self):
        """Weymouth ΔP should be positive."""
        dp = weymouth_dp(1.0, 100, 500, 50, 1.2e-5, 70e5, 0.9, 283.15)
        assert dp > 0

    def test_equivalent_flow_scaling(self):
        """Flow should scale inversely with volumetric HV."""
        Q_ref = 1.0
        hv_ref = 35.8  # CH4
        hv_blend = 25.0  # ~20% H2
        Q_eq = equivalent_flow_rate(Q_ref, hv_ref, hv_blend)
        expected = Q_ref * hv_ref / hv_blend
        assert abs(Q_eq - expected) < 1e-6

    def test_velocity_calculation(self):
        """Velocity = Q / A."""
        Q = 1.0
        D = 500  # mm
        v = gas_velocity(Q, D)
        A = np.pi * (D/1000/2)**2
        expected = Q / A
        assert abs(v - expected) < 1e-6


class TestMaterialsRisk:
    """Risk tier thresholds."""

    def test_embrittlement_tiers(self):
        assert embrittlement_tier(5.0)[0] == RiskTier.LOW
        assert embrittlement_tier(15.0)[0] == RiskTier.MODERATE
        assert embrittlement_tier(30.0)[0] == RiskTier.HIGH

    def test_velocity_tiers(self):
        assert velocity_tier(10.0)[0] == RiskTier.LOW
        assert velocity_tier(17.0)[0] == RiskTier.MODERATE
        assert velocity_tier(25.0)[0] == RiskTier.HIGH

    def test_pressure_drop_tiers(self):
        assert pressure_drop_tier(0.10)[0] == RiskTier.LOW
        assert pressure_drop_tier(0.20)[0] == RiskTier.MODERATE
        assert pressure_drop_tier(0.40)[0] == RiskTier.HIGH


class TestKnownValidationCases:
    """
    Regression tests against hand calculations or literature values.
    """

    def test_20pct_h2_70bar_283k_density(self):
        """20% H2 at 70 bar, 10°C: ρ ≈ 45-50 kg/m³ (GPSA range)."""
        props = blend_properties(0.2, 283.15, 70e5)
        assert 40 < props["rho_blend"] < 55

    def test_20pct_h2_viscosity_increase(self):
        """20% H2 increases viscosity slightly vs pure CH4 (Wilke peak effect)."""
        mu_ch4 = wilke_viscosity(0.0, 283.15)
        mu_20 = wilke_viscosity(0.2, 283.15)
        increase = (mu_20 - mu_ch4) / mu_ch4
        assert 0.005 < increase < 0.05  # ~1-5% increase at peak

    def test_flow_increase_20pct_h2(self):
        """20% H2 blend requires ~15% more volumetric flow for same energy (STP basis)."""
        hv_ch4 = 35.8
        hv_20 = blend_properties(0.2, 283.15, 70e5)["hv_vol_blend"]
        flow_inc = (hv_ch4 / hv_20 - 1) * 100
        assert 10 < flow_inc < 20  # Volumetric HV at STP is linear blend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])