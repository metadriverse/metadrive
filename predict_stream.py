"""
CARLA Large Action Model (LAM) Stream Prediction Module.
Consumes realistically deviated or real-world CARLA telemetry data points and runs the LAM network
to edit and correct the variables, guaranteeing minimum deviance from crashless autopilot models.
Now supports full city-block routes with scenario-aware reporting.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np

from carla_schema import CarlaFrame
from carla_lam_corrector import CarlaLAMCorrector
from carla_autopilot_generator import CarlaAutopilotStreamGenerator
from carla_random_stream_generator import CarlaDevianceInductionGenerator


def detect_scenario_label(frame: CarlaFrame) -> str:
    """Returns a human-readable label for the current driving scenario."""
    abs_steer = abs(frame.control.steer)
    abs_wz = abs(frame.ego_state.angular_velocity.wz)
    speed_kmh = frame.ego_state.velocity.speed_kmh
    tl = frame.environment_context.traffic_light_state
    lead_dist = frame.environment_context.distance_to_lead_vehicle_m

    if abs_steer > 0.15 and speed_kmh < 30.0:
        return "TURN"
    if abs_steer > 0.03 or abs_wz > 0.02:
        return "CURVE"
    if tl in ("Red", "Yellow") or (speed_kmh < 5.0 and frame.control.brake > 0.1):
        return "INTRSCT"
    if 0 < lead_dist < 40.0:
        return "FOLLOW"
    return "STRGHT"


class CarlaStreamPredictor:
    """Inference & runtime execution engine for altering CARLA JSON data streams."""

    def __init__(self, model_path: str = "models/carla_lam_corrector.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. Run train_carla_lam.py first."
            )
        self.corrector = CarlaLAMCorrector(model_path=model_path)

    def process_json_file(
        self,
        input_file: str,
        output_file: Optional[str] = "carla_corrected_stream.json"
    ) -> List[Dict[str, Any]]:
        with open(input_file, "r") as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        frames = [CarlaFrame.from_dict(item) for item in raw_data]
        corrected_frames, metrics = self.corrector.alter_stream(frames)

        if output_file:
            with open(output_file, "w") as f:
                json.dump([cf.to_dict() for cf in corrected_frames], f, indent=2)
            print(f"Flawless corrected stream saved to: {output_file}")

        return metrics

    def run_live_demonstration(self, num_frames: int = 40) -> None:
        """
        Demonstrates live alteration of the full city-block route:
        straights → curves → turns → intersections → traffic lights → lead vehicles.
        """
        auto_gen = CarlaAutopilotStreamGenerator()
        dev_gen = CarlaDevianceInductionGenerator(seed=123)

        # Generate full city-block route
        reference_frames = auto_gen.generate_city_block_route()
        if len(reference_frames) > num_frames:
            # Sample evenly across the route to show all scenario types
            step = max(1, len(reference_frames) // num_frames)
            sample_indices = list(range(0, len(reference_frames), step))[:num_frames]
        else:
            sample_indices = list(range(len(reference_frames)))

        # Generate deviated sequence for entire route
        deviated_frames = dev_gen.generate_deviated_sequence(reference_frames)

        # Run LAM correction on the deviated stream
        corrected_frames, all_metrics = self.corrector.alter_stream(deviated_frames)

        print("\n" + "=" * 118)
        print(" " * 18 + "CARLA CITY-BLOCK ROUTE — LAM DEVIANCE ALTERATION & MINIMIZATION")
        print("=" * 118)
        header = (
            f"{'Frame':<6} | {'Scene':<8} | "
            f"{'Dev Steer':<9} -> {'LAM Steer':<9} (Ref)   | "
            f"{'Dev Throt':<9} -> {'LAM Throt':<9} | "
            f"{'Dev Off':<8} -> {'LAM Off':<8} | {'Status'}"
        )
        print(header)
        print("-" * 118)

        total_offset_raw = 0.0
        total_offset_corr = 0.0
        total_steer_err_raw = 0.0
        total_steer_err_corr = 0.0
        scenario_counts = {}
        count = 0

        for idx in sample_indices:
            raw_f = deviated_frames[idx]
            corr_f = corrected_frames[idx]
            ref_f = reference_frames[idx]
            m = all_metrics[idx]

            scenario = detect_scenario_label(ref_f)
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

            raw_steer = raw_f.control.steer
            corr_steer = corr_f.control.steer
            ref_steer = ref_f.control.steer
            raw_throt = raw_f.control.throttle
            corr_throt = corr_f.control.throttle
            raw_off = raw_f.environment_context.lane_offset_meters
            corr_off = corr_f.environment_context.lane_offset_meters

            total_offset_raw += abs(raw_off)
            total_offset_corr += abs(corr_off)
            total_steer_err_raw += abs(raw_steer - ref_steer)
            total_steer_err_corr += abs(corr_steer - ref_steer)
            count += 1

            status = "OPTIMAL" if abs(corr_off) < 0.05 else ("WARN" if abs(corr_off) > 0.3 else "STBLZNG")

            print(
                f"{idx:<6} | {scenario:<8} | "
                f"{raw_steer:>7.3f}   -> {corr_steer:>7.3f}  ({ref_steer:>5.2f}) | "
                f"{raw_throt:>7.3f}   -> {corr_throt:>7.3f}  | "
                f"{raw_off:>6.3f}m -> {corr_off:>6.3f}m | {status}"
            )

        print("-" * 118)
        avg_off_raw = total_offset_raw / count
        avg_off_corr = total_offset_corr / count
        avg_st_raw = total_steer_err_raw / count
        avg_st_corr = total_steer_err_corr / count

        print("DEVIANCE REDUCTION SUMMARY:")
        pct_steer = (1 - avg_st_corr / max(avg_st_raw, 1e-6)) * 100
        pct_off = (1 - avg_off_corr / max(avg_off_raw, 1e-6)) * 100
        print(f"  • Mean Steer Deviance:  {avg_st_raw:.4f}  --->  {avg_st_corr:.4f}  ({pct_steer:.1f}% reduction)")
        print(f"  • Mean Lane Offset:     {avg_off_raw:.4f}m --->  {avg_off_corr:.4f}m ({pct_off:.1f}% reduction)")
        print(f"  • Scenario Coverage:    {scenario_counts}")
        print(f"  • Total Route Frames:   {len(reference_frames)}, Sampled: {count}")
        print("=" * 118)


def main():
    parser = argparse.ArgumentParser(description="CARLA LAM Stream Prediction Module")
    parser.add_argument("--input", type=str, default=None, help="Path to input CARLA JSON file")
    parser.add_argument("--output", type=str, default="carla_corrected_stream.json", help="Output corrected JSON")
    parser.add_argument("--demo", action="store_true", help="Run live city-block demonstration")
    parser.add_argument("--frames", type=int, default=40, help="Number of sampled frames in demo")
    args = parser.parse_args()

    predictor = CarlaStreamPredictor()

    if args.input:
        print(f"Running LAM variable correction over: {args.input}")
        predictor.process_json_file(args.input, args.output)
    else:
        predictor.run_live_demonstration(num_frames=args.frames)


if __name__ == "__main__":
    main()
