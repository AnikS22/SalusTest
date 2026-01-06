#!/usr/bin/env python
"""
Minimal test to check if AppLauncher works at all
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("=" * 70, flush=True)
print("MINIMAL ISAAC LAB TEST", flush=True)
print("=" * 70, flush=True)
print(f"Args: {args}", flush=True)
print("Creating AppLauncher...", flush=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print(f"✅ SUCCESS! AppLauncher created: {simulation_app}", flush=True)
print("=" * 70, flush=True)

# Close app
simulation_app.close()
print("App closed successfully", flush=True)
