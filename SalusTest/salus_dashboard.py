#!/usr/bin/env python3
"""
SALUS Enhanced Dashboard - See Everything SALUS Sees and Thinks

Shows:
- Camera image input (what VLA sees)
- Signal extraction process (how SALUS analyzes it)
- VLA internal processing
- Risk computation in real-time
- Alert decision making
"""

import sys
import time
import threading
import queue
from collections import deque
from pathlib import Path
import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from salus_live_demo import SALUSWrapper, MockVLA
from salus_state_machine import AlertState

class EnhancedDashboard:
    """
    Enhanced dashboard showing EVERYTHING SALUS sees and thinks.
    """

    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Setup complex layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="vision", size=12),
            Layout(name="processing", ratio=1),
            Layout(name="footer", size=10)
        )

        self.layout["vision"].split_row(
            Layout(name="camera_view"),
            Layout(name="vla_processing")
        )

        self.layout["processing"].split_row(
            Layout(name="signal_extraction", ratio=1),
            Layout(name="risk_computation", ratio=1),
            Layout(name="alert_decision", ratio=1)
        )

        # Data queues
        self.camera_queue = queue.Queue(maxsize=10)
        self.vla_queue = queue.Queue(maxsize=10)
        self.signal_queue = queue.Queue(maxsize=10)
        self.risk_queue = queue.Queue(maxsize=10)
        self.alert_queue = queue.Queue(maxsize=10)
        self.stats_queue = queue.Queue(maxsize=10)

        # Latest data
        self.latest_camera = None
        self.latest_vla = None
        self.latest_signals = None
        self.latest_risk = None
        self.latest_alert = None
        self.latest_stats = None

    def make_header(self):
        """Header with timestamp"""
        text = Text()
        text.append("SALUS ", style="bold cyan")
        text.append("Enhanced Dashboard", style="bold white")
        text.append(" - See What SALUS Sees & Thinks", style="dim cyan")
        text.append(f"  |  {datetime.now().strftime('%H:%M:%S.%f')[:-3]}", style="dim")
        return Panel(text, style="cyan", box=box.HEAVY)

    def make_camera_view(self):
        """Show what the camera/VLA sees"""
        if self.latest_camera is None:
            return Panel("Waiting for camera input...", title="[cyan]Camera View")

        # ASCII art representation of camera image
        img_shape = self.latest_camera.get('shape', (224, 224, 3))
        img_mean = self.latest_camera.get('mean', 0.0)
        img_std = self.latest_camera.get('std', 0.0)

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Info", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Input Shape", f"{img_shape}")
        table.add_row("Mean Intensity", f"{img_mean:.3f}")
        table.add_row("Std Dev", f"{img_std:.3f}")
        table.add_row("", "")

        # Mock image visualization
        scene_desc = self.latest_camera.get('scene', 'Unknown')
        table.add_row("Scene", scene_desc)

        # ASCII art camera view
        ascii_img = self.generate_ascii_image(img_mean, img_std)

        content = Columns([table, Panel(ascii_img, title="Image", border_style="dim")])

        return Panel(content, title="[cyan bold]📷 Camera View (What VLA Sees)", border_style="cyan")

    def generate_ascii_image(self, mean, std):
        """Generate ASCII art representation of image"""
        # Simple ASCII art based on intensity
        chars = " .:-=+*#%@"
        intensity = int((mean + std) * 5)
        intensity = max(0, min(len(chars)-1, intensity))

        art = []
        art.append("┌────────────────┐")

        # Pattern depends on current state
        if std > 1.5:  # High variance = cluttered scene
            for i in range(6):
                line = "│"
                for j in range(16):
                    idx = int((i + j + intensity) / 2) % len(chars)
                    line += chars[idx]
                line += "│"
                art.append(line)
        else:  # Low variance = simple scene
            for i in range(6):
                line = "│" + chars[intensity] * 16 + "│"
                art.append(line)

        art.append("└────────────────┘")
        return "\n".join(art)

    def make_vla_processing(self):
        """Show VLA internal processing"""
        if self.latest_vla is None:
            return Panel("Waiting for VLA...", title="[yellow]VLA Processing")

        table = Table(title="[bold yellow]VLA Internal State", box=box.SIMPLE)
        table.add_column("Layer", style="yellow", width=15)
        table.add_column("Status", width=35)

        vla_data = self.latest_vla

        # Show processing pipeline
        table.add_row(
            "Vision Encoder",
            Text("✓ Processing image", style="green") if vla_data.get('vision_active') else Text("○ Idle", style="dim")
        )

        table.add_row(
            "Language Model",
            Text(f"✓ '{vla_data.get('language', 'unknown')}'", style="green") if vla_data.get('language_active') else Text("○ No input", style="dim")
        )

        table.add_row(
            "Transformer",
            Text(f"✓ {vla_data.get('transformer_layers', 0)} layers", style="green")
        )

        table.add_row(
            "Action Head",
            Text(f"✓ Output: {vla_data.get('action_dim', 0)}D", style="green")
        )

        # Show predicted action
        action = vla_data.get('action', np.zeros(7))
        action_str = " ".join([f"{a:+.2f}" for a in action[:7]])

        action_panel = Panel(
            Text(f"[{action_str}]", style="bold yellow"),
            title="Predicted Action (7-DoF)",
            border_style="yellow"
        )

        # Confidence
        confidence = vla_data.get('confidence', 0.0)
        conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        conf_style = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"

        conf_panel = Panel(
            Text(conf_bar, style=conf_style) + Text(f"  {confidence:.1%}", style="bold"),
            title="VLA Confidence",
            border_style="yellow"
        )

        content = Columns([table, Columns([action_panel, conf_panel], expand=True)])

        return Panel(content, title="[yellow bold]🤖 VLA Processing", border_style="yellow")

    def make_signal_extraction(self):
        """Show signal extraction process"""
        if self.latest_signals is None:
            return Panel("Extracting signals...", title="[green]Signal Extraction")

        signals = self.latest_signals['signals']
        extraction_method = self.latest_signals.get('method', 'unknown')

        table = Table(title="[bold green]12D Signal Extraction", box=box.SIMPLE, show_header=True)
        table.add_column("Signal", style="green", width=18)
        table.add_column("Value", justify="right", width=8)
        table.add_column("Source", width=20, style="dim")

        signal_info = [
            ("z₁ Volatility", signals[0], "Action history"),
            ("z₂ Magnitude", signals[1], "Current action"),
            ("z₃ Acceleration", signals[2], "Action derivative"),
            ("z₄ Divergence", signals[3], "Planned vs actual"),
            ("z₅ Hidden Norm", signals[4], "VLA internals"),
            ("z₆ Hidden Std", signals[5], "VLA internals"),
            ("z₇ Hidden Skew", signals[6], "VLA internals"),
            ("z₈ Entropy", signals[7], "Action probs"),
            ("z₉ Max Prob", signals[8], "Action probs"),
            ("z₁₀ Norm Violation", signals[9], "Physics check"),
            ("z₁₁ Force Anomaly", signals[10], "Force sensor"),
            ("z₁₂ Consistency", signals[11], "Temporal"),
        ]

        for name, val, source in signal_info:
            # Color based on value
            if val > 1.5:
                val_style = "red bold"
            elif val > 0.8:
                val_style = "yellow"
            else:
                val_style = "green"

            table.add_row(name, Text(f"{val:.3f}", style=val_style), source)

        # Show extraction method
        method_text = Text("\nExtraction: ", style="dim")
        if extraction_method == 'full':
            method_text.append("Full 12D", style="green bold")
        else:
            method_text.append("Minimal 6D", style="yellow bold")
            method_text.append(" (no VLA internals)", style="dim")

        return Panel(
            Columns([table, Panel(method_text, border_style="dim")]),
            title="[green bold]📊 Signal Extraction Process",
            border_style="green"
        )

    def make_risk_computation(self):
        """Show risk computation process"""
        if self.latest_risk is None:
            return Panel("Computing risk...", title="[magenta]Risk Analysis")

        risk_data = self.latest_risk

        table = Table(title="[bold magenta]Temporal Window Analysis", box=box.SIMPLE)
        table.add_column("Stage", style="magenta")
        table.add_column("Detail", width=30)

        # Window info
        window_size = risk_data.get('window_size', 20)
        window_filled = risk_data.get('window_filled', False)

        table.add_row(
            "Input Window",
            f"{'✓' if window_filled else '○'} {window_size} timesteps (667ms)"
        )

        # Conv1D processing
        table.add_row(
            "Conv1D Layers",
            "✓ Extract local patterns (k=5,3,3)"
        )

        # GRU processing
        table.add_row(
            "BiGRU Layers",
            "✓ Temporal dependencies (h=128)"
        )

        # Multi-horizon predictions
        table.add_row(
            "Horizon Heads",
            "✓ 300ms, 500ms, 1000ms forecasts"
        )

        # Show risk scores
        risks = risk_data.get('scores', {'300ms': 0, '500ms': 0, '1000ms': 0})

        risk_viz = []
        for horizon, risk in risks.items():
            bar_len = int(risk * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)

            if risk > 0.7:
                style = "red bold"
            elif risk > 0.4:
                style = "yellow"
            else:
                style = "green"

            risk_viz.append(f"{horizon:>6}: {Text(bar, style=style)} {risk:.3f}")

        risk_panel = Panel(
            "\n".join([str(x) for x in risk_viz]),
            title="Multi-Horizon Predictions",
            border_style="magenta"
        )

        return Panel(
            Columns([table, risk_panel]),
            title="[magenta bold]🧠 Risk Computation (Conv1D+GRU)",
            border_style="magenta"
        )

    def make_alert_decision(self):
        """Show alert state machine decision"""
        if self.latest_alert is None:
            return Panel("Processing...", title="[red]Alert Decision")

        alert_data = self.latest_alert
        state = alert_data.get('state', AlertState.NORMAL)
        ema_risk = alert_data.get('ema_risk', 0.0)

        table = Table(title="[bold red]State Machine Logic", box=box.SIMPLE)
        table.add_column("Check", style="red", width=20)
        table.add_column("Status", width=25)

        # EMA smoothing
        smoothed = alert_data.get('smoothed', False)
        table.add_row(
            "1. EMA Smoothing",
            Text(f"✓ α=0.3 → {ema_risk:.3f}", style="green") if smoothed else Text("○ Waiting", style="dim")
        )

        # Persistence
        persistent = alert_data.get('persistent', False)
        persist_count = alert_data.get('persist_count', 0)
        table.add_row(
            "2. Persistence",
            Text(f"✓ {persist_count}/4 ticks", style="green") if persistent else Text(f"○ {persist_count}/4 ticks", style="yellow")
        )

        # Hysteresis
        above_on = ema_risk > 0.40
        above_off = ema_risk > 0.35
        table.add_row(
            "3. Hysteresis",
            Text(f"✓ Above τ_on (0.40)", style="red" if above_on else "yellow") if above_off else Text("✓ Below τ_off (0.35)", style="green")
        )

        # Cooldown
        in_cooldown = alert_data.get('cooldown', False)
        table.add_row(
            "4. Cooldown",
            Text("⏳ Active (2s)", style="yellow") if in_cooldown else Text("✓ Ready", style="green")
        )

        # Final decision
        if state == AlertState.CRITICAL:
            decision = Text("🚨 CRITICAL ALERT", style="red bold blink")
            action = Text("→ INTERVENTION TRIGGERED\n   Slow mode: action * 0.5", style="red bold")
        elif state == AlertState.WARNING:
            decision = Text("⚠ WARNING", style="yellow bold")
            action = Text("→ Monitoring closely", style="yellow")
        else:
            decision = Text("✓ NORMAL", style="green bold")
            action = Text("→ Continue normal operation", style="green")

        decision_panel = Panel(
            decision + Text("\n\n") + action,
            title="Alert State",
            border_style="red" if state == AlertState.CRITICAL else "yellow" if state == AlertState.WARNING else "green"
        )

        return Panel(
            Columns([table, decision_panel]),
            title="[red bold]🚨 Alert State Machine Decision",
            border_style="red"
        )

    def make_footer(self):
        """System statistics"""
        if self.latest_stats is None:
            stats = {}
        else:
            stats = self.latest_stats

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")

        table.add_row(
            "Step", f"{stats.get('step', 0)}",
            "Episode", f"{stats.get('episode', 0)}",
            "FPS", f"{stats.get('fps', 0.0):.1f}"
        )
        table.add_row(
            "Alerts", f"{stats.get('alerts', 0)}",
            "Interventions", f"{stats.get('interventions', 0)}",
            "Success Rate", f"{stats.get('success_rate', 0.0):.1%}"
        )
        table.add_row(
            "Latency", f"{stats.get('latency_ms', 0.0):.1f}ms",
            "Failures", f"{stats.get('failures', 0)}",
            "Memory", f"{stats.get('memory_mb', 0):.0f}MB"
        )

        return Panel(table, title="[bold]System Performance", border_style="cyan")

    def update_display(self):
        """Update all panels"""
        # Get latest data from queues
        try:
            while not self.camera_queue.empty():
                self.latest_camera = self.camera_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.vla_queue.empty():
                self.latest_vla = self.vla_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.signal_queue.empty():
                self.latest_signals = self.signal_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.risk_queue.empty():
                self.latest_risk = self.risk_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.alert_queue.empty():
                self.latest_alert = self.alert_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.stats_queue.empty():
                self.latest_stats = self.stats_queue.get_nowait()
        except queue.Empty:
            pass

        # Update layout
        self.layout["header"].update(self.make_header())
        self.layout["camera_view"].update(self.make_camera_view())
        self.layout["vla_processing"].update(self.make_vla_processing())
        self.layout["signal_extraction"].update(self.make_signal_extraction())
        self.layout["risk_computation"].update(self.make_risk_computation())
        self.layout["alert_decision"].update(self.make_alert_decision())
        self.layout["footer"].update(self.make_footer())

        return self.layout

    def run(self):
        """Run the dashboard"""
        with Live(self.layout, console=self.console, refresh_per_second=10, screen=True) as live:
            while True:
                live.update(self.update_display())
                time.sleep(0.1)

def run_enhanced_demo(model_path='salus_fixed_pipeline.pt'):
    """
    Run enhanced demo with full visibility into SALUS.
    """
    console = Console()

    console.print("\n[bold cyan]╔════════════════════════════════════════════════════╗")
    console.print("[bold cyan]║  SALUS Enhanced Dashboard - See Everything!       ║")
    console.print("[bold cyan]╚════════════════════════════════════════════════════╝\n")

    # Initialize VLA
    console.print("[yellow]Initializing VLA model...[/yellow]")
    vla = MockVLA()

    # Wrap with SALUS
    console.print("[yellow]Loading SALUS wrapper...[/yellow]")
    salus = SALUSWrapper(vla, model_path=model_path)

    # Initialize dashboard
    console.print("[yellow]Starting enhanced dashboard...[/yellow]")
    dashboard = EnhancedDashboard()

    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=dashboard.run, daemon=True)
    dashboard_thread.start()

    console.print("[green]✓ Dashboard ready![/green]\n")
    time.sleep(2)

    # Main control loop
    step = 0
    episode = 0
    episode_steps = 0
    start_time = time.time()
    last_fps_time = start_time

    try:
        while True:
            step_start = time.time()

            # Generate mock camera observation
            observation = torch.randn(1, 3, 224, 224)
            language = "pick up the red mug"

            # Camera data
            dashboard.camera_queue.put({
                'shape': (1, 3, 224, 224),
                'mean': observation.mean().item(),
                'std': observation.std().item(),
                'scene': f"Cluttered table scene (step {step})"
            })

            # VLA processing
            action_raw = vla(observation, language)
            action_np = action_raw.cpu().numpy() if torch.is_tensor(action_raw) else action_raw

            dashboard.vla_queue.put({
                'vision_active': True,
                'language_active': True,
                'language': language,
                'transformer_layers': 12,
                'action_dim': 7,
                'action': action_np,
                'confidence': 0.95 if step < 50 else max(0.3, 0.95 - (step - 50) / 100.0)
            })

            # SALUS prediction
            action, signals, risk_scores, alert_result = salus.predict(
                observation, language
            )

            # Signal extraction
            dashboard.signal_queue.put({
                'signals': signals,
                'method': 'minimal' if salus.hidden_states is None else 'full'
            })

            # Risk computation
            dashboard.risk_queue.put({
                'window_size': salus.window_size,
                'window_filled': len(salus.signal_history) >= salus.window_size,
                'scores': risk_scores
            })

            # Alert decision
            dashboard.alert_queue.put({
                'state': alert_result['state'],
                'ema_risk': alert_result.get('risk_ema', 0.0),
                'smoothed': True,
                'persistent': alert_result.get('persistence_met', False),
                'persist_count': alert_result.get('persistence_count', 0),
                'cooldown': alert_result.get('in_cooldown', False)
            })

            # Check intervention
            if salus.should_intervene(alert_result):
                action = salus.apply_intervention(action, 'slow_mode')

            # Update stats
            step += 1
            episode_steps += 1

            # FPS calculation
            if step % 10 == 0:
                current_time = time.time()
                fps = 10.0 / (current_time - last_fps_time)
                last_fps_time = current_time
            else:
                fps = 10.0

            # Latency
            latency_ms = (time.time() - step_start) * 1000

            # Episode management
            if episode_steps >= 100:
                episode += 1
                episode_steps = 0

            dashboard.stats_queue.put({
                'step': step,
                'episode': episode,
                'fps': fps,
                'alerts': salus.alert_count,
                'interventions': salus.intervention_count,
                'failures': episode // 5,  # Mock
                'success_rate': 0.8,  # Mock
                'latency_ms': latency_ms,
                'memory_mb': 1200  # Mock
            })

            # Control loop rate
            time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SALUS Enhanced Dashboard")
    parser.add_argument('--model', type=str, default='salus_fixed_pipeline.pt',
                       help='Path to SALUS model checkpoint')

    args = parser.parse_args()

    run_enhanced_demo(model_path=args.model)
