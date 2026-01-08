#!/usr/bin/env python3
"""
SALUS Live Demo with Isaac Sim + Terminal GUI

Shows VLA controlling robot in Isaac Sim while displaying all internal
metrics and predictions in a rich terminal interface.

Requirements:
    pip install rich numpy torch scipy
    Isaac Sim 2023.1+ installed
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
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress
from rich.text import Text
from rich import box
from datetime import datetime

# SALUS imports
sys.path.append(str(Path(__file__).parent))
from salus.models.temporal_predictor import HybridTemporalPredictor
from salus_state_machine import SALUSStateMachine, AlertState

# ============================================================================
# SALUS Wrapper - Snaps onto ANY VLA Model
# ============================================================================

class SALUSWrapper:
    """
    Universal wrapper that snaps onto any VLA model to extract signals
    and provide failure predictions.

    Integration: Just wrap your VLA model!
        vla = YourVLAModel()
        salus = SALUSWrapper(vla, model_path='salus_fixed_pipeline.pt')
        action, signals, risk = salus.predict(observation, language)
    """

    def __init__(self, vla_model, model_path, device='cuda', window_size=20):
        self.vla = vla_model
        self.device = device
        self.window_size = window_size

        # Load SALUS predictor
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Determine architecture from checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Check output size to determine num_horizons
        if 'head.3.weight' in state_dict:
            output_size = state_dict['head.3.weight'].shape[0]
        elif 'base_model.head.3.weight' in state_dict:
            output_size = state_dict['base_model.head.3.weight'].shape[0]
        else:
            output_size = 3  # Default

        self.predictor = HybridTemporalPredictor(
            signal_dim=12,
            conv_dim=64,
            gru_dim=128,
            num_horizons=output_size,
            num_failure_types=1  # Single output per horizon
        ).to(device)

        # Handle different checkpoint formats
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Check if wrapped in UnsaturatedPredictor
        if 'base_model.conv.weight' in state_dict:
            # Remove 'base_model.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('base_model.'):
                    new_state_dict[k.replace('base_model.', '')] = v
                elif k != 'temperature':  # Skip temperature parameter
                    new_state_dict[k] = v
            state_dict = new_state_dict

        self.predictor.load_state_dict(state_dict, strict=False)
        self.predictor.eval()

        # Alert state machine
        self.state_machine = SALUSStateMachine(
            loop_rate_hz=10.0,
            ema_alpha=0.3,
            persistence_ticks=4,
            threshold_on=0.40,
            threshold_off=0.35,
            cooldown_seconds=2.0
        )

        # Signal history
        self.action_history = deque(maxlen=10)
        self.signal_history = deque(maxlen=window_size)
        self.hidden_states = None

        # Statistics
        self.step_count = 0
        self.alert_count = 0
        self.intervention_count = 0

        # Register hook to capture VLA hidden states (if accessible)
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture VLA internal states"""
        try:
            # Try to find transformer layers (works for OpenVLA, RT-2, etc.)
            if hasattr(self.vla, 'transformer'):
                if hasattr(self.vla.transformer, 'layers'):
                    final_layer = self.vla.transformer.layers[-1]
                elif hasattr(self.vla.transformer, 'h'):
                    final_layer = self.vla.transformer.h[-1]
                else:
                    return

                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        self.hidden_states = output[0].detach()
                    else:
                        self.hidden_states = output.detach()

                final_layer.register_forward_hook(hook_fn)
        except Exception as e:
            print(f"Warning: Could not register hooks: {e}")
            print("Will use minimal signal set (6D) without VLA internals")

    def extract_signals(self, action, action_probs=None):
        """
        Extract 12D signal vector from current VLA execution state.

        This is the CORE of SALUS - how we detect failure precursors.
        """
        signals = np.zeros(12, dtype=np.float32)

        # z1: Action volatility (temporal dynamics)
        if len(self.action_history) >= 5:
            recent = np.array(list(self.action_history)[-5:])
            signals[0] = np.std(recent)

        # z2: Action magnitude
        signals[1] = np.linalg.norm(action)

        # z3: Action acceleration (jerk)
        if len(self.action_history) >= 2:
            actions = np.array(list(self.action_history)[-3:] + [action])
            if len(actions) >= 3:
                signals[2] = np.linalg.norm(np.diff(actions[-3:], n=2, axis=0))

        # z4: Trajectory divergence (requires planned trajectory - set to 0 for now)
        signals[3] = 0.0

        # z5-z7: VLA internal states (if available)
        if self.hidden_states is not None:
            h = self.hidden_states.cpu().numpy().flatten()
            signals[4] = np.linalg.norm(h)
            signals[5] = np.std(h)
            try:
                from scipy.stats import skew
                signals[6] = float(skew(h))
            except:
                signals[6] = 0.0

        # z8-z9: Action uncertainty
        if action_probs is not None:
            # Entropy
            probs = action_probs + 1e-10
            signals[7] = -np.sum(probs * np.log(probs))
            # Max probability
            signals[8] = np.max(probs)

        # z10: Norm violation (assuming max_norm = 1.0)
        signals[9] = max(0, signals[1] - 1.0)

        # z11: Force anomaly (requires force sensors - set to 0 for now)
        signals[10] = 0.0

        # z12: Temporal consistency
        if len(self.action_history) >= 1:
            prev = self.action_history[-1]
            if len(action) == len(prev):
                corr_matrix = np.corrcoef(action.flatten(), prev.flatten())
                if corr_matrix.shape == (2, 2):
                    signals[11] = corr_matrix[0, 1]

        return signals

    def predict(self, observation, language_instruction=None, action_probs=None):
        """
        Main prediction interface.

        Args:
            observation: Robot observation (image, state, etc.)
            language_instruction: Optional language command
            action_probs: Optional action probability distribution

        Returns:
            action: VLA predicted action
            signals: 12D signal vector
            risk_scores: Dict with 300ms, 500ms, 1000ms predictions
            alert_result: State machine result
        """
        # Get VLA action
        with torch.no_grad():
            if language_instruction is not None:
                action = self.vla(observation, language_instruction)
            else:
                action = self.vla(observation)

        # Convert to numpy
        if torch.is_tensor(action):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)

        # Extract signals
        signals = self.extract_signals(action_np, action_probs)

        # Add to history
        self.action_history.append(action_np)
        self.signal_history.append(signals)

        # Predict with SALUS (need full window)
        if len(self.signal_history) >= self.window_size:
            window = np.stack(list(self.signal_history), axis=0)
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.predictor(window_tensor)
                probs = torch.sigmoid(logits[0]).cpu().numpy()

            risk_scores = {
                '300ms': float(probs[0]),
                '500ms': float(probs[1]),
                '1000ms': float(probs[2])
            }
        else:
            risk_scores = {'300ms': 0.0, '500ms': 0.0, '1000ms': 0.0}

        # Update state machine
        primary_risk = risk_scores['500ms']
        alert_result = self.state_machine.update(primary_risk)

        # Track statistics
        self.step_count += 1
        if alert_result['state'] == AlertState.CRITICAL:
            if alert_result.get('state_changed', False):
                self.alert_count += 1

        return action, signals, risk_scores, alert_result

    def should_intervene(self, alert_result):
        """Check if intervention is needed"""
        return alert_result['state'] == AlertState.CRITICAL

    def apply_intervention(self, action, intervention_type='slow_mode'):
        """
        Apply intervention to action.

        Args:
            action: Original VLA action
            intervention_type: 'slow_mode', 'freeze', or 'safe_retreat'

        Returns:
            Modified action
        """
        if intervention_type == 'slow_mode':
            # Scale action by 0.5
            return action * 0.5
        elif intervention_type == 'freeze':
            # Zero action
            return action * 0.0
        elif intervention_type == 'safe_retreat':
            # Reverse last action
            if len(self.action_history) >= 1:
                return -self.action_history[-1] * 0.3
            return action * 0.0
        return action

# ============================================================================
# Terminal GUI using Rich
# ============================================================================

class SALUSTerminalGUI:
    """Real-time terminal interface showing all SALUS metrics"""

    def __init__(self):
        self.console = Console()
        self.layout = Layout()

        # Setup layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=7)
        )

        self.layout["main"].split_row(
            Layout(name="signals", ratio=1),
            Layout(name="predictions", ratio=1)
        )

        # Data queues
        self.signal_queue = queue.Queue(maxsize=100)
        self.prediction_queue = queue.Queue(maxsize=100)
        self.stats_queue = queue.Queue(maxsize=100)

        # Episode stats
        self.episode_count = 0
        self.total_steps = 0
        self.total_alerts = 0
        self.total_interventions = 0
        self.failure_count = 0

    def make_header(self):
        """Create header panel"""
        text = Text()
        text.append("SALUS ", style="bold cyan")
        text.append("Live Monitoring System", style="bold white")
        text.append(f"  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        return Panel(text, style="cyan", box=box.HEAVY)

    def make_signals_panel(self, signals, alert_state):
        """Create real-time signals display"""
        table = Table(title="[bold cyan]12D Signal Vector", box=box.SIMPLE, show_header=True)
        table.add_column("Signal", style="cyan", width=20)
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Bar", width=30)

        signal_names = [
            ("z₁", "Action Volatility"),
            ("z₂", "Action Magnitude"),
            ("z₃", "Action Acceleration"),
            ("z₄", "Trajectory Divergence"),
            ("z₅", "Hidden State Norm"),
            ("z₆", "Hidden State Std"),
            ("z₇", "Hidden State Skew"),
            ("z₈", "Action Entropy"),
            ("z₉", "Max Probability"),
            ("z₁₀", "Norm Violation"),
            ("z₁₁", "Force Anomaly"),
            ("z₁₂", "Temporal Consistency")
        ]

        if signals is not None:
            for i, (idx, name) in enumerate(signal_names):
                val = signals[i]
                # Normalize for bar display (clip to reasonable range)
                normalized = min(max(val, 0.0), 2.0) / 2.0
                bar_len = int(normalized * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)

                # Color based on value
                if val > 1.5:
                    bar_style = "red"
                elif val > 0.8:
                    bar_style = "yellow"
                else:
                    bar_style = "green"

                table.add_row(
                    f"{idx} {name}",
                    f"{val:.4f}",
                    Text(bar, style=bar_style)
                )
        else:
            table.add_row("Waiting for data...", "", "")

        # Add alert state
        state_text = Text("\n")
        if alert_state == AlertState.CRITICAL:
            state_text.append("⚠ CRITICAL ALERT", style="bold red blink")
        elif alert_state == AlertState.WARNING:
            state_text.append("⚡ WARNING", style="bold yellow")
        else:
            state_text.append("✓ NORMAL", style="bold green")

        return Panel(table, subtitle=state_text)

    def make_predictions_panel(self, risk_scores, ema_risk):
        """Create predictions display"""
        table = Table(title="[bold cyan]Multi-Horizon Predictions", box=box.SIMPLE)
        table.add_column("Horizon", style="cyan", width=12)
        table.add_column("Risk", justify="right", style="yellow", width=10)
        table.add_column("Probability", width=40)

        horizons = [
            ("300ms", "300ms"),
            ("500ms", "500ms (primary)"),
            ("1000ms", "1000ms")
        ]

        if risk_scores is not None:
            for key, label in horizons:
                risk = risk_scores.get(key, 0.0)
                bar_len = int(risk * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)

                # Color based on risk
                if risk > 0.7:
                    bar_style = "red"
                elif risk > 0.4:
                    bar_style = "yellow"
                else:
                    bar_style = "green"

                table.add_row(label, f"{risk:.3f}", Text(bar, style=bar_style))
        else:
            table.add_row("Waiting...", "", "")

        # Add EMA smoothed risk
        table.add_row("", "", "")
        if ema_risk is not None:
            ema_bar_len = int(ema_risk * 40)
            ema_bar = "█" * ema_bar_len + "░" * (40 - ema_bar_len)
            ema_style = "red" if ema_risk > 0.4 else "yellow" if ema_risk > 0.35 else "green"
            table.add_row(
                "[bold]EMA Smoothed",
                f"[bold]{ema_risk:.3f}",
                Text(ema_bar, style=ema_style)
            )

        return Panel(table)

    def make_footer(self, stats):
        """Create statistics footer"""
        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")

        if stats:
            table.add_row(
                "Episode", f"{stats.get('episode', 0)}",
                "Total Steps", f"{stats.get('total_steps', 0)}"
            )
            table.add_row(
                "Alerts", f"{stats.get('alerts', 0)}",
                "Interventions", f"{stats.get('interventions', 0)}"
            )
            table.add_row(
                "Failures", f"{stats.get('failures', 0)}",
                "Success Rate", f"{stats.get('success_rate', 0.0):.1%}"
            )

        return Panel(table, title="[bold]Episode Statistics", border_style="cyan")

    def update_display(self):
        """Update all display panels"""
        # Get latest data
        signals = None
        risk_scores = None
        ema_risk = None
        alert_state = AlertState.NORMAL
        stats = None

        try:
            while not self.signal_queue.empty():
                data = self.signal_queue.get_nowait()
                signals = data['signals']
                alert_state = data['alert_state']
                ema_risk = data['ema_risk']
        except queue.Empty:
            pass

        try:
            while not self.prediction_queue.empty():
                risk_scores = self.prediction_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.stats_queue.empty():
                stats = self.stats_queue.get_nowait()
        except queue.Empty:
            pass

        # Update layout
        self.layout["header"].update(self.make_header())
        self.layout["signals"].update(self.make_signals_panel(signals, alert_state))
        self.layout["predictions"].update(self.make_predictions_panel(risk_scores, ema_risk))
        self.layout["footer"].update(self.make_footer(stats))

        return self.layout

    def run(self):
        """Run the terminal GUI (blocking)"""
        with Live(self.layout, console=self.console, refresh_per_second=10) as live:
            while True:
                live.update(self.update_display())
                time.sleep(0.1)

# ============================================================================
# Mock VLA Model (replace with real VLA)
# ============================================================================

class MockVLA:
    """
    Mock VLA for demonstration.

    REPLACE THIS WITH YOUR REAL VLA:
        from openvla import OpenVLA
        vla = OpenVLA.load_pretrained("openvla-7b")
    """

    def __init__(self):
        self.step = 0
        # Simulate transformer structure for hook registration
        self.transformer = type('obj', (object,), {
            'layers': [type('obj', (object,), {})() for _ in range(12)]
        })()

    def __call__(self, observation, language=None):
        """Mock action generation"""
        self.step += 1

        # Simulate normal behavior for first 50 steps
        if self.step < 50:
            action = np.random.randn(7) * 0.1  # Small random actions
        # Simulate failure precursor (increasing volatility)
        elif self.step < 80:
            volatility = (self.step - 50) / 30.0
            action = np.random.randn(7) * (0.1 + volatility * 0.5)
        # Simulate failure
        else:
            action = np.random.randn(7) * 2.0  # Large erratic actions

        return torch.tensor(action, dtype=torch.float32)

# ============================================================================
# Main Orchestration
# ============================================================================

def run_demo(use_isaac_sim=False, model_path='salus_fixed_pipeline.pt'):
    """
    Run SALUS live demo.

    Args:
        use_isaac_sim: If True, connect to Isaac Sim for visualization
        model_path: Path to trained SALUS model
    """
    console = Console()

    console.print("\n[bold cyan]╔══════════════════════════════════════════╗")
    console.print("[bold cyan]║  SALUS Live Monitoring System Demo      ║")
    console.print("[bold cyan]╚══════════════════════════════════════════╝\n")

    # Initialize VLA
    console.print("[yellow]Initializing VLA model...[/yellow]")
    vla = MockVLA()  # Replace with: vla = YourVLA.load_pretrained(...)

    # Wrap with SALUS
    console.print("[yellow]Loading SALUS wrapper...[/yellow]")
    salus = SALUSWrapper(vla, model_path=model_path)

    # Initialize terminal GUI
    console.print("[yellow]Starting terminal GUI...[/yellow]")
    gui = SALUSTerminalGUI()

    # Start GUI in separate thread
    gui_thread = threading.Thread(target=gui.run, daemon=True)
    gui_thread.start()

    console.print("[green]✓ System ready![/green]\n")
    time.sleep(2)

    # Main control loop
    episode = 0
    step = 0
    episode_steps = 0
    episode_failures = 0

    try:
        while True:
            # Mock observation (replace with real robot observation)
            observation = torch.randn(1, 3, 224, 224)
            language = "pick up the mug"

            # Get VLA action + SALUS prediction
            action, signals, risk_scores, alert_result = salus.predict(
                observation, language
            )

            # Check for intervention
            if salus.should_intervene(alert_result):
                action = salus.apply_intervention(action, 'slow_mode')
                gui.total_interventions += 1

            # Update GUI data
            gui.signal_queue.put({
                'signals': signals,
                'alert_state': alert_result['state'],
                'ema_risk': alert_result['risk_ema']
            })
            gui.prediction_queue.put(risk_scores)

            # Update stats
            step += 1
            episode_steps += 1
            gui.total_steps = step
            gui.total_alerts = salus.alert_count

            # Simulate episode end
            if episode_steps >= 100:
                episode += 1
                episode_steps = 0
                gui.episode_count = episode

                # Random failure for demo
                if np.random.rand() < 0.2:
                    gui.failure_count += 1

            success_rate = 1.0 - (gui.failure_count / max(episode, 1))

            gui.stats_queue.put({
                'episode': episode,
                'total_steps': step,
                'alerts': salus.alert_count,
                'interventions': gui.total_interventions,
                'failures': gui.failure_count,
                'success_rate': success_rate
            })

            # Control loop rate (10Hz)
            time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SALUS Live Demo")
    parser.add_argument('--model', type=str, default='salus_fixed_pipeline.pt',
                       help='Path to SALUS model checkpoint')
    parser.add_argument('--isaac-sim', action='store_true',
                       help='Use Isaac Sim for visualization')

    args = parser.parse_args()

    run_demo(use_isaac_sim=args.isaac_sim, model_path=args.model)
