"""
SALUS Alert State Machine

Implements production-ready alert logic to eliminate spam:
1. EMA smoothing on risk scores
2. Persistence requirement (N consecutive high scores)
3. Hysteresis (separate on/off thresholds)
4. Cooldown after alerts
5. State tracking (NORMAL → WARNING → CRITICAL)

Target: <1 alert per minute in production
"""

import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Dict, Any


class AlertState(Enum):
    """Alert state levels"""
    NORMAL = 0      # No risk detected
    WARNING = 1     # Risk rising but not critical
    CRITICAL = 2    # Imminent failure - trigger intervention


class SALUSStateMachine:
    """
    Production-ready alert state machine for SALUS.

    Prevents alert spam through:
    - EMA smoothing
    - Persistence checking
    - Hysteresis thresholds
    - Cooldown periods
    """

    def __init__(
        self,
        loop_rate_hz: float = 30.0,
        ema_alpha: float = 0.3,
        persistence_ticks: int = 4,
        threshold_on: float = 0.55,
        threshold_off: float = 0.45,
        warning_threshold: float = 0.50,
        cooldown_seconds: float = 2.0,
        require_drop_before_rearm: bool = True
    ):
        """
        Initialize alert state machine.

        Args:
            loop_rate_hz: Control loop frequency (Hz)
            ema_alpha: Exponential moving average smoothing factor (0-1)
                      Higher = more reactive, Lower = more smooth
            persistence_ticks: Number of consecutive ticks above threshold
                              to enter CRITICAL state
            threshold_on: Risk score threshold to enter CRITICAL (hysteresis high)
            threshold_off: Risk score threshold to exit CRITICAL (hysteresis low)
            warning_threshold: Risk score threshold for WARNING state
            cooldown_seconds: Minimum time between CRITICAL alerts
            require_drop_before_rearm: If True, must drop below threshold_off
                                      before re-arming after cooldown
        """
        self.loop_rate_hz = loop_rate_hz
        self.dt = 1.0 / loop_rate_hz

        # Smoothing
        self.ema_alpha = ema_alpha
        self.risk_ema = 0.0

        # Persistence
        self.persistence_ticks = persistence_ticks
        self.high_risk_buffer = deque(maxlen=persistence_ticks)

        # Hysteresis thresholds
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.warning_threshold = warning_threshold

        # Cooldown
        self.cooldown_ticks = int(cooldown_seconds * loop_rate_hz)
        self.cooldown_counter = 0
        self.require_drop_before_rearm = require_drop_before_rearm
        self.has_dropped_below_off = True

        # State tracking
        self.state = AlertState.NORMAL
        self.prev_state = AlertState.NORMAL
        self.critical_count = 0
        self.total_ticks = 0

        # Metrics
        self.alert_times = []
        self.state_history = []

    def update(self, risk_score_raw: float) -> Dict[str, Any]:
        """
        Update state machine with new risk score.

        Args:
            risk_score_raw: Raw risk score from model [0, 1]

        Returns:
            dict with:
                - state: Current AlertState
                - state_changed: Whether state changed this tick
                - should_intervene: Whether to trigger emergency action
                - risk_ema: Smoothed risk score
                - risk_raw: Raw risk score
                - in_cooldown: Whether in cooldown period
        """
        self.total_ticks += 1

        # 1. Apply EMA smoothing
        if self.total_ticks == 1:
            self.risk_ema = risk_score_raw
        else:
            self.risk_ema = (self.ema_alpha * risk_score_raw +
                           (1 - self.ema_alpha) * self.risk_ema)

        # 2. Update cooldown counter
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            in_cooldown = True
        else:
            in_cooldown = False

        # 3. Track if risk has dropped below off threshold
        if self.risk_ema < self.threshold_off:
            self.has_dropped_below_off = True

        # 4. Check if we're armed (can trigger new alert)
        can_trigger = True
        if in_cooldown:
            can_trigger = False
        if self.require_drop_before_rearm and not self.has_dropped_below_off:
            can_trigger = False

        # 5. Update persistence buffer
        self.high_risk_buffer.append(self.risk_ema > self.threshold_on)

        # 6. Check persistence condition
        persistence_met = (len(self.high_risk_buffer) == self.persistence_ticks and
                          all(self.high_risk_buffer))

        # 7. State transition logic
        self.prev_state = self.state

        if self.state == AlertState.NORMAL:
            if self.risk_ema >= self.warning_threshold:
                self.state = AlertState.WARNING

        elif self.state == AlertState.WARNING:
            if self.risk_ema < self.warning_threshold:
                self.state = AlertState.NORMAL
            elif persistence_met and can_trigger:
                self.state = AlertState.CRITICAL
                self.critical_count += 1
                self.alert_times.append(self.total_ticks)
                # Enter cooldown
                self.cooldown_counter = self.cooldown_ticks
                self.has_dropped_below_off = False

        elif self.state == AlertState.CRITICAL:
            # Exit CRITICAL when risk drops below off-threshold
            if self.risk_ema < self.threshold_off:
                self.state = AlertState.WARNING if self.risk_ema >= self.warning_threshold else AlertState.NORMAL

        # 8. Determine if intervention should occur
        state_changed = (self.state != self.prev_state)
        should_intervene = (self.state == AlertState.CRITICAL and
                          self.prev_state != AlertState.CRITICAL)

        # 9. Record state
        self.state_history.append(self.state)

        return {
            'state': self.state,
            'state_changed': state_changed,
            'should_intervene': should_intervene,
            'risk_ema': self.risk_ema,
            'risk_raw': risk_score_raw,
            'in_cooldown': in_cooldown,
            'persistence_buffer': list(self.high_risk_buffer),
            'can_trigger': can_trigger
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            dict with:
                - total_alerts: Number of CRITICAL states entered
                - alerts_per_minute: Alert rate
                - total_time_seconds: Total elapsed time
                - state_distribution: Fraction of time in each state
        """
        total_time_seconds = self.total_ticks * self.dt
        alerts_per_minute = (self.critical_count / total_time_seconds * 60) if total_time_seconds > 0 else 0

        # Compute state distribution
        state_counts = {
            AlertState.NORMAL: 0,
            AlertState.WARNING: 0,
            AlertState.CRITICAL: 0
        }
        for state in self.state_history:
            state_counts[state] += 1

        state_distribution = {
            state.name: count / len(self.state_history) if self.state_history else 0
            for state, count in state_counts.items()
        }

        return {
            'total_alerts': self.critical_count,
            'alerts_per_minute': alerts_per_minute,
            'total_time_seconds': total_time_seconds,
            'total_ticks': self.total_ticks,
            'state_distribution': state_distribution,
            'alert_times': self.alert_times
        }

    def reset(self):
        """Reset state machine to initial state."""
        self.risk_ema = 0.0
        self.high_risk_buffer.clear()
        self.cooldown_counter = 0
        self.has_dropped_below_off = True
        self.state = AlertState.NORMAL
        self.prev_state = AlertState.NORMAL
        self.critical_count = 0
        self.total_ticks = 0
        self.alert_times = []
        self.state_history = []


def compute_lead_time_from_state_machine(
    state_history: list,
    failure_timestep: int,
    loop_rate_hz: float = 30.0
) -> Optional[float]:
    """
    Compute lead time as time from first CRITICAL state to failure.

    This is the CORRECT lead time measurement (not every tick above threshold).

    Args:
        state_history: List of AlertState values over time
        failure_timestep: Timestep when failure occurred
        loop_rate_hz: Control loop frequency

    Returns:
        Lead time in milliseconds, or None if no CRITICAL state before failure
    """
    # Find first CRITICAL state
    first_critical = None
    for t, state in enumerate(state_history):
        if state == AlertState.CRITICAL:
            first_critical = t
            break

    if first_critical is None:
        return None  # No alert issued

    if first_critical >= failure_timestep:
        return None  # Alert came too late

    lead_time_ticks = failure_timestep - first_critical
    lead_time_ms = (lead_time_ticks / loop_rate_hz) * 1000

    return lead_time_ms


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("\n" + "="*80)
    print("SALUS State Machine Example")
    print("="*80)

    # Create state machine with recommended parameters for 30Hz loop
    sm = SALUSStateMachine(
        loop_rate_hz=30.0,
        ema_alpha=0.3,              # Smooth over ~3 ticks
        persistence_ticks=4,         # Require 4 consecutive high readings (133ms)
        threshold_on=0.55,          # Enter CRITICAL at 0.55
        threshold_off=0.45,         # Exit CRITICAL at 0.45
        warning_threshold=0.50,     # Enter WARNING at 0.50
        cooldown_seconds=2.0,       # 2 second cooldown between alerts
        require_drop_before_rearm=True
    )

    # Simulate risk scores rising then falling
    print("\nSimulating risk score evolution:")
    print("="*80)

    # Pattern: low → spike → drop → spike again
    risk_pattern = (
        [0.3] * 30 +           # 1.0s normal
        [0.52] * 10 +          # 333ms warning (not persistent enough)
        [0.4] * 20 +           # 667ms drop
        [0.56] * 8 +           # 267ms CRITICAL trigger
        [0.58] * 10 +          # Stay high
        [0.4] * 30 +           # Drop, cooldown
        [0.60] * 5             # Try to spike again (should be in cooldown)
    )

    results = []
    for t, risk_raw in enumerate(risk_pattern):
        result = sm.update(risk_raw)
        results.append(result)

        # Print state changes
        if result['state_changed']:
            time_ms = (t / 30.0) * 1000
            print(f"t={time_ms:6.0f}ms: {result['state'].name:8s} "
                  f"(risk_ema={result['risk_ema']:.3f}, "
                  f"raw={result['risk_raw']:.3f})")

        # Print intervention triggers
        if result['should_intervene']:
            time_ms = (t / 30.0) * 1000
            print(f"  🚨 INTERVENTION TRIGGERED at t={time_ms:.0f}ms")

    # Get final metrics
    metrics = sm.get_metrics()
    print("\n" + "="*80)
    print("Final Metrics:")
    print("="*80)
    print(f"Total alerts: {metrics['total_alerts']}")
    print(f"Alerts/min: {metrics['alerts_per_minute']:.2f}")
    print(f"Total time: {metrics['total_time_seconds']:.2f}s")
    print(f"\nState distribution:")
    for state_name, fraction in metrics['state_distribution'].items():
        print(f"  {state_name:8s}: {fraction*100:5.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    times = np.arange(len(risk_pattern)) / 30.0

    # Plot 1: Risk scores
    ax1.plot(times, risk_pattern, 'b-', alpha=0.3, label='Raw risk')
    ax1.plot(times, [r['risk_ema'] for r in results], 'b-', linewidth=2, label='EMA risk')
    ax1.axhline(sm.threshold_on, color='r', linestyle='--', label='Threshold ON (0.55)')
    ax1.axhline(sm.threshold_off, color='orange', linestyle='--', label='Threshold OFF (0.45)')
    ax1.axhline(sm.warning_threshold, color='yellow', linestyle=':', label='Warning (0.50)')
    ax1.set_ylabel('Risk Score')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('SALUS State Machine Example')

    # Plot 2: States
    state_values = [r['state'].value for r in results]
    ax2.fill_between(times, 0, state_values, alpha=0.3, step='post')
    ax2.plot(times, state_values, 'k-', drawstyle='steps-post', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('State')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['NORMAL', 'WARNING', 'CRITICAL'])
    ax2.grid(True, alpha=0.3)

    # Mark intervention points
    for alert_tick in metrics['alert_times']:
        alert_time = alert_tick / 30.0
        ax2.axvline(alert_time, color='r', linestyle='--', alpha=0.5)
        ax2.text(alert_time, 2.1, '🚨', ha='center', fontsize=16)

    plt.tight_layout()
    plt.savefig('state_machine_example.png', dpi=150)
    print(f"\n✓ Saved plot: state_machine_example.png")

    print("\n" + "="*80)
    print("Key Features Demonstrated:")
    print("="*80)
    print("1. EMA smoothing prevents jitter")
    print("2. Persistence requirement (4 ticks) filters brief spikes")
    print("3. Hysteresis (0.55 on / 0.45 off) prevents flapping")
    print("4. Cooldown (2s) prevents alert spam")
    print("5. WARNING state provides advance notice without intervention")
    print("="*80)
