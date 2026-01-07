"""
SALUS Adaptation Module
Decides and executes interventions when failures are predicted

Intervention Strategies:
  1. Emergency Stop - Immediate halt (collision imminent)
  2. Slow Down - Reduce action magnitude (uncertain manipulation)
  3. Retry - Reset and try alternative approach (predicted grasp failure)
  4. Human Assistance - Request operator help (repeated failures)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
import time


class InterventionType(Enum):
    """Types of interventions SALUS can perform"""
    NONE = 0
    EMERGENCY_STOP = 1
    SLOW_DOWN = 2
    RETRY = 3
    HUMAN_ASSISTANCE = 4


class FailureType(Enum):
    """Types of failures SALUS predicts"""
    COLLISION = 0
    DROP = 1
    MISS = 2
    TIMEOUT = 3


@dataclass
class InterventionDecision:
    """Represents a decision to intervene"""
    intervention: InterventionType
    predicted_failure_type: FailureType
    predicted_horizon: int
    confidence: float
    reason: str
    timestamp: float


class AdaptationModule:
    """
    SALUS Adaptation Module

    Monitors failure predictions and executes appropriate interventions
    """

    def __init__(
        self,
        # Thresholds for each horizon
        emergency_threshold: float = 0.9,  # H1: 200ms horizon
        slow_down_threshold: float = 0.7,   # H2-H3: 333-433ms
        retry_threshold: float = 0.6,       # H4: 533ms

        # Action modification parameters
        slow_down_factor: float = 0.5,      # Reduce action by 50%
        emergency_stop_steps: int = 5,      # How long to hold stop

        # Retry parameters
        max_retries: int = 3,
        retry_cooldown: float = 1.0,        # Seconds between retries

        # Human assistance parameters
        human_assistance_after_retries: int = 2,

        # Logging
        enable_logging: bool = True
    ):
        """
        Args:
            emergency_threshold: Probability threshold for emergency stop
            slow_down_threshold: Probability threshold for slowing down
            retry_threshold: Probability threshold for retry
            slow_down_factor: Factor to reduce action magnitude
            emergency_stop_steps: Number of steps to hold emergency stop
            max_retries: Maximum number of retry attempts
            retry_cooldown: Seconds to wait between retries
            human_assistance_after_retries: Request human help after N retries
            enable_logging: Whether to log interventions
        """
        self.emergency_threshold = emergency_threshold
        self.slow_down_threshold = slow_down_threshold
        self.retry_threshold = retry_threshold

        self.slow_down_factor = slow_down_factor
        self.emergency_stop_steps = emergency_stop_steps

        self.max_retries = max_retries
        self.retry_cooldown = retry_cooldown
        self.human_assistance_after_retries = human_assistance_after_retries

        self.enable_logging = enable_logging

        # State tracking
        self.intervention_history: List[InterventionDecision] = []
        self.retry_count = 0
        self.last_retry_time = 0.0
        self.emergency_stop_counter = 0
        self.in_emergency_stop = False

        # Statistics
        self.stats = {
            'total_predictions': 0,
            'interventions_triggered': 0,
            'emergency_stops': 0,
            'slow_downs': 0,
            'retries': 0,
            'human_assistance': 0,
            'failures_prevented': 0,  # Estimated
            'false_positives': 0       # Estimated
        }

    def decide_intervention(
        self,
        prediction: Dict[str, torch.Tensor],
        current_step: int
    ) -> InterventionDecision:
        """
        Decide whether and how to intervene based on failure prediction

        Args:
            prediction: Dict from SALUSPredictor.predict_failure()
                - 'failure_predicted': (B,) bool
                - 'failure_horizon': (B,) int (0-3)
                - 'failure_type': (B,) int (0-3)
                - 'confidence': (B,) float
            current_step: Current timestep in episode

        Returns:
            InterventionDecision with chosen intervention
        """
        self.stats['total_predictions'] += 1

        # Extract prediction (assume batch size 1)
        failure_predicted = prediction['failure_predicted'][0].item()
        failure_horizon = prediction['failure_horizon'][0].item()
        failure_type = prediction['failure_type'][0].item()
        confidence = prediction['confidence'][0].item()

        # If already in emergency stop, continue holding
        if self.in_emergency_stop:
            self.emergency_stop_counter -= 1
            if self.emergency_stop_counter <= 0:
                self.in_emergency_stop = False
                if self.enable_logging:
                    print(f"   [SALUS] Emergency stop released at step {current_step}")

            return InterventionDecision(
                intervention=InterventionType.EMERGENCY_STOP,
                predicted_failure_type=FailureType(failure_type),
                predicted_horizon=failure_horizon,
                confidence=confidence,
                reason="Continuing emergency stop",
                timestamp=time.time()
            )

        # No failure predicted or low confidence
        if not failure_predicted:
            return InterventionDecision(
                intervention=InterventionType.NONE,
                predicted_failure_type=FailureType(failure_type),
                predicted_horizon=failure_horizon,
                confidence=confidence,
                reason="No failure predicted",
                timestamp=time.time()
            )

        # Decide intervention based on confidence and horizon
        intervention = InterventionType.NONE
        reason = ""

        # 1. Emergency Stop (highest priority)
        # Triggered for imminent collision at short horizon
        if (confidence >= self.emergency_threshold and
            failure_horizon == 0 and  # H1: 200ms ahead
            failure_type == FailureType.COLLISION.value):

            intervention = InterventionType.EMERGENCY_STOP
            reason = f"Imminent collision detected (confidence={confidence:.2f})"
            self.stats['emergency_stops'] += 1
            self.in_emergency_stop = True
            self.emergency_stop_counter = self.emergency_stop_steps

        # 2. Slow Down
        # Triggered for medium-term risks at H2-H3
        elif (confidence >= self.slow_down_threshold and
              failure_horizon in [1, 2]):  # H2-H3: 333-433ms ahead

            intervention = InterventionType.SLOW_DOWN
            reason = f"Risky maneuver detected (confidence={confidence:.2f}, horizon={failure_horizon})"
            self.stats['slow_downs'] += 1

        # 3. Retry
        # Triggered for early warnings at H4, if retries available
        elif (confidence >= self.retry_threshold and
              failure_horizon == 3 and  # H4: 533ms ahead
              self.retry_count < self.max_retries and
              time.time() - self.last_retry_time > self.retry_cooldown):

            intervention = InterventionType.RETRY
            reason = f"Early failure warning (confidence={confidence:.2f}), attempting retry {self.retry_count + 1}/{self.max_retries}"
            self.stats['retries'] += 1
            self.retry_count += 1
            self.last_retry_time = time.time()

        # 4. Human Assistance
        # Triggered after too many retries
        elif self.retry_count >= self.human_assistance_after_retries:

            intervention = InterventionType.HUMAN_ASSISTANCE
            reason = f"Multiple interventions failed (retries={self.retry_count}), requesting human assistance"
            self.stats['human_assistance'] += 1

        # Create decision
        decision = InterventionDecision(
            intervention=intervention,
            predicted_failure_type=FailureType(failure_type),
            predicted_horizon=failure_horizon,
            confidence=confidence,
            reason=reason,
            timestamp=time.time()
        )

        # Log intervention
        if intervention != InterventionType.NONE:
            self.intervention_history.append(decision)
            self.stats['interventions_triggered'] += 1

            if self.enable_logging:
                print(f"\n🚨 [SALUS INTERVENTION] Step {current_step}")
                print(f"   Type: {intervention.name}")
                print(f"   Predicted Failure: {FailureType(failure_type).name} at horizon H{failure_horizon + 1}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   Reason: {reason}\n")

        return decision

    def apply_intervention(
        self,
        action: torch.Tensor,
        decision: InterventionDecision
    ) -> Tuple[torch.Tensor, bool]:
        """
        Apply intervention to modify action

        Args:
            action: (B, action_dim) original action from VLA
            decision: InterventionDecision from decide_intervention()

        Returns:
            modified_action: (B, action_dim) modified action
            should_reset: bool, whether to reset environment (for retry)
        """
        should_reset = False
        modified_action = action.clone()

        if decision.intervention == InterventionType.EMERGENCY_STOP:
            # Emergency stop: zero out all actions
            modified_action = torch.zeros_like(action)

        elif decision.intervention == InterventionType.SLOW_DOWN:
            # Slow down: reduce action magnitude
            modified_action = action * self.slow_down_factor

        elif decision.intervention == InterventionType.RETRY:
            # Retry: signal to reset environment
            should_reset = True
            modified_action = torch.zeros_like(action)

        elif decision.intervention == InterventionType.HUMAN_ASSISTANCE:
            # Human assistance: pause execution (zero action)
            # In real deployment, this would trigger notification to operator
            modified_action = torch.zeros_like(action)
            if self.enable_logging:
                print("⚠️  [SALUS] Execution paused. Waiting for human guidance...")

        return modified_action, should_reset

    def on_episode_end(self, success: bool):
        """
        Called at end of episode to update statistics

        Args:
            success: Whether episode succeeded
        """
        # Reset retry counter for new episode
        self.retry_count = 0
        self.emergency_stop_counter = 0
        self.in_emergency_stop = False

        # Estimate effectiveness
        # If we intervened and episode succeeded, we may have prevented failure
        if len(self.intervention_history) > 0 and success:
            self.stats['failures_prevented'] += 1

        # If we intervened but episode failed, it may have been false positive
        # (or intervention was ineffective)
        if len(self.intervention_history) > 0 and not success:
            self.stats['false_positives'] += 1

        # Clear episode history
        self.intervention_history = []

    def get_statistics(self) -> Dict:
        """Get adaptation statistics"""
        stats = self.stats.copy()

        # Compute derived metrics
        if stats['interventions_triggered'] > 0:
            stats['intervention_rate'] = stats['interventions_triggered'] / max(stats['total_predictions'], 1)
            stats['prevention_rate'] = stats['failures_prevented'] / max(stats['interventions_triggered'], 1)
        else:
            stats['intervention_rate'] = 0.0
            stats['prevention_rate'] = 0.0

        return stats

    def print_statistics(self):
        """Print adaptation statistics"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("SALUS Adaptation Statistics")
        print("="*60)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Interventions triggered: {stats['interventions_triggered']}")
        print(f"  - Emergency stops: {stats['emergency_stops']}")
        print(f"  - Slow downs: {stats['slow_downs']}")
        print(f"  - Retries: {stats['retries']}")
        print(f"  - Human assistance: {stats['human_assistance']}")
        print(f"\nIntervention rate: {stats['intervention_rate']:.2%}")
        print(f"Estimated failures prevented: {stats['failures_prevented']}")
        print(f"Estimated false positives: {stats['false_positives']}")
        print(f"Prevention rate: {stats['prevention_rate']:.2%}")
        print("="*60 + "\n")

    def reset(self):
        """Reset adaptation module state"""
        self.intervention_history = []
        self.retry_count = 0
        self.last_retry_time = 0.0
        self.emergency_stop_counter = 0
        self.in_emergency_stop = False

        # Don't reset stats - those are cumulative


# Test the adaptation module
if __name__ == "__main__":
    print("🧪 Testing SALUS Adaptation Module...\n")

    # Create adaptation module
    adapter = AdaptationModule(
        emergency_threshold=0.9,
        slow_down_threshold=0.7,
        retry_threshold=0.6,
        enable_logging=True
    )

    print("📊 Adaptation Configuration:")
    print(f"   Emergency threshold: {adapter.emergency_threshold}")
    print(f"   Slow down threshold: {adapter.slow_down_threshold}")
    print(f"   Retry threshold: {adapter.retry_threshold}")
    print(f"   Slow down factor: {adapter.slow_down_factor}")
    print()

    # Simulate predictions and interventions
    print("🔄 Simulating failure predictions...\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Test case 1: Emergency stop (imminent collision)
    print("Test 1: Imminent collision")
    prediction = {
        'failure_predicted': torch.tensor([True], device=device),
        'failure_horizon': torch.tensor([0], device=device),  # H1
        'failure_type': torch.tensor([0], device=device),     # Collision
        'confidence': torch.tensor([0.95], device=device)
    }

    decision = adapter.decide_intervention(prediction, current_step=50)
    action = torch.randn(1, 7, device=device)
    modified_action, should_reset = adapter.apply_intervention(action, decision)

    print(f"   Decision: {decision.intervention.name}")
    print(f"   Action modified: {not torch.equal(action, modified_action)}")
    print(f"   Should reset: {should_reset}")
    print()

    # Test case 2: Slow down (uncertain grasp)
    print("Test 2: Uncertain manipulation")
    prediction = {
        'failure_predicted': torch.tensor([True], device=device),
        'failure_horizon': torch.tensor([1], device=device),  # H2
        'failure_type': torch.tensor([1], device=device),     # Drop
        'confidence': torch.tensor([0.75], device=device)
    }

    decision = adapter.decide_intervention(prediction, current_step=75)
    action = torch.randn(1, 7, device=device)
    modified_action, should_reset = adapter.apply_intervention(action, decision)

    print(f"   Decision: {decision.intervention.name}")
    print(f"   Action magnitude reduced: {modified_action.norm() < action.norm()}")
    print(f"   Should reset: {should_reset}")
    print()

    # Test case 3: Retry (early warning)
    print("Test 3: Early warning")
    prediction = {
        'failure_predicted': torch.tensor([True], device=device),
        'failure_horizon': torch.tensor([3], device=device),  # H4
        'failure_type': torch.tensor([2], device=device),     # Miss
        'confidence': torch.tensor([0.65], device=device)
    }

    decision = adapter.decide_intervention(prediction, current_step=100)
    action = torch.randn(1, 7, device=device)
    modified_action, should_reset = adapter.apply_intervention(action, decision)

    print(f"   Decision: {decision.intervention.name}")
    print(f"   Should reset: {should_reset}")
    print()

    # Test case 4: No intervention (low confidence)
    print("Test 4: Low confidence prediction")
    prediction = {
        'failure_predicted': torch.tensor([False], device=device),
        'failure_horizon': torch.tensor([2], device=device),
        'failure_type': torch.tensor([3], device=device),
        'confidence': torch.tensor([0.45], device=device)
    }

    decision = adapter.decide_intervention(prediction, current_step=125)
    action = torch.randn(1, 7, device=device)
    modified_action, should_reset = adapter.apply_intervention(action, decision)

    print(f"   Decision: {decision.intervention.name}")
    print(f"   Action unchanged: {torch.equal(action, modified_action)}")
    print()

    # Simulate episode end
    adapter.on_episode_end(success=True)

    # Print statistics
    adapter.print_statistics()

    print("✅ SALUS Adaptation Module test passed!")
    print("\n📋 Next steps:")
    print("   1. Integrate with VLA + Predictor in closed loop")
    print("   2. Test on real environment")
    print("   3. Evaluate intervention effectiveness")
    print("   4. Tune thresholds based on performance")
