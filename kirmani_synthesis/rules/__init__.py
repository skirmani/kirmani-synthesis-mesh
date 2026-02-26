"""
Quick Wins Rule Engine Module

Automated trading rules that fire when cross-system conditions are met.
"""

from .engine import QuickWinsEngine, Rule, RuleResult

__all__ = ["QuickWinsEngine", "Rule", "RuleResult"]
