"""
Enhanced Entry/Exit Strategy UI Component

This module provides UI components for displaying enhanced entry/exit strategy information.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

# Import from other modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.scan_result import ScanResult
from trading.enhanced_entry_exit import get_enhanced_strategy
from utils.config import load_config


def create_enhanced_entry_exit_ui(result: ScanResult) -> html.Div:
    """
    Create a UI component for displaying enhanced entry/exit strategy information.
    
    Args:
        result: ScanResult object containing cycle and market data
        
    Returns:
        Dash Div component with enhanced entry/exit strategy information
    """
    if not result.success:
        return html.Div([
            html.H3("Enhanced Entry/Exit Strategy"),
            html.P("Not available for unsuccessful scan.")
        ])
    
    # Get config (with default fallback path)
    try:
        config = load_config("config/config.json")
    except Exception as e:
        config = {
            'analysis': {
                'gap_threshold': 0.01
            }
        }
    
    # Get enhanced strategy recommendations
    enhanced_strategy = get_enhanced_strategy(result, config)
    
    if not enhanced_strategy.get('valid', False):
        return html.Div([
            html.H3("Enhanced Entry/Exit Strategy"),
            html.P(enhanced_strategy.get('message', "Unable to generate enhanced strategy."))
        ])
    
    # Create the component
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H3("Enhanced Entry/Exit Strategy"),
                html.Div([
                    html.Strong("Closing Price: "),
                    html.Span(f"{result.price:.2f}", className="fw-bold text-primary")
                ], className="mt-2 small")
            ]),
            dbc.CardBody([
                html.P(
                    "This analysis evaluates cycle maturity and provides optimized entry/exit recommendations "
                    "based on cycle phases and alignment.",
                    className="text-muted mb-3"
                ),
                
                # Entry condition summary
                dbc.Alert(
                    enhanced_strategy['entry_conditions']['recommendation'],
                    color="success" if enhanced_strategy['entry_conditions']['favorable'] else "warning",
                    className="mb-3"
                ),
                
                # Tabs for different sections
                dbc.Tabs([
                    # Entry Window Tab
                    dbc.Tab([
                        html.Div([
                            html.H5("Entry Window Analysis", className="mt-3"),
                            
                            # Entry quality card
                            dbc.Card([
                                dbc.CardHeader("Entry Quality"),
                                dbc.CardBody([
                                    html.H3(
                                        enhanced_strategy['entry_windows']['entry_quality'],
                                        className=_get_quality_color_class(
                                            enhanced_strategy['entry_windows']['entry_quality']
                                        )
                                    ),
                                    html.P(enhanced_strategy['entry_windows']['description']),
                                    
                                    # Entry window score gauge
                                    dbc.Progress([
                                        dbc.Progress(
                                            value=enhanced_strategy['entry_windows']['score'] * 10,
                                            color="success",
                                            bar=True,
                                            label=f"{enhanced_strategy['entry_windows']['score']:.1f}/10",
                                            style={"height": "20px"}
                                        )
                                    ], style={"height": "20px"}),
                                    
                                    # Cycle phase breakdown
                                    html.Div([
                                        html.P("Cycle Phase Distribution:", className="mt-3 mb-1"),
                                        
                                        dbc.Row([
                                            dbc.Col([
                                                html.Span("Fresh Crossovers: "),
                                                html.Span(
                                                    str(enhanced_strategy['entry_windows']['fresh_crossovers']),
                                                    className="text-success fw-bold"
                                                )
                                            ], width=4),
                                            dbc.Col([
                                                html.Span("Early Cycles: "),
                                                html.Span(
                                                    str(enhanced_strategy['entry_windows']['early_cycles']),
                                                    className="text-success fw-bold"
                                                )
                                            ], width=4),
                                            dbc.Col([
                                                html.Span("Mid Cycles: "),
                                                html.Span(
                                                    str(enhanced_strategy['entry_windows']['mid_cycles']),
                                                    className="text-primary fw-bold"
                                                )
                                            ], width=4),
                                        ]),
                                        
                                        dbc.Row([
                                            dbc.Col([
                                                html.Span("Late Cycles: "),
                                                html.Span(
                                                    str(enhanced_strategy['entry_windows']['late_cycles']),
                                                    className="text-warning fw-bold"
                                                )
                                            ], width=4),
                                            dbc.Col([
                                                html.Span("End Cycles: "),
                                                html.Span(
                                                    str(enhanced_strategy['entry_windows']['end_cycles']),
                                                    className="text-danger fw-bold"
                                                )
                                            ], width=4),
                                        ], className="mt-2"),
                                    ]),
                                ]),
                            ], className="mb-3"),
                            
                            # Cycle maturity table
                            html.H5("Cycle Maturity Analysis"),
                            dbc.Table([
                                html.Thead(html.Tr([
                                    html.Th("Cycle"),
                                    html.Th("Phase"),
                                    html.Th("Completion"),
                                    html.Th("Days Remaining"),
                                    html.Th("Direction"),
                                ])),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(cycle['cycle_length']),
                                        html.Td(
                                            cycle['phase'],
                                            className=_get_phase_color_class(cycle['phase'])
                                        ),
                                        html.Td(f"{cycle['completion_pct']:.1f}%"),
                                        html.Td(cycle['days_remaining']),
                                        html.Td(
                                            "Bullish" if cycle['is_bullish'] else "Bearish",
                                            className="text-success" if cycle['is_bullish'] else "text-danger"
                                        ),
                                    ])
                                    for cycle in enhanced_strategy['cycle_maturity']
                                ]),
                            ], bordered=True, striped=True, hover=True, size="sm", className="mb-3"),
                            
                            # Cycle alignment card
                            dbc.Card([
                                dbc.CardHeader("Cycle Alignment"),
                                dbc.CardBody([
                                    html.H4(enhanced_strategy['alignment_score']['quality']),
                                    html.P(enhanced_strategy['alignment_score']['description']),
                                    
                                    # Alignment score gauge
                                    dbc.Progress([
                                        dbc.Progress(
                                            value=enhanced_strategy['alignment_score']['score'] * 10,
                                            color="success",
                                            bar=True,
                                            label=f"{enhanced_strategy['alignment_score']['score']:.1f}/10",
                                            style={"height": "20px"}
                                        )
                                    ], style={"height": "20px"}),
                                ]),
                            ]),
                        ]),
                    ], label="Entry Window", tab_id="entry-window-tab"),
                    
                    # Position Sizing Tab
                    dbc.Tab([
                        html.Div([
                            html.H5("Position Sizing Recommendations", className="mt-3"),
                            
                            # Position size card
                            dbc.Card([
                                dbc.CardHeader("Recommended Position Size"),
                                dbc.CardBody([
                                    html.H3(f"{enhanced_strategy['position_sizing']['position_pct']:.0f}%"),
                                    html.P(enhanced_strategy['position_sizing']['risk_text']),
                                    
                                    # Position size gauge
                                    dbc.Progress([
                                        dbc.Progress(
                                            value=enhanced_strategy['position_sizing']['position_pct'],
                                            color=_get_position_size_color(
                                                enhanced_strategy['position_sizing']['position_pct']
                                            ),
                                            bar=True,
                                            label=f"{enhanced_strategy['position_sizing']['position_pct']:.0f}%",
                                            style={"height": "20px"}
                                        )
                                    ], style={"height": "20px"}),
                                ]),
                            ], className="mb-3"),
                            
                            # Trade duration card
                            dbc.Card([
                                dbc.CardHeader("Recommended Trade Duration"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Minimum Hold"),
                                            html.H3(
                                                f"{enhanced_strategy['trade_duration']['min_hold']} days",
                                                className="text-info"
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.H5("Optimal Hold"),
                                            html.H3(
                                                f"{enhanced_strategy['trade_duration']['optimal_hold']} days",
                                                className="text-success"
                                            )
                                        ], width=4),
                                        dbc.Col([
                                            html.H5("Maximum Hold"),
                                            html.H3(
                                                f"{enhanced_strategy['trade_duration']['max_hold']} days",
                                                className="text-warning"
                                            )
                                        ], width=4),
                                    ]),
                                ]),
                            ], className="mb-3"),
                            
                            # Enhanced trade guidance
                            html.H5("Enhanced Risk Management"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Entry"),
                                        dbc.CardBody(
                                            html.H3(
                                                f"{enhanced_strategy['enhanced_position_guidance']['entry_price']:.2f}"
                                            )
                                        ),
                                    ]),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Adjusted Stop Loss"),
                                        dbc.CardBody(
                                            html.H3(
                                                f"{enhanced_strategy['enhanced_position_guidance']['adjusted_stop_loss']:.2f}",
                                                className="text-danger"
                                            )
                                        ),
                                    ]),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Adjusted Target"),
                                        dbc.CardBody(
                                            html.H3(
                                                f"{enhanced_strategy['enhanced_position_guidance']['adjusted_target']:.2f}",
                                                className="text-success"
                                            )
                                        ),
                                    ]),
                                ], width=4),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Adjusted Risk"),
                                        dbc.CardBody(
                                            html.H3(
                                                f"{enhanced_strategy['enhanced_position_guidance']['adjusted_risk_pct']:.2f}%",
                                                className="text-danger"
                                            )
                                        ),
                                    ]),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Adjusted Reward"),
                                        dbc.CardBody(
                                            html.H3(
                                                f"{enhanced_strategy['enhanced_position_guidance']['adjusted_reward_pct']:.2f}%",
                                                className="text-success"
                                            )
                                        ),
                                    ]),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("Adjusted R/R Ratio"),
                                        dbc.CardBody(
                                            html.H3(
                                                f"{enhanced_strategy['enhanced_position_guidance']['adjusted_rr_ratio']:.2f}",
                                                className=_get_rr_ratio_color_class(
                                                    enhanced_strategy['enhanced_position_guidance']['adjusted_rr_ratio']
                                                )
                                            )
                                        ),
                                    ]),
                                ], width=4),
                            ]),
                        ]),
                    ], label="Position Sizing", tab_id="position-sizing-tab"),
                    
                    # Strategy Notes Tab
                    dbc.Tab([
                        html.Div([
                            html.H5("Enhanced Strategy Notes", className="mt-3"),
                            
                            dbc.Card([
                                dbc.CardHeader("Key Considerations"),
                                dbc.CardBody([
                                    html.P([
                                        html.Strong("Trade Direction: "),
                                        html.Span(
                                            enhanced_strategy['trade_direction'].capitalize(),
                                            className="text-success" if enhanced_strategy['trade_direction'] == "long" else "text-danger"
                                        ),
                                        html.Span(
                                            f" ({enhanced_strategy['alignment_score']['direction'].capitalize()} Cycles)",
                                            className="text-muted ms-2 small"
                                        )
                                    ]),
                                    
                                    html.P([
                                        html.Strong("Entry Quality: "),
                                        html.Span(
                                            enhanced_strategy['entry_windows']['entry_quality'],
                                            className=_get_quality_color_class(
                                                enhanced_strategy['entry_windows']['entry_quality']
                                            )
                                        )
                                    ]),
                                    
                                    html.P([
                                        html.Strong("Cycle Alignment: "),
                                        html.Span(
                                            enhanced_strategy['alignment_score']['quality'],
                                            className=_get_quality_color_class(
                                                enhanced_strategy['alignment_score']['quality']
                                            )
                                        )
                                    ]),
                                    
                                    html.P([
                                        html.Strong("Position Size: "),
                                        html.Span(
                                            f"{enhanced_strategy['position_sizing']['position_pct']:.0f}% "
                                            f"({enhanced_strategy['position_sizing']['risk_text']})",
                                            className=_get_position_size_text_class(
                                                enhanced_strategy['position_sizing']['position_pct']
                                            )
                                        )
                                    ]),
                                    
                                    html.P([
                                        html.Strong("Trade Duration: "),
                                        html.Span(
                                            f"Optimal {enhanced_strategy['trade_duration']['optimal_hold']} days "
                                            f"(Max {enhanced_strategy['trade_duration']['max_hold']} days)"
                                        )
                                    ]),
                                    
                                    html.P([
                                        html.Strong("Risk/Reward: "),
                                        html.Span(
                                            f"{enhanced_strategy['enhanced_position_guidance']['adjusted_rr_ratio']:.2f}",
                                            className=_get_rr_ratio_color_class(
                                                enhanced_strategy['enhanced_position_guidance']['adjusted_rr_ratio']
                                            )
                                        )
                                    ]),
                                    
                                    # Warnings
                                    html.Div([
                                        html.H5("Warnings:", className="text-danger mt-4"),
                                        html.Ul([
                                            html.Li(warning, className="text-danger")
                                            for warning in enhanced_strategy['entry_conditions']['warnings']
                                        ])
                                    ]) if enhanced_strategy['entry_conditions']['warnings'] else None,
                                ]),
                            ], className="mb-3"),
                            
                            # Optimal Entry Window Explanation
                            dbc.Card([
                                dbc.CardHeader("Optimal Entry Windows Guide"),
                                dbc.CardBody([
                                    html.P("Optimal entry windows based on cycle phase:"),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Fresh Crossover (0-15%): "),
                                                html.Span(
                                                    "Primary entry zone with highest probability",
                                                    className="text-success"
                                                )
                                            ]),
                                        ], width=6),
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Early Cycle (15-25%): "),
                                                html.Span(
                                                    "Secondary entry zone with good probability",
                                                    className="text-success"
                                                )
                                            ]),
                                        ], width=6),
                                    ], className="mb-2"),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Mid Cycle (25-50%): "),
                                                html.Span(
                                                    "Reduced position size, tighter stops",
                                                    className="text-primary"
                                                )
                                            ]),
                                        ], width=6),
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("Late Cycle (50-80%): "),
                                                html.Span(
                                                    "Caution zone - strong momentum only",
                                                    className="text-warning"
                                                )
                                            ]),
                                        ], width=6),
                                    ], className="mb-2"),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Strong("End Cycle (>80%): "),
                                                html.Span(
                                                    "Avoid entry, prepare for potential reversal",
                                                    className="text-danger"
                                                )
                                            ]),
                                        ], width=6),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ], label="Strategy Notes", tab_id="strategy-notes-tab"),
                ], active_tab="entry-window-tab"),
            ]),
        ], className="mb-4"),
    ])


def _get_quality_color_class(quality: str) -> str:
    """Get Bootstrap color class for quality level."""
    quality_classes = {
        "Excellent": "text-success",
        "Very Good": "text-success",
        "Good": "text-primary",
        "Fair": "text-info",
        "Moderate": "text-info",
        "Poor": "text-warning",
        "Weak": "text-warning",
        "Avoid Entry": "text-danger",
        "No Clear Alignment": "text-secondary",
        "Perfect Alignment": "text-success",
        "Strong Alignment": "text-success",
        "Good Alignment": "text-primary",
        "Moderate Alignment": "text-info",
        "Weak Alignment": "text-warning"
    }
    return quality_classes.get(quality, "text-secondary")


def _get_phase_color_class(phase: str) -> str:
    """Get Bootstrap color class for cycle phase."""
    phase_classes = {
        "Fresh Crossover": "text-success fw-bold",
        "Early Cycle": "text-success",
        "Mid Cycle": "text-primary",
        "Late Cycle": "text-warning",
        "End Cycle": "text-danger fw-bold"
    }
    return phase_classes.get(phase, "text-secondary")


def _get_position_size_color(position_pct: float) -> str:
    """Get Bootstrap color for position size progress bar."""
    if position_pct >= 80:
        return "success"
    elif position_pct >= 50:
        return "info"
    elif position_pct >= 25:
        return "warning"
    else:
        return "danger"


def _get_position_size_text_class(position_pct: float) -> str:
    """Get Bootstrap text color class for position size."""
    if position_pct >= 80:
        return "text-success"
    elif position_pct >= 50:
        return "text-info"
    elif position_pct >= 25:
        return "text-warning"
    else:
        return "text-danger"


def _get_rr_ratio_color_class(rr_ratio: float) -> str:
    """Get Bootstrap color class for risk-reward ratio."""
    if rr_ratio >= 3.0:
        return "text-success fw-bold"
    elif rr_ratio >= 2.0:
        return "text-success"
    elif rr_ratio >= 1.5:
        return "text-info"
    elif rr_ratio >= 1.0:
        return "text-primary"
    else:
        return "text-danger"